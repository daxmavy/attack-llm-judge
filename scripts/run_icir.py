"""ICIR (Iterative Contextual Refinement) attack.

Per (fold, criterion):
  - Use the SAME 2 in-panel judges as the GRPO panel (so the attacker uses the
    same threat model as the RL run). The held-out judge is NOT used during
    refinement — it scores the final rewrite at eval time.
  - For each of 714 eval paragraphs:
      iter 0: rewrite via `lit_informed_tight` template (criterion-aware)
      iter k>=1: refine via ICIR_ITER_TEMPLATE using prev rewrite + mean
                 in-panel judge score + concatenated judge reasoning.
  - Save final iter's rewrite to attack_rewrites with method='icir',
    fold=N, criterion=X. Intermediate iterations are NOT saved.

Defaults: 4 total iterations (= 1 seed + 3 refinements), per ICRH paper.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ENV = "/home/shil6647/attack-llm-judge/.env"
if os.path.exists(ENV):
    for line in open(ENV):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v
os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")

sys.path.insert(0, "/data/shil6647/attack-llm-judge/grpo_run")
sys.path.insert(0, "/home/shil6647/attack-llm-judge")

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
DATASET = "/data/shil6647/attack-llm-judge/grpo_run/controversial_40_3fold.json"
REWRITER = "Qwen/Qwen2.5-1.5B-Instruct"

FOLDS = {
    1: {"in_panel": ["qwen95b", "llama8b"], "held_out": "gemma9b"},
    2: {"in_panel": ["qwen95b", "gemma9b"], "held_out": "llama8b"},
    3: {"in_panel": ["llama8b", "gemma9b"], "held_out": "qwen95b"},
}

# Reuse the JSON pattern logic from run_pilot_len_pen.py but extract reasoning too.
_SCORE_RE_JSON = re.compile(r"\{[^{}]*\"score\"\s*:\s*\d+(?:\.\d+)?[^{}]*\}", re.S)
_KV_RE = re.compile(r"\"score\"\s*:\s*(\d+(?:\.\d+)?)")
_REASON_RE = re.compile(r"\"reasoning\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"", re.S)


def parse_score_and_reasoning(text):
    if not text:
        return None, ""
    m = _SCORE_RE_JSON.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return float(obj["score"]), str(obj.get("reasoning", ""))[:600]
        except Exception:
            pass
    score = None
    m = _KV_RE.search(text)
    if m:
        try: score = float(m.group(1))
        except Exception: pass
    reasoning = ""
    m = _REASON_RE.search(text)
    if m:
        reasoning = m.group(1).encode().decode("unicode_escape", errors="replace")[:600]
    return score, reasoning


def load_eval():
    d = json.loads(Path(DATASET).read_text())
    return [r for r in d["rows"] if r.get("split") == "eval"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--criteria", nargs="+", default=["clarity", "informativeness"])
    ap.add_argument("--n-iter", type=int, default=4,
                    help="total iterations including iter 0 seed (default 4)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--in-panel-override", nargs="+", default=None,
                    help="override FOLDS[fold].in_panel with this judge list; applied to every fold")
    ap.add_argument("--method-tag", type=str, default="icir",
                    help="method tag written to attack_rewrites (default 'icir')")
    ap.add_argument("--rewriter", type=str, default=None,
                    help=f"override rewriter base model (default: {REWRITER})")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import torch
    from transformers import AutoTokenizer
    from rewriters.rewrite_prompts import (
        SYSTEM_PROMPT,
        LIT_INFORMED_TIGHT_USER_TEMPLATE,
        LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE,
        ICIR_ITER_TEMPLATE,
        length_bounds,
    )
    from run_pilot_len_pen import JUDGE_REGISTRY, JudgeVLLM, RUBRICS

    eval_rows = load_eval()
    print(f"[{time.strftime('%H:%M:%S')}] eval rows: {len(eval_rows)}", flush=True)

    rewriter = args.rewriter or REWRITER
    rew_tok = AutoTokenizer.from_pretrained(rewriter, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                             token=os.environ.get("HF_TOKEN"))
    if rew_tok.pad_token is None:
        rew_tok.pad_token = rew_tok.eos_token

    def _sys_ok():
        try:
            rew_tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True)
            return True
        except Exception:
            return False
    sys_ok = _sys_ok()

    def _rew_mem_util(mid: str) -> float:
        s = mid.lower()
        if "14b" in s or "13b" in s:
            return 0.45
        if "7b" in s or "8b" in s or "9b" in s:
            return 0.25
        return 0.18
    rew_mu = _rew_mem_util(rewriter)
    print(f"[{time.strftime('%H:%M:%S')}] loading rewriter {rewriter} (gpu_mem_util={rew_mu})", flush=True)
    rew_llm = LLM(model=rewriter, dtype="bfloat16", gpu_memory_utilization=rew_mu,
                   max_model_len=3072, enforce_eager=True, download_dir="/data/shil6647/attack-llm-judge/hf_cache")
    rew_sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=400, n=1)

    config = {"method": "icir", "n_iter": args.n_iter, "temperature": args.temperature}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    conn = sqlite3.connect(DB)

    for fold in args.folds:
        in_panel = args.in_panel_override if args.in_panel_override else FOLDS[fold]["in_panel"]
        # Load 2 judges for this fold (only on first criterion; reuse across criteria via rubric swap)
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold}: loading judges {in_panel}", flush=True)
        judges = [JudgeVLLM(*JUDGE_REGISTRY[j], rubric=args.criteria[0]) for j in in_panel]

        for cri in args.criteria:
            for j in judges:
                j.rubric_name = cri
                j.rubric_text = RUBRICS[cri]
            print(f"[{time.strftime('%H:%M:%S')}] fold {fold} × {cri}: starting ICIR ({len(eval_rows)} paragraphs, {args.n_iter} iters)", flush=True)
            t_start = time.time()

            # Per-paragraph state. lists aligned with eval_rows.
            current_text = [r["text"] for r in eval_rows]  # iter 0 input is the original
            propositions = [r["proposition"] for r in eval_rows]
            doc_ids = [r["document_id"] for r in eval_rows]
            prev_rewrite = [None] * len(eval_rows)
            prev_score = [None] * len(eval_rows)
            prev_reasoning = [""] * len(eval_rows)

            for it in range(args.n_iter):
                # Build prompts for this iter
                prompts = []
                for k, r in enumerate(eval_rows):
                    orig, lo, hi = length_bounds(r["text"], tolerance=0.10)
                    if it == 0:
                        # iter 0: lit_informed_tight seed (criterion-aware)
                        tmpl = (LIT_INFORMED_TIGHT_INFORMATIVENESS_TEMPLATE
                                if cri == "informativeness" else LIT_INFORMED_TIGHT_USER_TEMPLATE)
                        user_msg = tmpl.format(proposition=r["proposition"], paragraph=r["text"],
                                               orig_words=orig, min_words=lo, max_words=hi)
                    else:
                        user_msg = ICIR_ITER_TEMPLATE.format(
                            proposition=r["proposition"], paragraph=r["text"],
                            orig_words=orig, min_words=lo, max_words=hi,
                            prev_rewrite=prev_rewrite[k] or r["text"],
                            prev_clarity=int(prev_score[k] if prev_score[k] is not None else 50),
                            judge_reasoning=prev_reasoning[k] or "(no detailed feedback)",
                        )
                    if sys_ok:
                        chat = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg}]
                    else:
                        chat = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_msg}]
                    prompts.append(rew_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

                outs = rew_llm.generate(prompts, rew_sp, use_tqdm=False)
                rewrites = [o.outputs[0].text.strip() for o in outs]

                # Score with both judges (need reasoning, so call vLLM directly to get raw text)
                if it < args.n_iter - 1:
                    judge_outs = []
                    for j in judges:
                        prompts_j = [j._build_prompt(p, r) for p, r in zip(propositions, rewrites)]
                        outs_j = j.llm.generate(prompts_j, j.sp, use_tqdm=False)
                        judge_outs.append([o.outputs[0].text for o in outs_j])
                    # Aggregate
                    for k in range(len(rewrites)):
                        scores_k = []
                        reasons_k = []
                        for ji, judge in enumerate(judges):
                            s, rs = parse_score_and_reasoning(judge_outs[ji][k])
                            if s is not None: scores_k.append(s)
                            if rs: reasons_k.append(f"{judge.name}: {rs}")
                        prev_rewrite[k] = rewrites[k]
                        prev_score[k] = sum(scores_k) / len(scores_k) if scores_k else 50.0
                        prev_reasoning[k] = " | ".join(reasons_k)[:1200] or "(no feedback)"
                else:
                    # final iter — no need to score for the loop; just save the rewrite
                    pass

                print(f"  fold {fold} × {cri} iter {it}/{args.n_iter-1} done", flush=True)

            # Save final rewrites (the last iter's outputs)
            for k, r in enumerate(eval_rows):
                rid = f"{args.method_tag}_f{fold}_{cri}_{r['document_id']}"
                text = rewrites[k]
                conn.execute("""
                    INSERT OR REPLACE INTO attack_rewrites
                    (rewrite_id, source_doc_id, method, fold, criterion, config_json,
                     rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (rid, r["document_id"], args.method_tag, fold, cri,
                      json.dumps({**config, "config_hash": config_hash, "fold": fold, "criterion": cri,
                                  "in_panel": in_panel, "rewriter": rewriter}),
                      rewriter, json.dumps([f"judge_{j}" for j in in_panel]),
                      text, len(text.split()),
                      json.dumps({"final_iter_index": args.n_iter - 1})))
            conn.commit()
            dt = (time.time() - t_start) / 60
            print(f"  fold {fold} × {cri}: {len(eval_rows)} icir rewrites saved in {dt:.1f} min", flush=True)

        # Free judges before moving to next fold
        for j in judges:
            del j
        gc.collect()
        torch.cuda.empty_cache()

    del rew_llm
    gc.collect()
    torch.cuda.empty_cache()
    conn.close()
    print(f"[{time.strftime('%H:%M:%S')}] ICIR done", flush=True)


if __name__ == "__main__":
    main()
