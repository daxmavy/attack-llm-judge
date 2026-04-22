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

As of 2026-04-22 judges run in a separate HTTP server process (on GPU 0 by
default) and this script owns its own GPU for the rewriter. Launch with
`CUDA_VISIBLE_DEVICES=1 python -m scripts.run_icir ...` so the rewriter process
only sees GPU 1; `spawn_judge_server(gpu=0)` pins the judge subprocess to
physical GPU 0.
"""
import argparse
import gc
import hashlib
import json
import os
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

from config.models import REWRITER, JUDGE_REGISTRY, FOLDS, require_config
from judge.http_client import JudgeHTTP, spawn_judge_server
from stop_signal import announce as stop_announce, check_stop_signal, clear_stop_file

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
DATASET = "/data/shil6647/attack-llm-judge/grpo_run/controversial_40_3fold.json"
DEFAULT_STOP_DIR = "/data/shil6647/attack-llm-judge/grpo_run/stop_markers"


def load_eval():
    d = json.loads(Path(DATASET).read_text())
    return [r for r in d["rows"] if r.get("split") == "eval"]


def main():
    require_config()
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
    ap.add_argument("--judge-endpoint", type=str, default=None,
                    help="Reuse an already-running judge server instead of "
                         "spawning a new one.")
    ap.add_argument("--judge-port", type=int, default=8127)
    ap.add_argument("--judge-gpu", type=int, default=0,
                    help="Physical GPU for the spawned judge server.")
    ap.add_argument("--stop-file", type=str, default=None,
                    help="Graceful-stop marker path. Touching triggers a clean "
                         "exit after the current (fold × criterion) loop body. "
                         f"Default: {DEFAULT_STOP_DIR}/run_icir.STOP")
    ap.add_argument("--resume-skip-existing", action="store_true",
                    help="Skip (fold, criterion) combinations where --method-tag "
                         "rewrites already exist for every eval-set document.")
    args = ap.parse_args()

    stop_file = args.stop_file or f"{DEFAULT_STOP_DIR}/run_icir.STOP"
    Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
    clear_stop_file(stop_file)
    stop_announce(stop_file)

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
        # Rewriter now owns its own GPU (judges are on a separate GPU via the
        # HTTP server). Can afford a much larger share than the 2026-04-21
        # co-located pattern.
        s = mid.lower()
        if "32b" in s:
            return 0.75
        if "24b" in s:
            return 0.70
        if "14b" in s or "13b" in s or "12b" in s:
            return 0.65
        if "7b" in s or "8b" in s or "9b" in s:
            return 0.50
        return 0.40
    rew_mu = _rew_mem_util(rewriter)
    print(f"[{time.strftime('%H:%M:%S')}] loading rewriter {rewriter} (gpu_mem_util={rew_mu})", flush=True)
    rew_llm = LLM(model=rewriter, dtype="bfloat16", gpu_memory_utilization=rew_mu,
                   max_model_len=3072, enforce_eager=True, download_dir="/data/shil6647/attack-llm-judge/hf_cache")
    rew_sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=400, n=1)

    # Judge server — spawn or reuse.
    if args.judge_endpoint:
        judge_endpoint = args.judge_endpoint
        judge_server_proc = None
        print(f"[{time.strftime('%H:%M:%S')}] reusing judge server at {judge_endpoint}",
              flush=True)
    else:
        log_path = "/data/shil6647/attack-llm-judge/grpo_run/icir_judge_server.log"
        judge_server_proc, judge_endpoint = spawn_judge_server(
            port=args.judge_port, gpu=args.judge_gpu, log_path=log_path,
        )

    config = {"method": "icir", "n_iter": args.n_iter, "temperature": args.temperature}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    conn = sqlite3.connect(DB)

    for fold in args.folds:
        in_panel = args.in_panel_override if args.in_panel_override else FOLDS[fold]["in_panel"]
        # Load 2 judges for this fold (only on first criterion; reuse across criteria via rubric swap)
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold}: loading judges {in_panel}", flush=True)
        judges = [JudgeHTTP(name=slug, rubric=args.criteria[0],
                             endpoint=judge_endpoint) for slug in in_panel]

        for cri in args.criteria:
            if args.resume_skip_existing:
                done_ids = set(r[0] for r in conn.execute(
                    "SELECT DISTINCT source_doc_id FROM attack_rewrites "
                    "WHERE method=? AND fold=? AND criterion=? AND rewriter_model=?",
                    (args.method_tag, fold, cri, rewriter),
                ).fetchall())
                if len(done_ids) >= len(eval_rows):
                    print(f"  skip-existing: fold {fold} × {cri} already complete "
                          f"({len(done_ids)}/{len(eval_rows)})", flush=True)
                    continue
            for j in judges:
                if j.rubric_name != cri:
                    j.set_rubric(cri)
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

                # Score with both judges (need reasoning; JudgeHTTP.score_full
                # returns both scores and parsed reasoning strings).
                if it < args.n_iter - 1:
                    judge_results = [
                        j.score_full(propositions, rewrites) for j in judges
                    ]
                    for k in range(len(rewrites)):
                        scores_k = []
                        reasons_k = []
                        for judge, res in zip(judges, judge_results):
                            # score_full falls back to 50.0 server-side when
                            # parse fails — treat that as "no signal" here so
                            # we keep the ICRH semantics (ignore unparseable).
                            s = res["scores"][k]
                            rs = res["reasonings"][k]
                            if s != 50.0 or rs:
                                scores_k.append(float(s))
                            if rs:
                                reasons_k.append(f"{judge.name}: {rs}")
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

            if check_stop_signal(stop_file):
                print(f"[{time.strftime('%H:%M:%S')}] STOP detected after fold {fold} × {cri} — "
                      f"exiting. Re-run with --resume-skip-existing to continue.", flush=True)
                break

        # Free judges server-side before moving to next fold so GPU 0 doesn't
        # accumulate engines when the next fold pulls different slugs.
        for j in judges:
            try:
                j.unload()
            except Exception as e:
                print(f"  warn: unload({j.name}) failed: {e!r}", flush=True)
        del judges
        gc.collect()
        torch.cuda.empty_cache()

        if check_stop_signal(stop_file):
            print(f"[{time.strftime('%H:%M:%S')}] STOP detected after fold {fold} — "
                  f"exiting outer loop.", flush=True)
            break

    del rew_llm
    gc.collect()
    torch.cuda.empty_cache()
    conn.close()

    if judge_server_proc is not None:
        print(f"[{time.strftime('%H:%M:%S')}] shutting down judge server "
              f"(pid={judge_server_proc.pid})...", flush=True)
        try:
            JudgeHTTP.request_shutdown(judge_endpoint, timeout=10.0)
        except Exception as e:
            print(f"  warn: /shutdown failed: {e!r}", flush=True)
            try:
                judge_server_proc.terminate()
            except Exception:
                pass
        try:
            judge_server_proc.wait(timeout=60)
        except Exception:
            try:
                judge_server_proc.kill()
            except Exception:
                pass
    print(f"[{time.strftime('%H:%M:%S')}] ICIR done", flush=True)


if __name__ == "__main__":
    main()
