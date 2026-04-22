"""Mission runner: feedback-free attacks + BoN generation/scoring + DB persistence.

Subcommands:
    feedback_free       generate naive / lit_informed_tight / rubric_aware on eval set (fold-independent)
    bon_generate        generate K=16 candidates per eval paragraph (fold-independent)
    bon_score FOLD      score stored bon_candidate rewrites with FOLD's 2 in-panel judges; pick argmax → bon_panel/foldN

All output goes to paragraphs.db tables `attack_rewrites` and `attack_judge_scores`.
Uses local vLLM via the same `JudgeVLLM` class as RL training (run_pilot_len_pen.py).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# .env first (HF_TOKEN etc)
ENV_FILE = "/home/shil6647/attack-llm-judge/.env"
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v
os.environ.setdefault("HF_HOME", "/data/shil6647/attack-llm-judge/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/data/shil6647/attack-llm-judge/vllm_cache")

sys.path.insert(0, "/data/shil6647/attack-llm-judge/grpo_run")
sys.path.insert(0, "/home/shil6647/attack-llm-judge")

import sqlite3

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
DATASET = "/data/shil6647/attack-llm-judge/grpo_run/controversial_40_3fold.json"
REWRITER = "Qwen/Qwen2.5-1.5B-Instruct"

FOLDS = {
    1: {"in_panel": ["qwen95b", "llama8b"], "held_out": "gemma9b"},
    2: {"in_panel": ["qwen95b", "gemma9b"], "held_out": "llama8b"},
    3: {"in_panel": ["llama8b", "gemma9b"], "held_out": "qwen95b"},
}


def _rewriter_short_name(model_id: str) -> str:
    """Short tag for a rewriter base model, used in rewrite_id and HF repo names."""
    s = model_id.lower()
    if "qwen2.5-1.5b" in s: return "qwen25-15b"
    if "qwen3-14b" in s: return "qwen3-14b"
    if "lfm2.5-1.2b" in s: return "lfm25-12b"
    if "gemma-3-1b" in s: return "gemma3-1b"
    return s.replace("/", "-").replace(".", "-")


def _rewriter_vllm_mem_util(model_id: str) -> float:
    """Auto-pick gpu_memory_utilization for rewriter vLLM based on model size.

    Qwen2.5-1.5B (~3 GB bf16) → 0.15 budget easily fits weights+KV.
    Qwen3-14B (~28 GB bf16) → 0.55 budget = 44 GB / 80 GB A100 covers
    weights + KV headroom for the K=16 BoN rollout.
    """
    s = model_id.lower()
    if "14b" in s or "13b" in s:
        return 0.55
    if "7b" in s or "8b" in s:
        return 0.35
    return 0.20


def _supports_system_role(tok):
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True)
        return True
    except Exception:
        return False


def load_eval_paragraphs():
    d = json.loads(Path(DATASET).read_text())
    return [r for r in d["rows"] if r["split"] == "eval"]


def make_rewrite_id(doc_id, method, config_hash, fold=None, idx=None):
    parts = [doc_id, method, str(config_hash), str(fold), str(idx)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def insert_rewrite(conn, **kw):
    cols = ["rewrite_id", "source_doc_id", "method", "fold", "config_json",
            "rewriter_model", "judge_panel_json", "text", "word_count", "run_metadata_json"]
    conn.execute(
        f"INSERT OR REPLACE INTO attack_rewrites ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [kw.get(c) for c in cols]
    )


def insert_judge_score(conn, **kw):
    cols = ["rewrite_id", "judge_slug", "criterion", "score", "reasoning"]
    conn.execute(
        f"INSERT OR REPLACE INTO attack_judge_scores ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [kw.get(c) for c in cols]
    )


# ---------- feedback-free ----------
def cmd_feedback_free(args):
    from vllm import LLM, SamplingParams
    from rewriters.rewrite_prompts import build_rewrite_prompt, SYSTEM_PROMPT
    from transformers import AutoTokenizer

    rewriter = args.rewriter or REWRITER
    short = _rewriter_short_name(rewriter)
    eval_rows = load_eval_paragraphs()
    print(f"[{time.strftime('%H:%M:%S')}] loaded {len(eval_rows)} eval paragraphs", flush=True)

    tok = AutoTokenizer.from_pretrained(rewriter, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                          token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM rewriter {rewriter}", flush=True)
    llm = LLM(model=rewriter, dtype="bfloat16",
              gpu_memory_utilization=_rewriter_vllm_mem_util(rewriter),
              max_model_len=3072, enforce_eager=True, download_dir="/data/shil6647/attack-llm-judge/hf_cache")

    sys_ok = _supports_system_role(tok)
    sp = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=400, n=1)
    config = {"temperature": 0.7, "max_tokens": 400, "criterion": args.criterion, "rewriter": rewriter}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    methods = args.methods or ["naive", "lit_informed_tight", "rubric_aware"]
    conn = sqlite3.connect(DB)

    for method in methods:
        # naive is criterion-agnostic; skip if already have a clarity version (reuse same rewrite)
        if method == "naive" and args.criterion == "informativeness":
            existing = conn.execute(
                "SELECT COUNT(*) FROM attack_rewrites WHERE method='naive' AND criterion='clarity' AND rewriter_model=?",
                (rewriter,)
            ).fetchone()[0]
            if existing >= len(eval_rows):
                print(f"[{time.strftime('%H:%M:%S')}] skipping naive for informativeness: reuse clarity rewrites ({rewriter})", flush=True)
                continue
        print(f"[{time.strftime('%H:%M:%S')}] generating {method} (criterion={args.criterion}, rewriter={short})...", flush=True)
        prompts_text = []
        for r in eval_rows:
            user = build_rewrite_prompt(method, r["proposition"], r["text"], criterion=args.criterion)
            if sys_ok:
                chat = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
            else:
                chat = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}]
            prompts_text.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        t0 = time.time()
        outs = llm.generate(prompts_text, sp, use_tqdm=True)
        t_gen = time.time() - t0
        print(f"  generated {len(outs)} in {t_gen:.1f}s ({len(outs)/t_gen:.1f}/s)", flush=True)

        for r, o in zip(eval_rows, outs):
            text = o.outputs[0].text.strip()
            rid = make_rewrite_id(r["document_id"], method, f"{short}_{config_hash}")
            conn.execute("""
                INSERT OR REPLACE INTO attack_rewrites
                (rewrite_id, source_doc_id, method, fold, criterion, config_json,
                 rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
                VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, ?)
            """, (rid, r["document_id"], method, args.criterion,
                   json.dumps({**config, "method": method, "config_hash": config_hash}),
                   rewriter, text, len(text.split()),
                   json.dumps({"completion_tokens": len(o.outputs[0].token_ids)})))
        conn.commit()
        print(f"  saved {len(outs)} rows to attack_rewrites", flush=True)

    conn.close()
    del llm
    gc.collect()
    print(f"[{time.strftime('%H:%M:%S')}] feedback_free done", flush=True)


# ---------- BoN generate (once, fold-independent) ----------
def cmd_bon_generate(args):
    from vllm import LLM, SamplingParams
    from rewriters.rewrite_prompts import build_rewrite_prompt, SYSTEM_PROMPT
    from transformers import AutoTokenizer

    rewriter = args.rewriter or REWRITER
    short = _rewriter_short_name(rewriter)
    eval_rows = load_eval_paragraphs()
    K = args.k

    tok = AutoTokenizer.from_pretrained(rewriter, cache_dir="/data/shil6647/attack-llm-judge/hf_cache",
                                          token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM rewriter for BoN K={K} ({rewriter})", flush=True)
    llm = LLM(model=rewriter, dtype="bfloat16",
              gpu_memory_utilization=_rewriter_vllm_mem_util(rewriter),
              max_model_len=3072, enforce_eager=True, download_dir="/data/shil6647/attack-llm-judge/hf_cache")

    sys_ok = _supports_system_role(tok)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, n=K)
    config = {"K": K, "temperature": 1.0, "max_tokens": 400, "method": "bon_candidate",
              "criterion": args.criterion, "rewriter": rewriter}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    print(f"[{time.strftime('%H:%M:%S')}] generating {len(eval_rows)} × K={K} candidates (criterion={args.criterion})", flush=True)
    prompts_text = []
    for r in eval_rows:
        user = build_rewrite_prompt("bon_panel", r["proposition"], r["text"], criterion=args.criterion)
        if sys_ok:
            chat = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        else:
            chat = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user}]
        prompts_text.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    t0 = time.time()
    outs = llm.generate(prompts_text, sp, use_tqdm=True)
    t_gen = time.time() - t0
    n_total = sum(len(o.outputs) for o in outs)
    print(f"  generated {n_total} candidates in {t_gen:.1f}s ({n_total/t_gen:.1f}/s)", flush=True)

    conn = sqlite3.connect(DB)
    for r, o in zip(eval_rows, outs):
        for k_idx, cand in enumerate(o.outputs):
            text = cand.text.strip()
            rid = make_rewrite_id(r["document_id"], "bon_candidate", f"{short}_{config_hash}", idx=k_idx)
            conn.execute("""
                INSERT OR REPLACE INTO attack_rewrites
                (rewrite_id, source_doc_id, method, fold, criterion, config_json,
                 rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
                VALUES (?, ?, 'bon_candidate', NULL, ?, ?, ?, NULL, ?, ?, ?)
            """, (rid, r["document_id"], args.criterion,
                   json.dumps({**config, "candidate_idx": k_idx, "config_hash": config_hash}),
                   rewriter, text, len(text.split()),
                   json.dumps({"completion_tokens": len(cand.token_ids)})))
    conn.commit()
    conn.close()
    del llm
    gc.collect()
    print(f"[{time.strftime('%H:%M:%S')}] bon_generate done; saved {n_total} candidates", flush=True)


# ---------- BoN score per fold ----------
def cmd_bon_score(args):
    """For the given fold, score stored bon_candidate rows with that fold's 2 in-panel judges,
    pick argmax per source paragraph, save as method='bon_panel' fold=FOLD."""
    from run_pilot_len_pen import JUDGE_REGISTRY, JudgeVLLM

    fold = args.fold
    fold_spec = FOLDS[fold]
    in_panel_keys = fold_spec["in_panel"]
    print(f"[{time.strftime('%H:%M:%S')}] BoN scoring fold {fold}: in_panel={in_panel_keys}", flush=True)

    rewriter = args.rewriter or REWRITER
    short = _rewriter_short_name(rewriter)

    # Load the candidates for this criterion AND rewriter
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT r.rewrite_id, r.source_doc_id, r.text, p.proposition, r.config_json
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method='bon_candidate' AND r.criterion=? AND r.rewriter_model=?
    """, (args.criterion, rewriter)).fetchall()
    print(f"  loaded {len(rows)} candidates (criterion={args.criterion}, rewriter={rewriter}) from DB", flush=True)
    if not rows:
        print(f"  no candidates for criterion={args.criterion} × rewriter={rewriter}! run bon_generate first", flush=True)
        return

    judges = []
    for k in in_panel_keys:
        spec = JUDGE_REGISTRY[k]
        print(f"  loading judge {spec[0]} (rubric={args.criterion})", flush=True)
        judges.append(JudgeVLLM(*spec, rubric=args.criterion))

    props = [r[3] for r in rows]
    texts = [r[2] for r in rows]
    score_per_judge = {}
    for j in judges:
        t0 = time.time()
        s = j.score(props, texts)
        print(f"  {j.name} scored {len(s)} in {(time.time()-t0)/60:.1f} min", flush=True)
        score_per_judge[j.name] = s

    # Save per-judge scores to DB (use current criterion)
    for i, r in enumerate(rows):
        for jn, scores in score_per_judge.items():
            insert_judge_score(conn, rewrite_id=r[0], judge_slug=jn,
                                criterion=args.criterion, score=scores[i], reasoning=None)
    conn.commit()

    # Aggregate: per source_doc_id, mean across judges, pick argmax candidate
    by_doc = {}
    for i, r in enumerate(rows):
        doc_id = r[1]
        ej = sum(score_per_judge[jn][i] for jn in score_per_judge) / len(score_per_judge)
        by_doc.setdefault(doc_id, []).append((r[0], ej, r[2], i))

    saved = 0
    for doc_id, cands in by_doc.items():
        cands.sort(key=lambda x: x[1], reverse=True)
        top_rid, top_ej, top_text, top_i = cands[0]
        config = {"K": len(cands), "fold": fold, "in_panel": in_panel_keys,
                  "selected_candidate_rid": top_rid, "ej_score": top_ej,
                  "criterion": args.criterion}
        rid_winner = make_rewrite_id(doc_id, "bon_panel", f"{short}_fold{fold}_{args.criterion}")
        conn.execute("""
            INSERT OR REPLACE INTO attack_rewrites
            (rewrite_id, source_doc_id, method, fold, criterion, config_json,
             rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
            VALUES (?, ?, 'bon_panel', ?, ?, ?, ?, ?, ?, ?, ?)
        """, (rid_winner, doc_id, fold, args.criterion,
               json.dumps(config), rewriter, json.dumps(in_panel_keys),
               top_text, len(top_text.split()),
               json.dumps({"selected_from_candidate_id": top_rid})))
        saved += 1
    conn.commit()
    conn.close()
    print(f"[{time.strftime('%H:%M:%S')}] bon_score fold {fold} done; saved {saved} winners", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    common_criterion = lambda p: p.add_argument(
        "--criterion", choices=["clarity", "informativeness"], default="clarity")
    common_rewriter = lambda p: p.add_argument(
        "--rewriter", type=str, default=None,
        help=f"override rewriter base model (default: {REWRITER})")

    p_ff = sp.add_parser("feedback_free")
    p_ff.add_argument("--methods", nargs="+", default=None)
    common_criterion(p_ff)
    common_rewriter(p_ff)
    p_ff.set_defaults(func=cmd_feedback_free)

    p_bg = sp.add_parser("bon_generate")
    p_bg.add_argument("--k", type=int, default=16)
    common_criterion(p_bg)
    common_rewriter(p_bg)
    p_bg.set_defaults(func=cmd_bon_generate)

    p_bs = sp.add_parser("bon_score")
    p_bs.add_argument("fold", type=int, choices=[1, 2, 3])
    common_criterion(p_bs)
    common_rewriter(p_bs)
    p_bs.set_defaults(func=cmd_bon_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
