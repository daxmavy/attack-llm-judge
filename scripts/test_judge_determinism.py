"""Determinism check: re-score the same 50 paragraphs 10 times with the same
vLLM judge at temperature=0 and report whether scores are identical across reps.

If scores ARE identical, the planned variance-estimation experiment is moot —
temp=0 produces deterministic output and re-running yields zero variance.

If scores DIFFER, we have a genuine source of variance (continuous batching ->
non-associative float accumulation -> token logit perturbation -> argmax flip)
and the variance experiment is worth running with temp=0 (no need to artificially
introduce sampling noise via temp>0).

Uses phi35mini (3.8B, smallest of the 6-judge panel — fastest probe; ~5 min total).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ENV_FILE = "/home/max/attack-llm-judge/.env"
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("VLLM_CACHE_ROOT", "/workspace/vllm_cache")

sys.path.insert(0, "/workspace/grpo_run")
sys.path.insert(0, "/home/max/attack-llm-judge")

DB = "/home/max/attack-llm-judge/data/paragraphs.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi35mini",
                    help="judge key (phi35mini = fastest, 3.8B)")
    ap.add_argument("--n-paragraphs", type=int, default=50)
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--criterion", default="clarity",
                    choices=["clarity", "informativeness"])
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str,
                    default="/workspace/grpo_run/judge_determinism_test.json")
    args = ap.parse_args()

    from run_pilot_len_pen import JUDGE_REGISTRY, JudgeVLLM, RUBRICS

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT r.rewrite_id, p.proposition, r.text
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method = 'original'
        ORDER BY r.rewrite_id
    """).fetchall()
    print(f"[{time.strftime('%H:%M:%S')}] universe of human originals: {len(rows)}", flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.n_paragraphs)
    rids = [r[0] for r in sample]
    props = [r[1] for r in sample]
    texts = [r[2] for r in sample]
    print(f"[{time.strftime('%H:%M:%S')}] sampled {args.n_paragraphs} paragraphs (seed={args.seed})", flush=True)

    spec = JUDGE_REGISTRY[args.judge]
    print(f"[{time.strftime('%H:%M:%S')}] loading judge: {spec[0]} ({spec[1]})", flush=True)
    j = JudgeVLLM(*spec, rubric=args.criterion, gpu_mem_util=args.gpu_mem_util)
    print(f"[{time.strftime('%H:%M:%S')}] judge loaded; running {args.n_reps} reps × "
          f"{args.n_paragraphs} paragraphs at temperature=0", flush=True)

    all_scores = []
    for rep in range(args.n_reps):
        t0 = time.time()
        scores = j.score(props, texts)
        dt = time.time() - t0
        all_scores.append(list(scores))
        print(f"  rep {rep + 1}/{args.n_reps}: {dt:.1f}s ({len(scores)/dt:.1f}/s); "
              f"first 5 = {scores[:5]}", flush=True)

    # Cross-comparison
    print("\n=== DETERMINISM SUMMARY ===", flush=True)
    n_identical = 0
    delta_counts = Counter()
    max_abs_delta = 0
    per_paragraph = []
    for i in range(args.n_paragraphs):
        scores_i = [all_scores[r][i] for r in range(args.n_reps)]
        unique = set(scores_i)
        delta = max(scores_i) - min(scores_i)
        max_abs_delta = max(max_abs_delta, delta)
        delta_counts[delta] += 1
        if len(unique) == 1:
            n_identical += 1
        per_paragraph.append({
            "rewrite_id": rids[i],
            "scores": scores_i,
            "n_unique": len(unique),
            "delta": delta,
        })

    print(f"  paragraphs identical across all {args.n_reps} reps: "
          f"{n_identical}/{args.n_paragraphs} ({100*n_identical/args.n_paragraphs:.1f}%)", flush=True)
    print(f"  max abs delta across reps: {max_abs_delta}", flush=True)
    print(f"  delta distribution:", flush=True)
    for delta, count in sorted(delta_counts.items()):
        print(f"    delta={delta}: {count} paragraphs", flush=True)

    if n_identical == args.n_paragraphs:
        verdict = "deterministic"
        recommendation = "SKIP variance experiment — temp=0 gives identical scores"
    elif max_abs_delta <= 1:
        verdict = "near-deterministic"
        recommendation = "RUN variance experiment cautiously — small float noise visible but tiny"
    else:
        verdict = "non-deterministic"
        recommendation = "RUN variance experiment — temp=0 has real variance from batching"
    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  RECOMMENDATION: {recommendation}", flush=True)

    out = {
        "judge": args.judge,
        "judge_slug": spec[0],
        "model_id": spec[1],
        "criterion": args.criterion,
        "n_paragraphs": args.n_paragraphs,
        "n_reps": args.n_reps,
        "gpu_mem_util": args.gpu_mem_util,
        "seed": args.seed,
        "n_identical_paragraphs": n_identical,
        "max_abs_delta": max_abs_delta,
        "delta_distribution": dict(delta_counts),
        "verdict": verdict,
        "recommendation": recommendation,
        "per_paragraph": per_paragraph,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=int))
    print(f"  full record written to: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
