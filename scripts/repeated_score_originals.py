"""Re-score originals N times with each non-OpenRouter judge to measure
the score variance introduced by vLLM continuous batching at temperature=0.

Determinism test (test_judge_determinism.py) showed phi35mini gives non-trivial
variance (15/50 paragraphs differ across 10 reps, max delta = 16). This script
captures variance for the full set of originals × all 6 judges × both criteria.

Coverage:
  --subset eval-humans    -> 714 eval-split human originals only (method='original' in eval doc IDs)
  --subset all-originals  -> 1805 human + 1805 AI originals (method IN ('original','original_ai'))

Schema (new table — keeps attack_judge_scores untouched):
    CREATE TABLE attack_judge_repeated_scores (
        rewrite_id TEXT NOT NULL,
        judge_slug TEXT NOT NULL,
        criterion  TEXT NOT NULL,
        rep_id     INTEGER NOT NULL,
        score      REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (rewrite_id, judge_slug, criterion, rep_id)
    );

Idempotent: skips (rewrite_id, judge_slug, criterion, rep_id) tuples already
present. Load-once-per-judge: each judge loads once, then runs 2 criteria × N
reps before unloading.

Settings: temperature=0 (matching main experiment exactly). Batching variance
is intrinsic to vLLM continuous batching — no need to introduce sampling noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
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
DATASET = "/workspace/grpo_run/controversial_40_3fold.json"


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attack_judge_repeated_scores (
            rewrite_id TEXT NOT NULL,
            judge_slug TEXT NOT NULL,
            criterion  TEXT NOT NULL,
            rep_id     INTEGER NOT NULL,
            score      REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rewrite_id, judge_slug, criterion, rep_id)
        )
    """)
    conn.commit()


def load_universe(conn, subset: str):
    if subset == "eval-humans":
        d = json.loads(Path(DATASET).read_text())
        eval_doc_ids = [r["document_id"] for r in d["rows"] if r["split"] == "eval"]
        rows = conn.execute(f"""
            SELECT r.rewrite_id, p.proposition, r.text
            FROM attack_rewrites r
            JOIN paragraphs p ON r.source_doc_id = p.document_id
            WHERE r.method = 'original' AND r.source_doc_id IN ({','.join('?'*len(eval_doc_ids))})
            ORDER BY r.rewrite_id
        """, eval_doc_ids).fetchall()
    elif subset == "all-originals":
        rows = conn.execute("""
            SELECT r.rewrite_id, p.proposition, r.text
            FROM attack_rewrites r
            JOIN paragraphs p ON r.source_doc_id = p.document_id
            WHERE r.method IN ('original', 'original_ai')
            ORDER BY r.rewrite_id
        """).fetchall()
    else:
        raise ValueError(f"unknown subset: {subset}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["eval-humans", "all-originals"], default="all-originals")
    ap.add_argument("--judges", nargs="+",
                    default=["qwen95b", "llama8b", "gemma9b", "mistral7b", "phi35mini", "cmdr7b"])
    ap.add_argument("--criteria", nargs="+", default=["clarity", "informativeness"])
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    args = ap.parse_args()

    import torch
    from run_pilot_len_pen import JUDGE_REGISTRY, JudgeVLLM, RUBRICS

    conn = sqlite3.connect(DB, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_table(conn)

    universe = load_universe(conn, args.subset)
    print(f"[{time.strftime('%H:%M:%S')}] subset={args.subset}  rows={len(universe)}", flush=True)
    print(f"  judges: {args.judges}", flush=True)
    print(f"  criteria: {args.criteria}", flush=True)
    print(f"  n_reps: {args.n_reps}", flush=True)
    print(f"  total expected scores: {len(universe) * len(args.judges) * len(args.criteria) * args.n_reps}",
          flush=True)

    rids = [r[0] for r in universe]
    props = [r[1] for r in universe]
    texts = [r[2] for r in universe]

    for judge_key in args.judges:
        spec = JUDGE_REGISTRY[judge_key]
        slug = spec[0]

        # Find what's already done for this judge (resume support)
        already = {(r[0], r[1], r[2]) for r in conn.execute(
            "SELECT rewrite_id, criterion, rep_id FROM attack_judge_repeated_scores WHERE judge_slug=?",
            (slug,)).fetchall()}
        # plan: which (criterion, rep_id) combos still need scoring on the universe?
        per_combo_remaining = {}
        for cri in args.criteria:
            for rep_id in range(args.n_reps):
                missing = [rid for rid in rids if (rid, cri, rep_id) not in already]
                per_combo_remaining[(cri, rep_id)] = missing
        total_remaining = sum(len(v) for v in per_combo_remaining.values())
        print(f"\n[{time.strftime('%H:%M:%S')}] judge={slug}: {total_remaining} score calls remaining "
              f"(of {len(rids)*len(args.criteria)*args.n_reps} total)", flush=True)
        if total_remaining == 0:
            print(f"  skipping {slug}: all reps already done", flush=True)
            continue

        first_cri = args.criteria[0]
        print(f"[{time.strftime('%H:%M:%S')}] loading {slug} (initial rubric={first_cri}, "
              f"gpu_mem_util={args.gpu_mem_util})", flush=True)
        j = JudgeVLLM(*spec, rubric=first_cri, gpu_mem_util=args.gpu_mem_util)

        for cri in args.criteria:
            j.rubric_name = cri
            j.rubric_text = RUBRICS[cri]
            for rep_id in range(args.n_reps):
                missing_rids = per_combo_remaining[(cri, rep_id)]
                if not missing_rids:
                    print(f"  [{slug} × {cri} × rep{rep_id}] already complete; skipping", flush=True)
                    continue
                # Build the prop+text lists in same order as missing_rids
                idx = {rid: i for i, rid in enumerate(rids)}
                p_subset = [props[idx[rid]] for rid in missing_rids]
                t_subset = [texts[idx[rid]] for rid in missing_rids]
                t0 = time.time()
                scores = j.score(p_subset, t_subset)
                dt = time.time() - t0
                rate = len(scores) / max(1e-6, dt)
                inserts = [(rid, slug, cri, rep_id, float(sc))
                           for rid, sc in zip(missing_rids, scores)]
                conn.executemany("""
                    INSERT OR IGNORE INTO attack_judge_repeated_scores
                    (rewrite_id, judge_slug, criterion, rep_id, score)
                    VALUES (?, ?, ?, ?, ?)
                """, inserts)
                conn.commit()
                print(f"  [{slug} × {cri} × rep{rep_id}] {len(scores)} scores in {dt:.1f}s "
                      f"({rate:.1f}/s)", flush=True)

        del j
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{time.strftime('%H:%M:%S')}] {slug} unloaded", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE — repeated scoring complete.", flush=True)


if __name__ == "__main__":
    main()
