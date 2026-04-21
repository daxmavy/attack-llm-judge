"""End-to-end smoke test: N writer paragraphs → all 10 methods → attack-panel judging → metrics → benchmarks.

Prints per-step timing, call counts, and final cost. Used to (a)
validate the whole pipeline works end-to-end, (b) calibrate the cost
projections with real token numbers before the full run.

Does NOT touch the gold panel by default — that's the expensive part
and we want operator approval first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from eval_suite.judge_runner import run_panel
from eval_suite.metrics import embedding, hallucinated_specifics, word_count
from eval_suite.rewrite_runner import METHOD_SPEC, run_method
from eval_suite.schema import DEFAULT_DB_PATH, connect


METHODS = list(METHOD_SPEC.keys())


def main(n: int = 20, methods: list[str] | None = None,
         db_path: Path = DEFAULT_DB_PATH,
         include_gold: bool = False,
         max_workers: int = 4) -> dict:
    if methods is None:
        methods = METHODS
    t0 = time.time()
    print(f"[smoke] source paragraphs: {n}, methods: {methods}")

    # 1. Generate rewrites (each method runs on the same N writer paragraphs).
    rewriter_stats = {}
    for m in methods:
        s = time.time()
        stats = run_method(m, origin_kind="original_writer", limit=n, max_workers=max_workers)
        rewriter_stats[m] = {**stats, "seconds": round(time.time() - s, 1)}
        print(f"  rewrites {m}: {stats.get('n_done')} in {rewriter_stats[m]['seconds']}s, "
              f"rewriter $={stats.get('cost_rewriter_usd'):.4f}")

    # 2. Run attack-panel judging on originals (writers only, limited to n) + all rewrites so far.
    #    Scope with a WHERE filtering to just the N writers used here + their rewrites.
    con = connect(db_path)
    try:
        docs = pd.read_sql_query(
            f"""SELECT document_id FROM paragraphs
                WHERE origin_kind='original_writer' LIMIT {int(n)}""", con)
        writer_ids = docs["document_id"].tolist()
        if writer_ids:
            placeholders = ",".join([f"'{x}'" for x in writer_ids])
            rewrite_where = (f"(origin_kind='original_writer' AND document_id IN ({placeholders})) "
                             f"OR (origin_kind='rewrite' AND base_document_id IN ({placeholders}))")
        else:
            rewrite_where = "1=0"
    finally:
        con.close()
    judge_stats = {}
    s = time.time()
    judge_stats["attack"] = run_panel("clarity", "attack", where=rewrite_where,
                                       max_workers=max_workers)
    judge_stats["attack"]["seconds"] = round(time.time() - s, 1)
    if include_gold:
        s = time.time()
        judge_stats["gold"] = run_panel("clarity", "gold", where=rewrite_where,
                                         max_workers=max_workers)
        judge_stats["gold"]["seconds"] = round(time.time() - s, 1)
    print(f"  judge stats: {judge_stats}")

    # 3. Non-LLM metrics for rewrites.
    print("  computing word_count ...")
    n_wc = word_count.score_all(db_path)
    print("  computing embed_cosine ...")
    try:
        n_emb = embedding.score_rewrites(db_path=db_path)
    except Exception as e:
        n_emb = -1; print(f"  embedding skipped: {e}")
    print("  computing hallucinated_specifics ...")
    try:
        n_hal = hallucinated_specifics.score_rewrites(db_path=db_path)
    except Exception as e:
        n_hal = -1; print(f"  hallucinated skipped: {e}")

    # 4. Summary.
    summary = {
        "n_source": n,
        "methods": methods,
        "rewriter_stats": rewriter_stats,
        "judge_stats": judge_stats,
        "metric_rows_written": {"word_count": n_wc, "embed": n_emb, "hallucinated_specifics": n_hal},
        "total_seconds": round(time.time() - t0, 1),
    }
    out_path = Path(db_path).parent / "smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--methods", default=None)
    p.add_argument("--include-gold", action="store_true")
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()
    methods = args.methods.split(",") if args.methods else None
    main(n=args.n, methods=methods, include_gold=args.include_gold,
         max_workers=args.max_workers)
