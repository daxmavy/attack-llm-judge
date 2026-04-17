"""Unified benchmark runner: run every metric's paul_data benchmark and write results.

Metrics with benchmarks:
- Attack-panel judges: Pearson/Spearman/MAE of each judge's clarity score vs human_mean_clarity on originals.
- Gold-panel judges: same (once populated).
- Agreement regressor: from deberta_regressor.benchmark_agreement.
- Clarity regressor: from deberta_regressor.benchmark_clarity (if trained).
- Embedding sim: from embedding.benchmark.
- Hallucinated specifics: from hallucinated_specifics.benchmark.
- AI detector: from ai_detector.benchmark.
- Word count: descriptive stats (word_count.benchmark).

Writes the whole block to eval_suite/benchmarks/benchmarks_<timestamp>.json
and prints a short summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

from eval_suite.schema import DEFAULT_DB_PATH, connect


REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "eval_suite" / "benchmarks"
BENCH_DIR.mkdir(exist_ok=True)


def judges_vs_humans(criterion: str = "clarity", panel: str | None = None,
                      db_path: Path = DEFAULT_DB_PATH) -> dict:
    con = connect(db_path)
    try:
        where_panel = f" AND e.panel='{panel}'" if panel else ""
        df = pd.read_sql_query(f"""
            SELECT p.document_id, p.origin_kind, e.source AS judge, e.panel,
                   e.value AS judge_score,
                   p.human_mean_clarity AS human_clarity
            FROM paragraphs p
            JOIN evaluations e ON e.paragraph_id=p.document_id
            WHERE e.metric='judge_score' AND e.criterion=?
              AND p.human_mean_clarity IS NOT NULL
              {where_panel}
        """, con, params=(criterion,))
    finally:
        con.close()
    if len(df) == 0:
        return {"note": "no judge scores on originals yet"}
    out = {"overall": {}, "by_judge": {}, "by_judge_and_origin_kind": {}}
    for judge, g in df.groupby("judge"):
        if len(g) < 5:
            continue
        out["by_judge"][judge] = {
            "n": int(len(g)),
            "panel": g["panel"].iloc[0],
            "pearson": float(pearsonr(g["judge_score"], g["human_clarity"])[0]),
            "spearman": float(spearmanr(g["judge_score"], g["human_clarity"]).correlation),
            "mae": float(mean_absolute_error(g["human_clarity"], g["judge_score"])),
            "judge_mean": float(g["judge_score"].mean()),
            "human_mean": float(g["human_clarity"].mean()),
        }
        for ok, gg in g.groupby("origin_kind"):
            if len(gg) < 5:
                continue
            key = f"{judge} | {ok}"
            out["by_judge_and_origin_kind"][key] = {
                "n": int(len(gg)),
                "pearson": float(pearsonr(gg["judge_score"], gg["human_clarity"])[0]),
                "spearman": float(spearmanr(gg["judge_score"], gg["human_clarity"]).correlation),
                "mae": float(mean_absolute_error(gg["human_clarity"], gg["judge_score"])),
            }
    return out


def run(db_path: Path = DEFAULT_DB_PATH) -> dict:
    from eval_suite.metrics import word_count, embedding, hallucinated_specifics
    from eval_suite.metrics import deberta_regressor, ai_detector
    results: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("[bench] word_count ...")
    try:
        results["word_count"] = word_count.benchmark(db_path)
    except Exception as e:
        results["word_count"] = {"error": str(e)}

    print("[bench] attack judges vs human_clarity ...")
    results["judges_attack_vs_human_clarity"] = judges_vs_humans("clarity", "attack", db_path)
    print("[bench] gold judges vs human_clarity ...")
    results["judges_gold_vs_human_clarity"] = judges_vs_humans("clarity", "gold", db_path)

    print("[bench] agreement_model ...")
    try:
        results["agreement_model"] = deberta_regressor.benchmark_agreement(db_path=db_path)
    except Exception as e:
        results["agreement_model"] = {"error": str(e)}

    print("[bench] clarity_regressor ...")
    try:
        results["clarity_regressor"] = deberta_regressor.benchmark_clarity(db_path=db_path)
    except Exception as e:
        results["clarity_regressor"] = {"error": str(e)}

    print("[bench] embedding_sim ...")
    try:
        results["embedding_sim"] = embedding.benchmark(db_path=db_path)
    except Exception as e:
        results["embedding_sim"] = {"error": str(e)}

    print("[bench] hallucinated_specifics ...")
    try:
        results["hallucinated_specifics"] = hallucinated_specifics.benchmark(db_path=db_path)
    except Exception as e:
        results["hallucinated_specifics"] = {"error": str(e)}

    print("[bench] ai_detector ...")
    try:
        results["ai_detector"] = ai_detector.benchmark(db_path=db_path)
    except Exception as e:
        results["ai_detector"] = {"error": str(e)}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = BENCH_DIR / f"benchmarks_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = p.parse_args()
    print(json.dumps(run(args.db), indent=2))
