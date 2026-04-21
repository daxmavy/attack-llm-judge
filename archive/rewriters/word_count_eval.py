"""Word-count fidelity evaluation across rewriter methods.

Ideal rewrites match the original word count (plan.md constraint). This
script quantifies how close each method gets and writes a comparison
CSV + prints a table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "rewriters" / "results"


def fidelity(df: pd.DataFrame) -> dict:
    d = df.copy()
    d = d[d["rewrite_ok"] & d["rewrite_text"].notna()]
    d["ratio"] = d["rewrite_words"] / d["orig_words"]
    d["abs_err"] = (d["rewrite_words"] - d["orig_words"]).abs()
    d["pct_err"] = d["abs_err"] / d["orig_words"]
    return {
        "n": int(len(d)),
        "orig_mean": float(d["orig_words"].mean()),
        "rewrite_mean": float(d["rewrite_words"].mean()),
        "ratio_mean": float(d["ratio"].mean()),
        "ratio_median": float(d["ratio"].median()),
        "ratio_std": float(d["ratio"].std()),
        "ratio_min": float(d["ratio"].min()),
        "ratio_max": float(d["ratio"].max()),
        "abs_err_mean": float(d["abs_err"].mean()),
        "pct_err_mean": float(d["pct_err"].mean()),
        "pct_err_median": float(d["pct_err"].median()),
        "pct_within_5": float((d["pct_err"] <= 0.05).mean()),
        "pct_within_10": float((d["pct_err"] <= 0.10).mean()),
        "pct_within_20": float((d["pct_err"] <= 0.20).mean()),
        "pct_under_0_80": float((d["ratio"] < 0.80).mean()),
        "pct_over_1_20": float((d["ratio"] > 1.20).mean()),
    }


def main(tags: list[str]) -> None:
    rows = []
    for tag in tags:
        path = RESULTS_DIR / f"rewrites_{tag}.csv"
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        # rewrite_words may be missing on older rewrite CSVs; recompute if needed.
        if "rewrite_words" not in df.columns:
            df["rewrite_words"] = df["rewrite_text"].fillna("").astype(str).str.split().str.len()
        if "orig_words" not in df.columns:
            df["orig_words"] = df["document_text"].str.split().str.len()
        for method, g in df.groupby("method"):
            stats = fidelity(g)
            stats.update({"tag": tag, "method": method})
            rows.append(stats)
    out = pd.DataFrame(rows)
    cols = ["tag", "method", "n", "orig_mean", "rewrite_mean", "ratio_mean", "ratio_median",
            "abs_err_mean", "pct_err_mean", "pct_within_5", "pct_within_10", "pct_within_20",
            "pct_under_0_80", "pct_over_1_20"]
    out = out[cols].sort_values(["method"])
    out.to_csv(RESULTS_DIR / "wordcount_fidelity.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 180):
        print(out.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default="v1,v2_tight")
    args = p.parse_args()
    main([t.strip() for t in args.tags.split(",") if t.strip()])
