"""Cross-method summary: attack deltas, gold deltas, unknown-judge penalty,
word-count fidelity, embed sim, hallucinated-specifics, detector deltas,
agreement-score drift, clarity-regressor drift.

Run after the main pipeline finishes. Produces a wide CSV per method
and a top-level JSON with the headline comparisons we care about
(attack-vs-gold gap, method ranking on each axis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from eval_suite.schema import DEFAULT_DB_PATH, connect


def load_long(con, sample_tag: str) -> pd.DataFrame:
    """Everything in `evaluations` for the sample scope, with paragraph context joined in."""
    sql = f"""
        SELECT p.document_id, p.origin_kind, p.method_slug, p.base_document_id,
               p.word_count, p.writer_agreement_quintile, p.writer_is_top_decile,
               e.metric, e.criterion, e.source, e.panel, e.value
        FROM paragraphs p
        JOIN evaluations e ON e.paragraph_id = p.document_id
        WHERE (p.origin_kind='original_writer'
               AND p.document_id IN (SELECT document_id FROM sampled_writers WHERE tag='{sample_tag}'))
           OR (p.origin_kind='rewrite'
               AND p.base_document_id IN (SELECT document_id FROM sampled_writers WHERE tag='{sample_tag}'))
    """
    return pd.read_sql_query(sql, con)


def _mean_by_group(df: pd.DataFrame, by: list[str], value_col: str = "value") -> pd.DataFrame:
    return df.groupby(by, dropna=False)[value_col].mean().reset_index()


def method_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, metric, criterion, source, panel) with the mean
    score across rewrites. For rewrites we additionally compute the delta
    vs. the base original's metric value for the same (metric, source).
    """
    orig = long_df[long_df["origin_kind"] == "original_writer"].copy()
    orig_map = orig.set_index(["document_id", "metric", "criterion", "source"])["value"].to_dict()

    rw = long_df[long_df["origin_kind"] == "rewrite"].copy()
    rw["orig_value"] = rw.apply(
        lambda r: orig_map.get((r["base_document_id"], r["metric"], r["criterion"], r["source"])),
        axis=1,
    )
    rw["delta"] = rw["value"].astype(float) - rw["orig_value"].astype(float)

    rows = []
    for keys, g in rw.groupby(["method_slug", "metric", "criterion", "source", "panel"], dropna=False):
        m, met, crit, src, pnl = keys
        vals = g["value"].astype(float).dropna()
        deltas = g["delta"].astype(float).dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "method": m, "metric": met, "criterion": crit, "source": src, "panel": pnl,
            "n": int(len(vals)),
            "mean_value": float(vals.mean()),
            "mean_delta_vs_base": float(deltas.mean()) if len(deltas) else None,
            "p05_value": float(np.quantile(vals, 0.05)),
            "p95_value": float(np.quantile(vals, 0.95)),
        })
    return pd.DataFrame(rows)


def attack_vs_gold_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """For each method × criterion, compute (mean gold score) − (mean attack score)
    averaged across judges in each panel. The sign and size of this gap is the
    unknown-judge penalty proxy."""
    rw_judges = summary[
        (summary["metric"] == "judge_score") & (summary["panel"].isin(["attack", "gold"]))
    ].copy()
    if rw_judges.empty:
        return pd.DataFrame()
    panel_means = (rw_judges.groupby(["method", "criterion", "panel"], dropna=False)
                           ["mean_value"].mean().unstack("panel").reset_index())
    panel_means["gold_minus_attack"] = panel_means.get("gold", 0) - panel_means.get("attack", 0)
    delta_means = (rw_judges.groupby(["method", "criterion", "panel"], dropna=False)
                            ["mean_delta_vs_base"].mean().unstack("panel").reset_index())
    delta_means.rename(columns={"attack": "delta_attack", "gold": "delta_gold"}, inplace=True)
    out = panel_means.merge(delta_means, on=["method", "criterion"], how="outer")
    return out.sort_values(["criterion", "method"])


def leave_one_out_panel(long_df: pd.DataFrame, criterion: str = "clarity",
                          panel: str = "attack") -> pd.DataFrame:
    """For each judge in the panel: mean rewrite score when we pretend that
    judge was held out. This is a proxy for the within-panel "unknown-judge"
    penalty — because for each held-out judge h, the remaining 4 judges are
    what a theoretical attacker would see."""
    rw = long_df[(long_df["origin_kind"] == "rewrite")
                  & (long_df["metric"] == "judge_score")
                  & (long_df["panel"] == panel)
                  & (long_df["criterion"] == criterion)].copy()
    if rw.empty:
        return pd.DataFrame()
    rows = []
    judges = sorted(rw["source"].unique())
    for held_out in judges:
        visible = rw[rw["source"] != held_out]
        held = rw[rw["source"] == held_out]
        mean_visible_by_method = visible.groupby("method_slug")["value"].mean()
        mean_held_by_method = held.groupby("method_slug")["value"].mean()
        both = mean_visible_by_method.index.intersection(mean_held_by_method.index)
        for m in both:
            rows.append({
                "method": m,
                "held_out_judge": held_out,
                "visible_mean": float(mean_visible_by_method[m]),
                "held_mean": float(mean_held_by_method[m]),
                "held_minus_visible": float(mean_held_by_method[m] - mean_visible_by_method[m]),
            })
    return pd.DataFrame(rows)


def run(sample_tag: str, criterion: str = "clarity",
         db_path: Path = DEFAULT_DB_PATH) -> dict:
    con = connect(db_path)
    try:
        long_df = load_long(con, sample_tag)
    finally:
        con.close()
    out_dir = Path(db_path).parent / f"analysis_{sample_tag}"
    out_dir.mkdir(exist_ok=True)

    summary = method_summary(long_df)
    summary.to_csv(out_dir / "method_summary.csv", index=False)

    gap = attack_vs_gold_gap(summary)
    gap.to_csv(out_dir / "attack_vs_gold_gap.csv", index=False)

    loo_attack = leave_one_out_panel(long_df, criterion, "attack")
    loo_attack.to_csv(out_dir / "loo_attack.csv", index=False)
    loo_gold = leave_one_out_panel(long_df, criterion, "gold")
    loo_gold.to_csv(out_dir / "loo_gold.csv", index=False)

    # Top-level headlines for quick reading.
    headlines = {
        "sample_tag": sample_tag,
        "criterion": criterion,
        "n_paragraphs_scored": int(long_df["document_id"].nunique()),
        "n_methods_with_rewrites": int(long_df[long_df["origin_kind"]=="rewrite"]["method_slug"].nunique()),
        "metrics_populated": sorted(long_df["metric"].unique().tolist()),
    }
    # Per-method headline table for clarity judges + length + detector.
    rw = long_df[long_df["origin_kind"] == "rewrite"]
    orig = long_df[long_df["origin_kind"] == "original_writer"]
    def _mean_metric(df, metric, criterion=None, source=None):
        m = df[df["metric"] == metric]
        if criterion is not None: m = m[m["criterion"] == criterion]
        if source is not None: m = m[m["source"] == source]
        return float(m["value"].mean()) if len(m) else None
    by_method_rows = []
    for method in sorted(rw["method_slug"].dropna().unique()):
        sub = rw[rw["method_slug"] == method]
        row = {"method": method}
        # attack/gold mean clarity score
        for pnl in ("attack", "gold"):
            sc = sub[(sub["metric"]=="judge_score") & (sub["panel"]==pnl) & (sub["criterion"]==criterion)]
            row[f"{pnl}_mean_clarity"] = float(sc["value"].mean()) if len(sc) else None
        # word_count
        wc = sub[sub["metric"]=="word_count"]
        row["rewrite_mean_words"] = float(wc["value"].mean()) if len(wc) else None
        ratio = sub[sub["metric"]=="word_count_ratio_vs_base"]
        row["word_ratio_vs_base"] = float(ratio["value"].mean()) if len(ratio) else None
        # embed sim
        es = sub[sub["metric"]=="embed_cosine_sim_to_base"]
        row["embed_sim_mean"] = float(es["value"].mean()) if len(es) else None
        # hallucinated specifics
        hs = sub[sub["metric"]=="hallucinated_specifics_rate_per100w"]
        row["hallucinated_specifics_per100w"] = float(hs["value"].mean()) if len(hs) else None
        # agreement_score_pred
        ag = sub[sub["metric"]=="agreement_score_pred"]
        row["agreement_pred_mean"] = float(ag["value"].mean()) if len(ag) else None
        # clarity_regressor_pred
        cr = sub[sub["metric"]=="clarity_score_pred"]
        row["clarity_regressor_pred_mean"] = float(cr["value"].mean()) if len(cr) else None
        by_method_rows.append(row)
    # baseline row for originals
    row = {"method": "ORIGINAL_WRITER"}
    for pnl in ("attack", "gold"):
        sc = orig[(orig["metric"]=="judge_score") & (orig["panel"]==pnl) & (orig["criterion"]==criterion)]
        row[f"{pnl}_mean_clarity"] = float(sc["value"].mean()) if len(sc) else None
    wc = orig[orig["metric"]=="word_count"]
    row["rewrite_mean_words"] = float(wc["value"].mean()) if len(wc) else None
    row["word_ratio_vs_base"] = 1.0
    row["embed_sim_mean"] = None
    row["hallucinated_specifics_per100w"] = None
    ag = orig[orig["metric"]=="agreement_score_pred"]
    row["agreement_pred_mean"] = float(ag["value"].mean()) if len(ag) else None
    cr = orig[orig["metric"]=="clarity_score_pred"]
    row["clarity_regressor_pred_mean"] = float(cr["value"].mean()) if len(cr) else None
    by_method_rows.insert(0, row)

    by_method_df = pd.DataFrame(by_method_rows)
    by_method_df.to_csv(out_dir / "by_method_headlines.csv", index=False)
    headlines["by_method"] = by_method_rows

    (out_dir / "headlines.json").write_text(json.dumps(headlines, indent=2))
    print(json.dumps(headlines, indent=2))
    print(f"Wrote {out_dir}/")
    return headlines


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-tag", default="main_20pct")
    p.add_argument("--criterion", default="clarity")
    args = p.parse_args()
    run(args.sample_tag, args.criterion)
