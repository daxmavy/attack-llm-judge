"""Regenerate Figure 1 (judge-panel uplift) and Figure 2 (quality vs uplift)
for the main body. Reads data/paragraphs.db. Default annotations only.

Figure 1: two horizontal sub-plots sharing a y-range that includes zero.
  Left:  feedback judges (training panel) vs. held-out in-sample (small OOS).
  Right: out-of-sample judges, matched criterion vs.\ cross criterion.

Figure 2: two side-by-side sub-plots sharing the x-axis (5-point buckets of
the original doc's panel mean on the matched criterion).
  Left:  mean (rewrite - original) raw judge score per (rewrite, judge) cell.
  Right: share of (rewrite, judge) cells where rewrite > original.
"""
import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data/paragraphs.db")
OUT = os.path.join(ROOT, "paper/figures")
os.makedirs(OUT, exist_ok=True)

NON_OPUS_METHODS = [
    "naive", "lit_informed_tight",
    "bon_panel", "bon_panel_nli", "bon_panel_single_nli",
    "grpo_400step", "grpo_nli_400step", "grpo_nli_single",
]
OPUS_METHOD = "lit_informed_tight_strictlen_opus47"
ALL_METHODS = NON_OPUS_METHODS + [OPUS_METHOD]

REWRITERS = [
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-3-1b-it",
    "anthropic/claude-opus-4-7",
]

TRAINING_JUDGES = ["judge_qwen95b", "judge_llama8b", "judge_gemma9b"]
SMALL_OOS = ["judge_mistral7b", "judge_phi35mini", "judge_cmdr7b"]
LARGE_OOS = ["judge_qwen3_5_flash_02_23", "judge_gemini_2_5_flash_lite", "judge_mimo_v2_flash"]
ALL_OOS = SMALL_OOS + LARGE_OOS
ALL_JUDGES = TRAINING_JUDGES + ALL_OOS

METHOD_LABEL = {
    "naive": "Naïve",
    "lit_informed_tight": "Lit-informed",
    "bon_panel": "BoN",
    "bon_panel_nli": "BoN-NLI",
    "bon_panel_single_nli": "BoN-NLI (single)",
    "grpo_400step": "GRPO",
    "grpo_nli_400step": "GRPO-NLI",
    "grpo_nli_single": "GRPO-NLI (single)",
    OPUS_METHOD: "Lit-informed (Opus)",
}
OKABE = {
    "orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00",
    "pink": "#CC79A7", "black": "#000000",
}
METHOD_COLOR = {
    "naive": OKABE["blue"],
    "lit_informed_tight": OKABE["vermilion"],
    "bon_panel": OKABE["green"],
    "bon_panel_nli": OKABE["orange"],
    "bon_panel_single_nli": OKABE["pink"],
    "grpo_400step": OKABE["sky_blue"],
    "grpo_nli_400step": OKABE["yellow"],
    "grpo_nli_single": OKABE["black"],
    OPUS_METHOD: "#7B0099",
}


def cluster_boot_mean(df, value_col, rid_col="rewrite_id", n_boot=400, seed=42):
    """95% cluster-robust CI on the mean of `value_col`, clustered by rewrite_id.

    Uses the per-cluster (sum, count) trick to do each bootstrap in O(n_rids)
    instead of materialising the full sample.
    """
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    g = df.groupby(rid_col)[value_col].agg(["sum", "count"])
    sums = g["sum"].to_numpy()
    counts = g["count"].to_numpy()
    n = len(sums)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = sums[idx].sum() / counts[idx].sum()
    point = sums.sum() / counts.sum()
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_data():
    con = sqlite3.connect(DB)

    rewrites = pd.read_sql_query(
        "SELECT rewrite_id, source_doc_id, method, fold, rewriter_model, criterion AS rew_crit "
        "FROM attack_rewrites "
        "WHERE method IN (" + ",".join(f"'{m}'" for m in ALL_METHODS) + ") "
        "  AND rewriter_model IN (" + ",".join(f"'{r}'" for r in REWRITERS) + ")",
        con,
    )
    rid_set = set(rewrites["rewrite_id"])

    js = pd.read_sql_query(
        "SELECT rewrite_id, judge_slug, criterion AS judge_crit, score "
        "FROM attack_judge_scores "
        "WHERE judge_slug IN (" + ",".join(f"'{j}'" for j in ALL_JUDGES) + ")",
        con,
    )

    rs = js[js["rewrite_id"].isin(rid_set)].merge(rewrites, on="rewrite_id", how="inner")

    # Original scores: rewrite_id 'orig_<src>' for human originals
    orig_rid = pd.read_sql_query(
        "SELECT rewrite_id, source_doc_id FROM attack_rewrites WHERE method='original'",
        con,
    )
    orig = js[js["rewrite_id"].isin(set(orig_rid["rewrite_id"]))]
    orig = orig.merge(orig_rid, on="rewrite_id", how="inner")
    orig = orig[["source_doc_id", "judge_slug", "judge_crit", "score"]].rename(columns={"score": "orig_score"})

    rs = rs.merge(orig, on=["source_doc_id", "judge_slug", "judge_crit"], how="inner")
    rs["delta"] = rs["score"] - rs["orig_score"]  # raw 0-100 scale
    rs["matched"] = rs["judge_crit"] == rs["rew_crit"]
    rs["improved"] = (rs["delta"] > 0).astype(int)

    # Original panel mean on matched criterion (for Fig 2 bucketing): per (source_doc, criterion)
    orig_panel = (orig[orig["judge_slug"].isin(ALL_OOS)]
                  .groupby(["source_doc_id", "judge_crit"])["orig_score"].mean()
                  .rename("orig_panel_mean").reset_index())
    rs = rs.merge(
        orig_panel.rename(columns={"judge_crit": "rew_crit"}),
        on=["source_doc_id", "rew_crit"], how="left",
    )

    con.close()
    return rs


def fig1(rs):
    """Two panels, lines per method, y-axis includes 0."""
    # Define the four (panel_left/right, x_position) cells
    LEFT_CATS = [("Used for training", lambda d: d["judge_slug"].isin(TRAINING_JUDGES) & d["matched"]),
                 ("Not used for training", lambda d: d["judge_slug"].isin(SMALL_OOS) & d["matched"])]
    RIGHT_CATS = [("Same criterion for\ntraining and evaluation", lambda d: d["judge_slug"].isin(ALL_OOS) & d["matched"]),
                  ("Different criterion for\ntraining and evaluation", lambda d: d["judge_slug"].isin(ALL_OOS) & ~d["matched"])]

    rows = []
    for panel_name, cats in [("left", LEFT_CATS), ("right", RIGHT_CATS)]:
        for cat_name, mask_fn in cats:
            mask = mask_fn(rs)
            for m in ALL_METHODS:
                sub = rs[mask & (rs["method"] == m)]
                pt, lo, hi = cluster_boot_mean(sub, "delta")
                rows.append(dict(panel=panel_name, cat=cat_name, method=m,
                                 mean=pt, lo=lo, hi=hi, n=len(sub)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fig_judge_panel_uplift_combined_data.csv"), index=False)

    # plot
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey=True)
    ymin, ymax = df[["lo", "hi"]].to_numpy().min(), df[["lo", "hi"]].to_numpy().max()
    pad = 0.06 * (ymax - ymin)
    ymin = min(0.0, ymin - pad)
    ymax = ymax + pad

    for ax, (panel_name, cats) in zip(axes, [("left", LEFT_CATS), ("right", RIGHT_CATS)]):
        cat_names = [c[0] for c in cats]
        xs = np.arange(len(cat_names))
        for m in ALL_METHODS:
            sub = df[(df["panel"] == panel_name) & (df["method"] == m)].set_index("cat").loc[cat_names]
            color = METHOD_COLOR[m]
            marker = "*" if m == OPUS_METHOD else "o"
            size = 110 if m == OPUS_METHOD else 55
            ax.errorbar(xs, sub["mean"], yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
                        fmt="none", ecolor=color, alpha=0.5, lw=0.9, capsize=2.5, zorder=2)
            ax.plot(xs, sub["mean"], color=color, alpha=0.5, lw=1.2, zorder=2)
            ax.scatter(xs, sub["mean"], s=size, marker=marker, color=color,
                       edgecolor="white", linewidth=0.8, zorder=3, label=METHOD_LABEL[m])
        ax.set_xticks(xs)
        ax.set_xticklabels(cat_names, fontsize=10)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="#888", lw=0.5, zorder=1)

    axes[0].set_ylabel("Average uplift in judge score\n(feedback judge panel)")
    axes[1].set_ylabel("Average uplift in judge score\n(evaluation judge panel)")
    axes[0].set_title("Differing LLM judge")
    axes[1].set_title("Differing judge criterion")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.suptitle("Rewriting effectiveness degrades if judge configuration "
                 "(LLM or prompt) differs between training and evaluation",
                 fontsize=11, y=0.99)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_judge_panel_uplift_combined.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def fig2(rs):
    """Quality vs uplift, 5-point buckets on the original panel mean."""
    # Restrict to OOS-matched cells
    sub = rs[rs["judge_slug"].isin(ALL_OOS) & rs["matched"]].copy()
    sub = sub.dropna(subset=["orig_panel_mean"])
    bin_edges = np.arange(15, 105, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    sub["bucket"] = pd.cut(sub["orig_panel_mean"], bin_edges, right=False, labels=bin_centers)
    sub["bucket"] = sub["bucket"].astype(float)

    rows = []
    for m in ALL_METHODS:
        for b in bin_centers:
            cell = sub[(sub["method"] == m) & (sub["bucket"] == b)]
            if len(cell) < 30:
                continue
            d_mean, d_lo, d_hi = cluster_boot_mean(cell, "delta")
            i_mean, i_lo, i_hi = cluster_boot_mean(cell, "improved")
            rows.append(dict(method=m, bucket=b,
                             delta_mean=d_mean, delta_lo=d_lo, delta_hi=d_hi,
                             improved_mean=i_mean, improved_lo=i_lo, improved_hi=i_hi,
                             n=len(cell)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fig_quality_main_oos_data.csv"), index=False)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for m in ALL_METHODS:
        sub_m = df[df["method"] == m].sort_values("bucket")
        if sub_m.empty:
            continue
        c = METHOD_COLOR[m]
        marker = "*" if m == OPUS_METHOD else "o"
        size = 90 if m == OPUS_METHOD else 35
        # left: delta
        axes[0].errorbar(sub_m["bucket"], sub_m["delta_mean"],
                         yerr=[sub_m["delta_mean"] - sub_m["delta_lo"],
                               sub_m["delta_hi"] - sub_m["delta_mean"]],
                         fmt="none", ecolor=c, alpha=0.4, lw=0.7, capsize=1.5)
        axes[0].plot(sub_m["bucket"], sub_m["delta_mean"], color=c, alpha=0.5, lw=1.2)
        axes[0].scatter(sub_m["bucket"], sub_m["delta_mean"], s=size,
                        marker=marker, color=c, edgecolor="white", linewidth=0.6,
                        label=METHOD_LABEL[m], zorder=3)
        # right: improved share (× 100 for percent)
        axes[1].errorbar(sub_m["bucket"], sub_m["improved_mean"] * 100,
                         yerr=[(sub_m["improved_mean"] - sub_m["improved_lo"]) * 100,
                               (sub_m["improved_hi"] - sub_m["improved_mean"]) * 100],
                         fmt="none", ecolor=c, alpha=0.4, lw=0.7, capsize=1.5)
        axes[1].plot(sub_m["bucket"], sub_m["improved_mean"] * 100, color=c, alpha=0.5, lw=1.2)
        axes[1].scatter(sub_m["bucket"], sub_m["improved_mean"] * 100, s=size,
                        marker=marker, color=c, edgecolor="white", linewidth=0.6,
                        label=METHOD_LABEL[m], zorder=3)

    axes[0].set_xlabel("Avg. original document score (evaluation judge panel)")
    axes[0].set_ylabel("Avg. improvement in LLM judge score (rewritten - original)")
    axes[0].set_title("Average uplift")
    axes[0].axhline(0, color="#888", lw=0.5, zorder=1)
    axes[1].set_xlabel("Avg. original document score (evaluation judge panel)")
    axes[1].set_ylabel("% of rewrites with higher score than original")
    axes[1].set_title("Improvement rate")
    axes[1].axhline(50, color="#888", lw=0.5, zorder=1)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    fig.suptitle("Rewriting effectiveness decreases for documents which judges already rate highly",
                 fontsize=11, y=0.99)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_quality_main_oos.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def main():
    print("loading data...")
    rs = load_data()
    print(f"  {len(rs):,} (rewrite, judge, criterion) cells loaded")
    fig1(rs)
    fig2(rs)


if __name__ == "__main__":
    main()
