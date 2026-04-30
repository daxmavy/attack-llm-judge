"""Regenerate the main-body Pareto frontier figure (Figure 3) and compute X/Y
counts for the Experiment Overview text.

Reads data/paragraphs.db. Aggregates across all 6 out-of-sample judges
(matched criterion) and across all rewriter base models. One mark per method.
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

REWRITERS_NON_OPUS = [
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-3-1b-it",
]
REWRITER_OPUS = "anthropic/claude-opus-4-7"
ALL_REWRITERS = REWRITERS_NON_OPUS + [REWRITER_OPUS]

OOS_JUDGES = [
    "judge_mistral7b", "judge_phi35mini", "judge_cmdr7b",
    "judge_qwen3_5_flash_02_23",
    "judge_gemini_2_5_flash_lite",
    "judge_mimo_v2_flash",
]
TRAINING_JUDGES = ["judge_qwen95b", "judge_llama8b", "judge_gemma9b"]
PAPER_JUDGES = TRAINING_JUDGES + OOS_JUDGES

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


def main():
    con = sqlite3.connect(DB)

    # --- in-paper rewrites ---
    rewrites = pd.read_sql_query(
        "SELECT rewrite_id, source_doc_id, method, fold, rewriter_model, criterion "
        "FROM attack_rewrites "
        "WHERE method IN (" + ",".join(f"'{m}'" for m in ALL_METHODS) + ") "
        "  AND rewriter_model IN (" + ",".join(f"'{r}'" for r in ALL_REWRITERS) + ")",
        con,
    )
    X = len(rewrites)
    print(f"X (rewrites in paper) = {X:,}")
    print(rewrites.groupby("method").size().sort_values(ascending=False).to_string())

    rid_set = set(rewrites["rewrite_id"])

    # --- judge scores: rewrites and originals share the attack_judge_scores table ---
    js = pd.read_sql_query(
        "SELECT rewrite_id, judge_slug, criterion AS judge_crit, score "
        "FROM attack_judge_scores "
        "WHERE judge_slug IN (" + ",".join(f"'{j}'" for j in PAPER_JUDGES) + ")",
        con,
    )
    rs_scores = js[js["rewrite_id"].isin(rid_set)]
    Y = len(rs_scores)
    print(f"Y (rewrite scorings on paper rewrites, training+OOS judges) = {Y:,}")

    # --- reference distribution per (judge, criterion): both human + AI originals ---
    # Build (rewrite_id -> source_doc_id) map for method in {'original','original_ai'}
    ref_origs = pd.read_sql_query(
        "SELECT rewrite_id, source_doc_id FROM attack_rewrites "
        "WHERE method IN ('original','original_ai')",
        con,
    )
    ref_id_set = set(ref_origs["rewrite_id"])
    ref_scores = js[js["rewrite_id"].isin(ref_id_set)].copy()
    print(f"ref distribution rows: {len(ref_scores):,}")
    ref_arrs = {
        (j, c): np.sort(g["score"].to_numpy())
        for (j, c), g in ref_scores.groupby(["judge_slug", "judge_crit"])
    }

    def pct(scores, j, c):
        a = ref_arrs[(j, c)]
        lo = np.searchsorted(a, scores, side="left")
        hi = np.searchsorted(a, scores, side="right")
        return (lo + hi) / 2.0 / len(a)

    # --- original percentiles per (human source_doc, judge, criterion) ---
    human_origs = ref_origs[ref_origs["rewrite_id"].str.startswith("orig_")
                            & ~ref_origs["rewrite_id"].str.startswith("orig_ai_")
                            ].copy() if False else ref_origs.copy()  # all orig_* are human; orig_ai have own prefix? check below
    # The 'original' method rewrite_ids look like "orig_<src>"; for 'original_ai' check actual prefix.
    # For uplift baseline we want HUMAN originals only, since rewrites are derived from them.
    human_orig_rid = pd.read_sql_query(
        "SELECT rewrite_id, source_doc_id FROM attack_rewrites WHERE method='original'",
        con,
    )
    orig_scores = js[js["rewrite_id"].isin(set(human_orig_rid["rewrite_id"]))].copy()
    orig_scores = orig_scores.merge(human_orig_rid, on="rewrite_id", how="inner")
    orig_scores["orig_pct"] = np.nan
    for (j, c) in ref_arrs:
        m = (orig_scores["judge_slug"] == j) & (orig_scores["judge_crit"] == c)
        if m.any():
            orig_scores.loc[m, "orig_pct"] = pct(orig_scores.loc[m, "score"].to_numpy(), j, c)
    orig_scores = orig_scores[["source_doc_id", "judge_slug", "judge_crit", "score", "orig_pct"]]
    orig_scores = orig_scores.rename(columns={"score": "orig_score"})
    print(f"original judge-score rows (human originals, paper judges): {len(orig_scores):,}")

    # --- agreement scores ---
    agree_rw = pd.read_sql_query(
        "SELECT rewrite_id, score AS rew_agree FROM attack_agreement_scores_v2",
        con,
    )
    agree_orig = pd.read_sql_query(
        "SELECT paragraph_id AS source_doc_id, AVG(predicted_score) AS orig_agree "
        "FROM attack_agreement_score_predictions_v2 "
        "WHERE origin_kind='original_writer' GROUP BY paragraph_id",
        con,
    )
    con.close()

    # --- uplift table on OOS judges, matched criterion (percentile-rank uplift) ---
    rs = rs_scores[rs_scores["judge_slug"].isin(OOS_JUDGES)].merge(
        rewrites[["rewrite_id", "source_doc_id", "method", "rewriter_model", "criterion"]],
        on="rewrite_id", how="inner",
    ).rename(columns={"criterion": "rew_crit"})
    rs = rs[rs["judge_crit"] == rs["rew_crit"]].copy()
    rs["pct"] = np.nan
    for (j, c) in ref_arrs:
        m = (rs["judge_slug"] == j) & (rs["judge_crit"] == c)
        if m.any():
            rs.loc[m, "pct"] = pct(rs.loc[m, "score"].to_numpy(), j, c)
    rs = rs.merge(orig_scores, on=["source_doc_id", "judge_slug", "judge_crit"], how="inner")
    rs["delta_pct"] = rs["pct"] - rs["orig_pct"]
    print(f"OOS-matched (rewrite, judge) cells: {len(rs):,}")

    # --- preservation per rewrite (paper definition: stays on same side of 0.5) ---
    pres = rewrites.merge(agree_rw, on="rewrite_id", how="inner")
    pres = pres.merge(agree_orig, on="source_doc_id", how="inner")
    pres["polar"] = (pres["orig_agree"] <= 0.25) | (pres["orig_agree"] >= 0.75)
    pres["flipped"] = (
        ((pres["orig_agree"] >= 0.75) & (pres["rew_agree"] < 0.50))
        | ((pres["orig_agree"] <= 0.25) & (pres["rew_agree"] >= 0.50))
    )
    pres = pres[pres["polar"]].copy()
    print(f"polar rewrites: {len(pres):,}")

    # --- aggregate to one point per method ---
    def wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        d = 1 + z * z / n
        c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return ((p + z * z / (2 * n)) / d - c, (p + z * z / (2 * n)) / d + c)

    rows = []
    for m in ALL_METHODS:
        sub_up = rs[rs["method"] == m]["delta_pct"].dropna()
        sub_pr = pres[pres["method"] == m]
        n_pr = len(sub_pr)
        k_pr = int(sub_pr["flipped"].sum())
        if len(sub_up) == 0 or n_pr == 0:
            continue
        u_mean = sub_up.mean()
        u_se = sub_up.std(ddof=1) / np.sqrt(len(sub_up))
        flo, fhi = wilson(k_pr, n_pr)
        rows.append(dict(
            method=m,
            uplift=u_mean,
            u_lo=u_mean - 1.96 * u_se,
            u_hi=u_mean + 1.96 * u_se,
            preservation=1 - k_pr / n_pr,
            p_lo=1 - fhi,
            p_hi=1 - flo,
            n_up=len(sub_up),
            n_pr=n_pr,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fig_tradeoff_oos_data.csv"), index=False)
    print(df.round(3).to_string(index=False))

    # --- Pareto envelope on non-Opus ---
    def pareto_max(d, x="preservation", y="uplift"):
        s = d.sort_values(x, ascending=False).reset_index(drop=True)
        keep = []
        ymax = -np.inf
        for i, r in s.iterrows():
            if r[y] > ymax:
                ymax = r[y]
                keep.append(i)
        return s.loc[keep].sort_values(x)

    non_opus = df[df["method"].isin(NON_OPUS_METHODS)]
    front = pareto_max(non_opus)

    # --- plot ---
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    if len(front) > 1:
        ax.plot(front["preservation"], front["uplift"],
                color="#888", lw=1.3, ls="--", zorder=1,
                label="Pareto envelope (non-Opus)")

    handles_for_legend = []
    for _, r in df.iterrows():
        m = r["method"]
        c = METHOD_COLOR[m]
        marker = "*" if m == OPUS_METHOD else "o"
        size = 200 if m == OPUS_METHOD else 80
        ax.errorbar(
            r["preservation"], r["uplift"],
            xerr=[[r["preservation"] - r["p_lo"]], [r["p_hi"] - r["preservation"]]],
            yerr=[[r["uplift"] - r["u_lo"]], [r["u_hi"] - r["uplift"]]],
            fmt="none", ecolor=c, alpha=0.5, lw=0.9, capsize=2.5, zorder=2,
        )
        sc = ax.scatter(
            r["preservation"], r["uplift"],
            s=size, marker=marker, color=c,
            edgecolor="white", linewidth=0.9, zorder=3,
            label=METHOD_LABEL[m],
        )
        handles_for_legend.append(sc)
        ax.annotate(
            METHOD_LABEL[m],
            xy=(r["preservation"], r["uplift"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="#333",
        )

    ax.set_xlabel("Stance preservation rate")
    ax.set_ylabel("Uplift in score (normalised)")
    ax.set_title("All OOS judges (matched criterion)")
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_tradeoff_oos.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    print(f"\nSUMMARY: X = {X:,}, Y = {Y:,}")


if __name__ == "__main__":
    main()
