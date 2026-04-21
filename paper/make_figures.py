"""Generate figures for the paper from data/paragraphs.db."""
import sqlite3, numpy as np, pandas as pd, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data/paragraphs.db")
OUT = os.path.join(ROOT, "paper/figures")
os.makedirs(OUT, exist_ok=True)

ATTACK = ["judge_qwen7b", "judge_llama8b"]
GOLD = ["claude-haiku-4.5", "deepseek-v3.2", "gemini-2.5-flash", "gpt-5-mini", "llama-4-maverick"]

# Short display labels used in figure text (plain, no LaTeX).
SHORT = {
    "naive": "naive",
    "naive_tight": "naive_t",
    "rubric_aware": "rubric_aware",
    "injection_leadin": "injection",
    "lit_informed": "lit_inf",
    "lit_informed_tight": "lit_inf_t",
    "bon_panel": "bon",
    "icir_single": "icir",
    "rules_explicit": "rules",
    "scaffolded_cot_distill": "scaffold",
}
GOLD_LABEL = {
    "claude-haiku-4.5": "Haiku-4.5",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-2.5-flash": "Gemini-2.5-F",
    "gpt-5-mini": "GPT-5-mini",
    "llama-4-maverick": "L4-Mav",
}

con = sqlite3.connect(DB)
df = pd.read_sql_query(
    """
    SELECT p.document_id, p.origin_kind, p.method_slug, p.base_document_id, p.proposition_id,
           e.source, e.panel, e.value
    FROM paragraphs p JOIN evaluations e ON e.paragraph_id=p.document_id
    WHERE e.metric='judge_score' AND e.criterion='clarity'
      AND (
        (p.origin_kind='original_writer'
         AND p.document_id IN (SELECT document_id FROM sampled_writers WHERE tag='main_20pct'))
        OR (p.origin_kind='rewrite'
         AND p.base_document_id IN (SELECT document_id FROM sampled_writers WHERE tag='main_20pct'))
      )
    """,
    con,
)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["value"])

orig = df[df["origin_kind"] == "original_writer"]
rw = df[df["origin_kind"] == "rewrite"]


def cluster_bootstrap(series, clusters, n=1000, seed=17):
    rng = np.random.default_rng(seed)
    df_ = pd.DataFrame({"v": series.values, "c": clusters.values}).dropna()
    uc = np.array(sorted(df_["c"].unique()))
    groups = {c: df_[df_["c"] == c]["v"].values for c in uc}
    boots = []
    for _ in range(n):
        samp = rng.choice(uc, size=len(uc), replace=True)
        vals = np.concatenate([groups[c] for c in samp])
        boots.append(float(vals.mean()) if len(vals) else np.nan)
    arr = np.array(boots)
    return float(series.mean()), float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))


def doc_panel_mean(subdf, judges):
    return (
        subdf[subdf["source"].isin(judges)]
        .groupby("document_id")
        .agg(score=("value", "mean"), n=("value", "count"),
             method=("method_slug", "first"),
             base=("base_document_id", "first"),
             prop=("proposition_id", "first"))
        .reset_index()
    )


orig_att = doc_panel_mean(orig, ATTACK)
orig_gold = doc_panel_mean(orig, GOLD)
o_att_map = dict(zip(orig_att["document_id"], orig_att["score"]))
o_gold_map = dict(zip(orig_gold["document_id"], orig_gold["score"]))
rw_att = doc_panel_mean(rw, ATTACK)
rw_gold = doc_panel_mean(rw, GOLD)
rw_gold_ok = rw_gold[rw_gold["n"] >= 3].copy()

# ------------------------------------------------------------------
# Per-document gold trimmed mean (drop top + bottom when 5 valid, or
# top + bottom when 4; return plain mean when 3).
# ------------------------------------------------------------------
def _trim(vs):
    vs = sorted(vs)
    n = len(vs)
    if n < 3: return float("nan")
    if n == 3: return float(np.mean(vs))
    if n == 4: return float(np.mean(vs[1:3]))
    return float(np.mean(vs[1:4]))

g_lists = df[df["source"].isin(GOLD)].groupby("document_id")["value"].apply(list)
trim_map = {doc: _trim(vs) for doc, vs in g_lists.items() if len(vs) >= 3}
rw_gold_ok = rw_gold_ok.assign(trim=rw_gold_ok["document_id"].map(trim_map))
orig_gold["trim"] = orig_gold["document_id"].map(trim_map)
o_trim_map = dict(zip(orig_gold["document_id"], orig_gold["trim"]))

methods = sorted(rw["method_slug"].unique())

# ------------------------------------------------------------------
# Per-method uplift on both panels (paired, proposition-cluster bootstrap)
# Includes both gold-mean and gold-trim.
# ------------------------------------------------------------------
rows = []
for m in methods:
    sa = rw_att[rw_att["method"] == m].copy()
    sg = rw_gold_ok[rw_gold_ok["method"] == m].copy()
    sa["d"] = sa["score"] - sa["base"].map(o_att_map)
    sg["d"]  = sg["score"] - sg["base"].map(o_gold_map)
    sg["dt"] = sg["trim"]  - sg["base"].map(o_trim_map)
    sa = sa.dropna(subset=["d"])
    sg = sg.dropna(subset=["d", "dt"])
    da, alo, ahi = cluster_bootstrap(sa["d"], sa["prop"]) if len(sa) else (np.nan,) * 3
    if len(sg) >= 50:
        dg, glo, ghi = cluster_bootstrap(sg["d"], sg["prop"])
        dt, tlo, thi = cluster_bootstrap(sg["dt"], sg["prop"])
    else:
        dg, glo, ghi = (np.nan,) * 3
        dt, tlo, thi = (np.nan,) * 3
    rows.append(dict(method=m, da=da, alo=alo, ahi=ahi,
                      dg=dg, glo=glo, ghi=ghi,
                      dt=dt, tlo=tlo, thi=thi,
                      n_a=len(sa), n_g=len(sg)))
mres = pd.DataFrame(rows)
mres.to_csv(os.path.join(OUT, "method_uplift_ci.csv"), index=False)

# ------------------------------------------------------------------
# Figure 1: four-quadrant transfer gap.
#   x = attack-panel uplift, y = gold-panel uplift, diagonal = no gap.
# ------------------------------------------------------------------
complete = mres.dropna(subset=["dg"]).copy()
incomplete = mres[mres["dg"].isna()].copy()

fig, ax = plt.subplots(figsize=(4.8, 3.8))
xlo, xhi = 30.5, 38.5
ylo, yhi = 21.0, 31.0
ax.plot([xlo, xhi], [xlo, xhi], "--", color="0.6", linewidth=0.8,
        zorder=0, label="equal uplift (no gap)")
# Solid = gold-mean; open = gold-trim
ax.errorbar(complete["da"], complete["dg"],
            xerr=[complete["da"] - complete["alo"], complete["ahi"] - complete["da"]],
            yerr=[complete["dg"] - complete["glo"], complete["ghi"] - complete["dg"]],
            fmt="o", color="#d62728", ecolor="#d62728", elinewidth=1.0, capsize=2,
            markersize=6, label="gold mean (5 judges)")
ax.errorbar(complete["da"], complete["dt"],
            xerr=[complete["da"] - complete["alo"], complete["ahi"] - complete["da"]],
            yerr=[complete["dt"] - complete["tlo"], complete["thi"] - complete["dt"]],
            fmt="s", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.0, capsize=2,
            markersize=6, mfc="white", label="gold trimmed (drop top+bottom)")
# connect mean and trim per method with a thin line
for _, r in complete.iterrows():
    ax.plot([r["da"], r["da"]], [r["dg"], r["dt"]], color="0.7", linewidth=0.6, zorder=0)
    ax.annotate(SHORT[r["method"]],
                (r["da"], max(r["dg"], r["dt"])), xytext=(6, 4),
                textcoords="offset points", fontsize=7)

# incomplete methods: short vertical tick inside the axis
tick_y = ylo + 0.3
for i, (_, r) in enumerate(incomplete.iterrows()):
    ax.plot([r["da"], r["da"]], [tick_y, tick_y + 0.5], color="0.45", linewidth=1.0)
    yoffset = tick_y + 1.3 + (i % 2) * 0.9
    ax.annotate(SHORT[r["method"]],
                (r["da"], yoffset), ha="center", fontsize=6, color="0.35")

ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
ax.set_xlabel(r"attack-panel uplift (clarity pts, paired $\Delta$ vs.\ source)")
ax.set_ylabel(r"gold-panel uplift (clarity pts, paired $\Delta$)")
ax.legend(loc="upper left", frameon=False, fontsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_transfer_gap.pdf"))
plt.savefig(os.path.join(OUT, "fig1_transfer_gap.png"))
plt.close()
print("wrote fig1")

# ------------------------------------------------------------------
# Figure 2: per-gold-judge score per method (reversal driver)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.0))
method_order = ["naive", "injection_leadin", "lit_informed_tight"]
colors = {"naive": "#4c72b0", "injection_leadin": "#2a9d8f", "lit_informed_tight": "#e15759"}
xs = np.arange(len(GOLD))
width = 0.25
for i, m in enumerate(method_order):
    ys = []
    for j in GOLD:
        v = rw[(rw.method_slug == m) & (rw.source == j)]["value"]
        ys.append(float(v.mean()) if len(v) else np.nan)
    ax.bar(xs + (i - 1) * width, ys, width=width, label=SHORT[m], color=colors[m])
# baseline (original writer) per judge
baseline = [float(orig[orig.source == j]["value"].mean()) for j in GOLD]
ax.plot(xs, baseline, "k.--", markersize=5, linewidth=0.8, label="original writer")

# Highlight the lit_informed_tight / DeepSeek outlier
deepseek_idx = GOLD.index("deepseek-v3.2")
litt_idx = method_order.index("lit_informed_tight")
v = float(rw[(rw.method_slug == "lit_informed_tight") & (rw.source == "deepseek-v3.2")]["value"].mean())
ax.annotate(
    "18 pts\nbelow peers",
    xy=(xs[deepseek_idx] + (litt_idx - 1) * width, v),
    xytext=(xs[deepseek_idx] + (litt_idx - 1) * width, v - 14),
    ha="center", fontsize=7,
    arrowprops=dict(arrowstyle="->", color="0.3", linewidth=0.8),
)

ax.set_xticks(xs)
ax.set_xticklabels([GOLD_LABEL[j] for j in GOLD], fontsize=7.5)
ax.set_ylabel("mean clarity (0--100)")
ax.set_ylim(40, 100)
ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_per_judge.pdf"))
plt.savefig(os.path.join(OUT, "fig2_per_judge.png"))
plt.close()
print("wrote fig2")

# ------------------------------------------------------------------
# Figure 3: attack-panel ranking (all 10) with CI bars
# ------------------------------------------------------------------
ar = mres.sort_values("da").copy()
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ys = np.arange(len(ar))
ax.errorbar(ar["da"], ys,
            xerr=[ar["da"] - ar["alo"], ar["ahi"] - ar["da"]],
            fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.0, capsize=2, markersize=5)
# highlight methods with gold coverage
for i, (_, r) in enumerate(ar.iterrows()):
    if pd.notna(r["dg"]):
        ax.plot(r["da"], i, "o", color="#d62728", markersize=5, zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels([SHORT[m] for m in ar["method"]], fontsize=8)
ax.set_xlabel(r"attack-panel uplift (clarity pts), paired $\Delta$, 95% CI")
ax.grid(axis="x", linestyle=":", alpha=0.4)
# legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=6,
           label="attack only"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=6,
           label="gold complete"),
]
ax.legend(handles=legend_elems, loc="lower right", frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_attack_uplift.pdf"))
plt.savefig(os.path.join(OUT, "fig3_attack_uplift.png"))
plt.close()
print("wrote fig3")

# ------------------------------------------------------------------
# Figure 4: drift vs. attack-panel uplift
# ------------------------------------------------------------------
ndf = pd.read_sql_query(
    """
    SELECT p.document_id, p.origin_kind, p.method_slug, p.base_document_id,
           e.metric, e.value
    FROM paragraphs p JOIN evaluations e ON e.paragraph_id=p.document_id
    WHERE e.metric IN ('agreement_score_pred','word_count_ratio_vs_base','clarity_score_pred')
      AND (
        (p.origin_kind='original_writer'
         AND p.document_id IN (SELECT document_id FROM sampled_writers WHERE tag='main_20pct'))
        OR (p.origin_kind='rewrite'
         AND p.base_document_id IN (SELECT document_id FROM sampled_writers WHERE tag='main_20pct'))
      )
    """,
    con,
)
ndf["value"] = pd.to_numeric(ndf["value"], errors="coerce")
ag = ndf[ndf["metric"] == "agreement_score_pred"]
ag_orig = ag[ag["origin_kind"] == "original_writer"].set_index("document_id")["value"].to_dict()
ag_rw = ag[ag["origin_kind"] == "rewrite"].copy()
ag_rw["drift"] = (ag_rw["value"] - ag_rw["base_document_id"].map(ag_orig)).abs()
drift_mean = ag_rw.groupby("method_slug")["drift"].mean()

x = ar.set_index("method")["da"]
y = drift_mean.reindex(x.index)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.scatter(x, y, s=42, color="#1f77b4")
for m in x.index:
    ax.annotate(SHORT[m],
                (x[m], y[m]),
                xytext=(4, 2), textcoords="offset points", fontsize=7)
ax.set_xlabel("attack-panel uplift (clarity pts)")
ax.set_ylabel(r"stance drift: mean $|\Delta$ agreement $|$")
ax.set_ylim(0, float(y.max()) * 1.18)
ax.grid(linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_drift.pdf"))
plt.savefig(os.path.join(OUT, "fig4_drift.png"))
plt.close()
print("wrote fig4")

# ------------------------------------------------------------------
# dump key numbers
# ------------------------------------------------------------------
with open(os.path.join(OUT, "numbers.json"), "w") as f:
    json.dump({
        "attack_uplift": ar.to_dict(orient="records"),
        "method_uplift_ci": mres.to_dict(orient="records"),
        "n_orig_att": int(len(orig_att)), "n_orig_gold": int(len(orig_gold)),
        "base_attack_mean": float(orig_att["score"].mean()),
        "base_gold_mean": float(orig_gold["score"].mean()),
    }, f, indent=2, default=str)
print("wrote numbers.json")
