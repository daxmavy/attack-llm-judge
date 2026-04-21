"""Run the LLM judges on a sample of paul_data documents and compare to humans.

We use the two in-scope judges (Llama 3.3 70B, Gemini 2.0 Flash) — not the
held-out GPT-4o-mini / Claude Haiku 3.5 judges.

Design:
- Stratified sample of paul_data/prepared/documents.csv across
  paragraph_type (writer / model / edited) and model_name so both
  human- and AI-written paragraphs are represented.
- Score each sampled document on clarity and informativeness with each
  judge (2 judges x 2 criteria x N docs = 4N calls).
- Compare to human means (paragraph_clarity, paragraph_informativeness
  aggregated per document from main_phase_2/annotations.csv).
- Report Pearson/Spearman + MAE per (judge, criterion, paragraph_type),
  plus simple judge-judge agreement.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from judge.client import JUDGE_MODELS, call_judge, load_api_key
from judge.rubrics import SYSTEM_PROMPT, build_prompt


REPO = Path(__file__).resolve().parent.parent
DOCS_CSV = REPO / "paul_data" / "prepared" / "documents.csv"
ANN_CSV = REPO / "paul_data" / "main_phase_2" / "annotations.csv"
RESULTS_DIR = REPO / "judge" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_dataset(n_per_cell: int, seed: int) -> pd.DataFrame:
    docs = pd.read_csv(DOCS_CSV)
    ann = pd.read_csv(ANN_CSV)
    key = ["writer_id", "proposition_id", "paragraph_type", "model_name", "model_input_condition"]
    ann_agg = ann.groupby(key, dropna=False).agg(
        n_ratings=("paragraph_clarity", "size"),
        human_clarity=("paragraph_clarity", "mean"),
        human_informativeness=("paragraph_informativeness", "mean"),
    ).reset_index()
    df = docs.merge(ann_agg, on=key, how="inner")

    # Stratify: writer (no model) + each model for paragraph_type=model + edited.
    cells = []
    cells.append(df[df["paragraph_type"] == "writer"].sample(n=n_per_cell, random_state=seed))
    for m in sorted(df[df["paragraph_type"] == "model"]["model_name"].dropna().unique()):
        sub = df[(df["paragraph_type"] == "model") & (df["model_name"] == m)]
        cells.append(sub.sample(n=min(n_per_cell, len(sub)), random_state=seed))
    edited = df[df["paragraph_type"] == "edited"]
    cells.append(edited.sample(n=min(n_per_cell, len(edited)), random_state=seed))
    sample = pd.concat(cells, ignore_index=True)
    return sample


def score_one(args):
    model_name, model_id, criterion, doc_id, proposition, paragraph, api_key = args
    prompt = build_prompt(criterion, proposition, paragraph)
    result = call_judge(model_id, SYSTEM_PROMPT, prompt, api_key, max_tokens=250, temperature=0.0)
    return {
        "document_id": doc_id,
        "judge": model_name,
        "criterion": criterion,
        "score": result.score,
        "ok": result.ok,
        "error": result.error,
        "reasoning": result.reasoning,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def run(n_per_cell: int, seed: int, max_workers: int, out_tag: str) -> None:
    api_key = load_api_key()
    sample = build_dataset(n_per_cell, seed)
    print(f"Sampled {len(sample)} documents across:")
    print(sample.groupby(["paragraph_type", "model_name"], dropna=False).size())

    tasks = []
    for _, row in sample.iterrows():
        for jname, jid in JUDGE_MODELS.items():
            for crit in ("clarity", "informativeness"):
                tasks.append((jname, jid, crit, row["document_id"], row["proposition"], row["document_text"], api_key))
    print(f"Total judge calls: {len(tasks)}")

    t0 = time.time()
    rows = []
    total_in = 0
    total_out = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(score_one, tasks), start=1):
            rows.append(res)
            total_in += res["prompt_tokens"] or 0
            total_out += res["completion_tokens"] or 0
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  {i}/{len(tasks)}  elapsed={elapsed:.0f}s  rate={rate:.1f} call/s  "
                      f"tokens in={total_in} out={total_out}")
    scores = pd.DataFrame(rows)

    # Cost estimate (per OpenRouter list prices from plan.md):
    # Llama 3.3 70B: $0.10/$0.32 per 1M, Gemini 2.0 Flash: $0.10/$0.40 per 1M.
    # Compute roughly using the per-row tokens and judge label.
    def row_cost(r):
        if r["judge"] == "llama-3.3-70b":
            return (r["prompt_tokens"] or 0) / 1e6 * 0.10 + (r["completion_tokens"] or 0) / 1e6 * 0.32
        return (r["prompt_tokens"] or 0) / 1e6 * 0.10 + (r["completion_tokens"] or 0) / 1e6 * 0.40
    scores["cost_usd"] = scores.apply(row_cost, axis=1)
    total_cost = scores["cost_usd"].sum()
    ok_rate = scores["ok"].mean()
    print(f"Parse OK rate: {ok_rate:.3f}")
    print(f"Approximate cost: ${total_cost:.4f}")

    # Pivot to one row per (doc, judge) with both criteria.
    wide = (
        scores[scores["ok"]]
        .pivot_table(index=["document_id", "judge"], columns="criterion", values="score", aggfunc="first")
        .reset_index()
    )
    merged = sample.merge(wide, on="document_id", how="inner")

    # Compute metrics.
    metrics = []
    for judge in JUDGE_MODELS:
        sub = merged[merged["judge"] == judge]
        for crit, human_col in [("clarity", "human_clarity"), ("informativeness", "human_informativeness")]:
            if crit not in sub.columns:
                continue
            ok = sub.dropna(subset=[crit, human_col])
            if len(ok) < 5:
                continue
            p = pearsonr(ok[crit], ok[human_col])[0]
            s = spearmanr(ok[crit], ok[human_col]).correlation
            mae = float(np.mean(np.abs(ok[crit] - ok[human_col])))
            metrics.append({
                "judge": judge, "criterion": crit, "n": len(ok),
                "pearson": float(p), "spearman": float(s), "mae": mae,
                "judge_mean": float(ok[crit].mean()), "human_mean": float(ok[human_col].mean()),
            })
            for ptype, pt_df in ok.groupby("paragraph_type"):
                if len(pt_df) < 5:
                    continue
                metrics.append({
                    "judge": judge, "criterion": crit, "n": len(pt_df),
                    "paragraph_type": ptype,
                    "pearson": float(pearsonr(pt_df[crit], pt_df[human_col])[0]),
                    "spearman": float(spearmanr(pt_df[crit], pt_df[human_col]).correlation),
                    "mae": float(np.mean(np.abs(pt_df[crit] - pt_df[human_col]))),
                    "judge_mean": float(pt_df[crit].mean()),
                    "human_mean": float(pt_df[human_col].mean()),
                })

    metrics_df = pd.DataFrame(metrics)
    print("\n=== Correlation with human means ===")
    print(metrics_df.to_string(index=False))

    # Judge-judge agreement on same documents.
    jj = merged.pivot_table(index="document_id", columns="judge",
                             values=["clarity", "informativeness"], aggfunc="first")
    jj_rows = []
    for crit in ("clarity", "informativeness"):
        sub = jj[crit].dropna()
        if sub.shape[1] == 2 and len(sub) >= 5:
            p = pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])[0]
            jj_rows.append({"criterion": crit, "pearson_between_judges": float(p), "n": len(sub)})
    if jj_rows:
        print("\n=== Judge-judge agreement ===")
        print(pd.DataFrame(jj_rows).to_string(index=False))

    # Persist.
    scores.to_csv(RESULTS_DIR / f"judge_scores_{out_tag}.csv", index=False)
    merged[["document_id", "paragraph_type", "model_name", "judge", "clarity", "informativeness",
            "human_clarity", "human_informativeness", "n_ratings"]].to_csv(
        RESULTS_DIR / f"merged_{out_tag}.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / f"metrics_{out_tag}.csv", index=False)
    summary = {
        "n_docs_sampled": int(len(sample)),
        "n_judge_calls": int(len(tasks)),
        "parse_ok_rate": float(ok_rate),
        "approx_cost_usd": float(total_cost),
        "seed": seed,
        "n_per_cell": n_per_cell,
        "judges": list(JUDGE_MODELS.keys()),
    }
    (RESULTS_DIR / f"summary_{out_tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {RESULTS_DIR}/ (tag={out_tag})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-cell", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--tag", default="baseline")
    args = p.parse_args()
    run(args.n_per_cell, args.seed, args.max_workers, args.tag)
