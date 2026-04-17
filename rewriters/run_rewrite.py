"""Run the judge-free rewriters on baseline-eval documents and re-score them.

Reuses the 300 baseline docs (judge/results/merged_baseline.csv) so we
keep the original judge scores as the untouched comparison and only need
to score the rewrites. Both rewriter prompts are applied to the SAME
source docs so the two methods are directly comparable.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from judge.client import JUDGE_MODELS, call_judge, load_api_key
from judge.rubrics import SYSTEM_PROMPT as JUDGE_SYSTEM, build_prompt
from rewriters.rewrite_prompts import SYSTEM_PROMPT as REWRITER_SYSTEM, build_rewrite_prompt
from rewriters.rewriter_client import REWRITER_LABEL, REWRITER_MODEL, call_rewriter


REPO = Path(__file__).resolve().parent.parent
DOCS_CSV = REPO / "paul_data" / "prepared" / "documents.csv"
BASELINE_MERGED = REPO / "judge" / "results" / "merged_baseline.csv"
OUT_DIR = REPO / "rewriters" / "results"
OUT_DIR.mkdir(exist_ok=True)


def load_sample(n: int, seed: int) -> pd.DataFrame:
    merged = pd.read_csv(BASELINE_MERGED)
    docs = pd.read_csv(DOCS_CSV)
    src = (
        merged[["document_id", "paragraph_type", "model_name"]]
        .drop_duplicates(subset=["document_id"])
        .merge(docs[["document_id", "proposition", "document_text"]], on="document_id", how="left")
    )
    # Stratified subsample.
    cells = []
    groups = [
        ("writer", None),
        ("model", "anthropic/claude-sonnet-4"),
        ("model", "deepseek/deepseek-chat-v3-0324"),
        ("model", "openai/chatgpt-4o-latest"),
        ("edited", None),
    ]
    per = max(1, n // len(groups))
    for ptype, m in groups:
        if ptype == "writer" or ptype == "edited":
            sub = src[src["paragraph_type"] == ptype]
        else:
            sub = src[(src["paragraph_type"] == ptype) & (src["model_name"] == m)]
        sub = sub.sample(n=min(per, len(sub)), random_state=seed)
        cells.append(sub)
    out = pd.concat(cells, ignore_index=True)
    return out


def do_rewrites(sample: pd.DataFrame, api_key: str, max_workers: int) -> pd.DataFrame:
    tasks = []
    for _, row in sample.iterrows():
        for method in ("naive", "lit_informed"):
            user = build_rewrite_prompt(method, row["proposition"], row["document_text"])
            tasks.append((row["document_id"], method, REWRITER_SYSTEM, user))

    results = []
    t0 = time.time()

    def _go(t):
        doc_id, method, sysp, userp = t
        r = call_rewriter(sysp, userp, api_key, model_id=REWRITER_MODEL, max_tokens=400, temperature=0.7)
        return {
            "document_id": doc_id,
            "method": method,
            "rewrite_ok": r.ok,
            "rewrite_text": r.text,
            "rewrite_error": r.error,
            "rewrite_prompt_tokens": r.prompt_tokens,
            "rewrite_completion_tokens": r.completion_tokens,
        }

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_go, tasks), start=1):
            results.append(res)
            if i % 20 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  rewrite {i}/{len(tasks)}  elapsed={elapsed:.0f}s")
    return pd.DataFrame(results)


def do_judging(rewrites: pd.DataFrame, sample: pd.DataFrame, api_key: str, max_workers: int) -> pd.DataFrame:
    meta = sample.set_index("document_id")[["proposition"]].to_dict(orient="index")
    tasks = []
    for _, row in rewrites[rewrites["rewrite_ok"]].iterrows():
        prop = meta[row["document_id"]]["proposition"]
        para = row["rewrite_text"]
        if not para:
            continue
        for jname, jid in JUDGE_MODELS.items():
            for crit in ("clarity", "informativeness"):
                prompt = build_prompt(crit, prop, para)
                tasks.append((row["document_id"], row["method"], jname, jid, crit, prompt))
    print(f"  judge calls queued: {len(tasks)}")
    results = []
    t0 = time.time()

    def _go(t):
        doc_id, method, jname, jid, crit, prompt = t
        r = call_judge(jid, JUDGE_SYSTEM, prompt, api_key, max_tokens=250, temperature=0.0)
        return {
            "document_id": doc_id,
            "method": method,
            "judge": jname,
            "criterion": crit,
            "score": r.score,
            "ok": r.ok,
            "error": r.error,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
        }

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(_go, tasks), start=1):
            results.append(res)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  judge {i}/{len(tasks)}  elapsed={elapsed:.0f}s")
    return pd.DataFrame(results)


def run(n: int, seed: int, max_workers: int, tag: str) -> None:
    api_key = load_api_key()
    sample = load_sample(n, seed)
    print(f"Sampled {len(sample)} source documents:")
    print(sample.groupby(["paragraph_type", "model_name"], dropna=False).size())

    rewrites = do_rewrites(sample, api_key, max_workers)
    rewrites = rewrites.merge(sample[["document_id", "proposition", "document_text", "paragraph_type", "model_name"]],
                              on="document_id", how="left")
    rewrites["orig_words"] = rewrites["document_text"].str.split().str.len()
    rewrites["rewrite_words"] = rewrites["rewrite_text"].fillna("").str.split().str.len()
    rewrites.to_csv(OUT_DIR / f"rewrites_{tag}.csv", index=False)
    print(f"Rewrites saved. ok rate = {rewrites['rewrite_ok'].mean():.3f}")
    print("Word-count control:")
    print(rewrites.groupby("method").agg(
        orig_mean=("orig_words", "mean"),
        rewrite_mean=("rewrite_words", "mean"),
        ratio=("rewrite_words", lambda s: float((s / rewrites.loc[s.index, "orig_words"]).mean())),
    ))

    judged = do_judging(rewrites, sample, api_key, max_workers)
    judged.to_csv(OUT_DIR / f"rewrite_judge_scores_{tag}.csv", index=False)
    parse_ok = judged["ok"].mean()
    print(f"Judge parse OK rate on rewrites: {parse_ok:.3f}")

    # Pivot judge scores: one row per (doc, method, judge) with both criteria.
    wide = (
        judged[judged["ok"]]
        .pivot_table(index=["document_id", "method", "judge"], columns="criterion", values="score", aggfunc="first")
        .reset_index()
    )

    # Load baseline (original) judge scores for these same docs, same judges.
    baseline = pd.read_csv(BASELINE_MERGED)
    baseline = baseline[baseline["document_id"].isin(sample["document_id"])]
    baseline_rn = baseline[["document_id", "judge", "clarity", "informativeness",
                             "human_clarity", "human_informativeness",
                             "paragraph_type", "model_name"]].rename(
        columns={"clarity": "orig_clarity", "informativeness": "orig_informativeness"})

    combined = wide.merge(baseline_rn, on=["document_id", "judge"], how="left")
    combined["delta_clarity"] = combined["clarity"] - combined["orig_clarity"]
    combined["delta_informativeness"] = combined["informativeness"] - combined["orig_informativeness"]
    combined.to_csv(OUT_DIR / f"combined_{tag}.csv", index=False)

    # Summary by (method, judge, paragraph_type).
    summary_rows = []
    for (method, judge), g in combined.groupby(["method", "judge"]):
        summary_rows.append({
            "method": method, "judge": judge, "paragraph_type": "ALL",
            "n": len(g),
            "orig_clarity": float(g["orig_clarity"].mean()),
            "rewrite_clarity": float(g["clarity"].mean()),
            "delta_clarity": float(g["delta_clarity"].mean()),
            "orig_informativeness": float(g["orig_informativeness"].mean()),
            "rewrite_informativeness": float(g["informativeness"].mean()),
            "delta_informativeness": float(g["delta_informativeness"].mean()),
        })
        for ptype, gg in g.groupby("paragraph_type"):
            summary_rows.append({
                "method": method, "judge": judge, "paragraph_type": ptype,
                "n": len(gg),
                "orig_clarity": float(gg["orig_clarity"].mean()),
                "rewrite_clarity": float(gg["clarity"].mean()),
                "delta_clarity": float(gg["delta_clarity"].mean()),
                "orig_informativeness": float(gg["orig_informativeness"].mean()),
                "rewrite_informativeness": float(gg["informativeness"].mean()),
                "delta_informativeness": float(gg["delta_informativeness"].mean()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / f"summary_{tag}.csv", index=False)
    print("\n=== Mean score deltas vs originals (rewrite − original) ===")
    show_cols = ["method", "judge", "paragraph_type", "n", "delta_clarity", "delta_informativeness",
                 "orig_clarity", "rewrite_clarity", "orig_informativeness", "rewrite_informativeness"]
    print(summary_df[show_cols].to_string(index=False))

    # Cost.
    def _judge_cost(r):
        p = r["prompt_tokens"] or 0
        c = r["completion_tokens"] or 0
        if r["judge"] == "llama-3.3-70b":
            return p / 1e6 * 0.10 + c / 1e6 * 0.32
        return p / 1e6 * 0.10 + c / 1e6 * 0.40
    judge_cost = judged.apply(_judge_cost, axis=1).sum()
    # Qwen 2.5 72B Instruct OpenRouter: ~$0.13 in / $0.40 out per 1M.
    rewrite_cost = (
        rewrites["rewrite_prompt_tokens"].fillna(0).sum() / 1e6 * 0.13
        + rewrites["rewrite_completion_tokens"].fillna(0).sum() / 1e6 * 0.40
    )
    total = float(judge_cost) + float(rewrite_cost)
    print(f"\nApprox cost: rewrite=${rewrite_cost:.4f} judge=${judge_cost:.4f} total=${total:.4f}")

    (OUT_DIR / f"summary_{tag}.json").write_text(json.dumps({
        "rewriter": REWRITER_LABEL,
        "rewriter_model": REWRITER_MODEL,
        "judges": list(JUDGE_MODELS.keys()),
        "n_source_docs": int(len(sample)),
        "n_rewrites": int(len(rewrites)),
        "n_judge_calls": int(len(judged)),
        "rewrite_ok_rate": float(rewrites["rewrite_ok"].mean()),
        "judge_parse_ok_rate": float(parse_ok),
        "cost_rewrite_usd": float(rewrite_cost),
        "cost_judge_usd": float(judge_cost),
        "cost_total_usd": total,
        "seed": seed,
    }, indent=2))
    print(f"Saved outputs to {OUT_DIR}/ (tag={tag})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="source docs (stratified across 5 cells)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--tag", default="v1")
    args = p.parse_args()
    run(args.n, args.seed, args.max_workers, args.tag)
