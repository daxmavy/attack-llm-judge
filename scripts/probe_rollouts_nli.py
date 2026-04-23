"""Probe NLI entailment on GRPO-rollout (rewrite, original-candidate) pairs.

Rollouts only record the proposition, not the source paragraph. There are
multiple training paragraphs per proposition. For each rollout we compute
NLI entailment against ALL candidate originals with the same proposition
(pulled from the SQLite training slice the rewriter actually saw:
`origin_kind='original_writer' AND writer_is_top_decile=1`) and take the max.

That gives "best-case entailment with any plausible original the rewriter
might have seen". Useful to characterize how the current /no_think clean
pipeline's rewrites score under MoritzLaurer/ModernBERT-large-zeroshot-v2.0.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"

MODEL_ID = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
CACHE = "/data/shil6647/attack-llm-judge/hf_cache"
ENT_IDX = 0


def batched_entail(tok, model, premises, hypotheses, batch=32, max_length=512):
    probs = []
    for i in range(0, len(premises), batch):
        p_b = premises[i : i + batch]
        h_b = hypotheses[i : i + batch]
        enc = tok(p_b, h_b, truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1)[:, ENT_IDX].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", default=[
        "/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_8b_fold1_clarity/rollouts.jsonl",
        "/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_8b_fold1_clarity/rollouts.no-embed-sim.jsonl",
        "/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_8b_fold1_clarity/rollouts.oom-dp.jsonl",
    ])
    ap.add_argument("--out", default="/home/shil6647/attack-llm-judge/notebooks/rollouts_nli.json")
    args = ap.parse_args()

    # Load the training-subset paragraphs (same slice the rewriter saw: SQLite
    # `origin_kind='original_writer' AND writer_is_top_decile=1`, matching
    # training/scripts/run_pilot_len_pen.py::load_data()).
    conn = sqlite3.connect(DB)
    db_rows = conn.execute("""
        SELECT document_id, proposition, text, word_count
        FROM paragraphs
        WHERE origin_kind='original_writer' AND writer_is_top_decile=1
        ORDER BY proposition_id, document_id
    """).fetchall()
    conn.close()
    # Group by proposition
    from collections import defaultdict
    by_prop = defaultdict(list)
    for doc_id, prop, text, wc in db_rows:
        by_prop[prop].append({"document_id": doc_id, "proposition": prop,
                              "text": text, "word_count": wc})

    # Load rollouts from all runs
    rollouts = []
    for rp in args.rollouts:
        p = Path(rp)
        if not p.exists():
            continue
        tag = p.stem
        for line in p.open():
            r = json.loads(line)
            r["_run"] = tag
            rollouts.append(r)
    print(f"[{time.strftime('%H:%M:%S')}] loaded {len(rollouts)} rollouts across {len(args.rollouts)} files",
          flush=True)

    # Map each rollout to candidate originals sharing its proposition
    present = 0
    missing = 0
    for r in rollouts:
        cands = by_prop.get(r["proposition"], [])
        r["_candidates"] = cands
        if cands:
            present += 1
        else:
            missing += 1
    print(f"[{time.strftime('%H:%M:%S')}] matched {present} rollouts to candidate paragraphs "
          f"({missing} had no matching proposition in train)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE, dtype=torch.bfloat16
    ).to("cuda").eval()

    # Flatten: for each rollout × each candidate, score fwd (rewrite -> original),
    # bwd (original -> rewrite), plus rewrite→prop and original→prop for stance gap.
    fwd_rows, bwd_rows, prop_rows, orig_prop_rows, cand_ix = [], [], [], [], []
    rew_texts, orig_texts, prop_texts = [], [], []
    for i, r in enumerate(rollouts):
        for j, cand in enumerate(r["_candidates"]):
            rew_texts.append(r["rewrite"])
            orig_texts.append(cand["text"])
            prop_texts.append(r["proposition"])
            cand_ix.append((i, j))

    if not rew_texts:
        print("no (rollout, candidate) pairs to score — exiting")
        return

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] scoring {len(rew_texts)} pairs in 4 directions", flush=True)
    fwd = batched_entail(tok, model, rew_texts, orig_texts)
    bwd = batched_entail(tok, model, orig_texts, rew_texts)
    prop_fwd = batched_entail(tok, model, rew_texts, prop_texts)
    orig_prop = batched_entail(tok, model, orig_texts, prop_texts)
    print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.1f}s", flush=True)

    # Group results back per rollout: take MAX fwd over candidates as "best-match original".
    per_rollout = {}
    for k, (i, j) in enumerate(cand_ix):
        per_rollout.setdefault(i, []).append({
            "cand_idx": j,
            "fwd": fwd[k], "bwd": bwd[k],
            "rewrite_to_prop": prop_fwd[k],
            "orig_to_prop": orig_prop[k],
        })

    summary = []
    for i, r in enumerate(rollouts):
        cand_scores = per_rollout.get(i, [])
        if not cand_scores:
            continue
        best = max(cand_scores, key=lambda d: d["fwd"])
        summary.append({
            "run": r["_run"],
            "step": r["step"],
            "rollout_idx": r["rollout_idx"],
            "ensemble_judge": r["ensemble_judge"],
            "length_ratio": r["length_ratio"],
            "proposition": r["proposition"][:80],
            "n_candidates": len(cand_scores),
            "max_fwd": best["fwd"],
            "max_bwd": max(d["bwd"] for d in cand_scores),
            "mean_fwd": float(np.mean([d["fwd"] for d in cand_scores])),
            "rewrite_to_prop": best["rewrite_to_prop"],
            "orig_to_prop": best["orig_to_prop"],
            "stance_gap": abs(best["rewrite_to_prop"] - best["orig_to_prop"]),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": MODEL_ID, "n": len(summary), "rows": summary}, f, indent=2)

    def stat(xs, name):
        a = np.asarray(xs, dtype=np.float64)
        print(f"  {name:22s} N={len(a):3d} mean={a.mean():.3f} p10={np.percentile(a,10):.3f} "
              f"med={np.median(a):.3f} p90={np.percentile(a,90):.3f} min={a.min():.3f} max={a.max():.3f}")

    print("\n=== overall (max over candidate-paragraph matches per rollout) ===")
    stat([s["max_fwd"] for s in summary], "max_fwd (rewrite→orig)")
    stat([s["max_bwd"] for s in summary], "max_bwd (orig→rewrite)")
    stat([s["mean_fwd"] for s in summary], "mean_fwd (vs all cands)")
    stat([s["rewrite_to_prop"] for s in summary], "rewrite→prop")
    stat([s["orig_to_prop"] for s in summary], "orig→prop")
    stat([s["stance_gap"] for s in summary], "|stance_gap|")

    print("\n=== by training step ===")
    from collections import defaultdict
    by_step = defaultdict(list)
    for s in summary:
        by_step[s["step"]].append(s)
    for step in sorted(by_step.keys()):
        group = by_step[step]
        v = [g["max_fwd"] for g in group]
        sg = [g["stance_gap"] for g in group]
        ej = [g["ensemble_judge"] for g in group]
        print(f"  step {step:3d} N={len(group):2d} max_fwd mean={np.mean(v):.3f} "
              f"stance_gap mean={np.mean(sg):.3f}  ej mean={np.mean(ej):.1f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
