"""Fixed evaluation reward for cross-pilot comparison.

Formula:
  eval_score(rewrite, original, target_wc, judge_scores) =
      mean(judge_scores)                                        # 0..100
    - 100 * (word_count(rewrite)/target_wc - 1)^2               # quadratic length
    - 20  * max(0, 0.85 - cos_sim(embed(original), embed(rewrite)))   # fidelity

Applied post-hoc to every pilot's eval_summary.json so all pilots can be
ranked on the same objective, regardless of what each pilot's training-time
reward actually optimised.

Embedding backbone: `intfloat/e5-large-v2` (≈1.34 GB, bf16). Query prefix for e5:
we pass both sides as `"query: <text>"` since we're comparing sentence-to-sentence.

Usage:
    python3 fixed_eval_reward.py /path/to/pilot/eval_summary.json \
        [--judge-slugs judge_qwen7b,judge_llama8b]   # which 2 in-panel judges to use

Writes `fixed_eval.json` alongside the input and prints a summary.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
for line in open("/home/max/attack-llm-judge/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


EMBED_MODEL = "intfloat/e5-large-v2"

LEN_ALPHA = 100.0     # same as training-time default
FIDELITY_BETA = 20.0  # 0.10 below threshold = 2 reward points
FIDELITY_THRESHOLD = 0.85


def load_embedder():
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, cache_dir="/workspace/hf_cache",
                                         token=os.environ.get("HF_TOKEN"))
    model = AutoModel.from_pretrained(EMBED_MODEL, cache_dir="/workspace/hf_cache",
                                       dtype=torch.bfloat16, device_map="cuda",
                                       token=os.environ.get("HF_TOKEN"))
    model.eval()
    return tok, model


@torch.no_grad()
def embed_batch(tok, model, texts, prefix="query: ", batch_size=32, max_length=512):
    # Mean-pool over attention mask, as recommended by e5 authors.
    embs = []
    for i in range(0, len(texts), batch_size):
        b = [prefix + t for t in texts[i:i + batch_size]]
        enc = tok(b, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        out = model(**enc)
        h = out.last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # L2 normalise for cosine-via-dot
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-6)
        embs.append(pooled.cpu().numpy())
    return np.concatenate(embs, axis=0)


def word_count(s):
    return len(s.split())


def fixed_score(judge_mean, rewrite, original, target_wc, emb_orig, emb_rw):
    # Length penalty
    r = word_count(rewrite) / max(target_wc, 1)
    len_pen = LEN_ALPHA * (r - 1.0) ** 2
    # Fidelity penalty
    cos_sim = float(np.dot(emb_orig, emb_rw))
    fid_pen = FIDELITY_BETA * max(0.0, FIDELITY_THRESHOLD - cos_sim)
    return judge_mean - len_pen - fid_pen, {
        "judge_mean": judge_mean,
        "len_ratio": r,
        "len_penalty": len_pen,
        "cos_sim": cos_sim,
        "fidelity_penalty": fid_pen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_path", type=str, help="path to eval_summary.json from a pilot")
    ap.add_argument("--judge-slugs", type=str, default=None,
                    help="comma-separated judge slugs for the in-panel ensemble (overrides auto-detect)")
    args = ap.parse_args()

    summary_path = Path(args.summary_path)
    data = json.loads(summary_path.read_text())
    pre = data["pre_rewrites"]
    post = data["post_rewrites"]
    summary_in = data.get("summary", {})

    # Decide which judges are the in-panel pair (not held-out)
    if args.judge_slugs:
        in_panel = args.judge_slugs.split(",")
    else:
        in_panel = [jn for jn, s in summary_in.items()
                    if s.get("role", "train") != "held_out"]
    assert len(in_panel) == 2, f"expected 2 in-panel judges, got {in_panel}"
    print(f"in-panel judges: {in_panel}", flush=True)

    # Reconstruct originals + target_wcs. Newer pilots save these directly.
    if "eval_originals" in data:
        originals = data["eval_originals"]
        target_wcs = data["eval_word_counts"]
    else:
        # Legacy: re-query from DB using eval split logic. Needs n_propositions + n_train/n_eval.
        import sqlite3
        conn = sqlite3.connect("/home/max/attack-llm-judge/data/paragraphs.db")
        rows = conn.execute("""
            SELECT document_id, proposition, text, word_count
            FROM paragraphs
            WHERE origin_kind='original_writer' AND writer_is_top_decile=1
            ORDER BY proposition_id, document_id
        """).fetchall()
        # Assume 33 props, 130 train, 35 eval (fold 1/2/3 convention). If different, caller should upgrade.
        seen, filt = [], []
        for r in rows:
            if r[1] not in seen:
                if len(seen) >= 33: continue
                seen.append(r[1])
            filt.append(r)
        rows = [r for r in filt if r[1] in seen]
        eval_rows = rows[130:165]
        originals = [r[2] for r in eval_rows]
        target_wcs = [int(r[3]) for r in eval_rows]
        print(f"(legacy) reconstructed {len(originals)} originals from DB", flush=True)

    assert len(originals) == len(pre) == len(post), (len(originals), len(pre), len(post))

    # Per-prompt judge means from saved per-judge scores (need to look at `heldout_pre_scores` or
    # summary table — the summary has pre_mean/post_mean but not per-prompt). For now,
    # fall back to computing the mean from the summary numbers — which gives ONE number per side,
    # not per-prompt. That's OK if we also compute per-prompt later in the pilot script.
    # For this post-hoc pass, we approximate: use summary means as constants across all prompts.
    pre_jmean = sum(summary_in[j]["pre_mean"] for j in in_panel) / 2.0
    post_jmean = sum(summary_in[j]["post_mean"] for j in in_panel) / 2.0
    print(f"  pre in-panel judge mean: {pre_jmean:.2f}")
    print(f"  post in-panel judge mean: {post_jmean:.2f}")

    # Load embedder + score
    print(f"[fixed-eval] loading {EMBED_MODEL}...", flush=True)
    tok, model = load_embedder()
    emb_orig = embed_batch(tok, model, originals)
    emb_pre = embed_batch(tok, model, pre)
    emb_post = embed_batch(tok, model, post)
    print(f"embeddings ready", flush=True)

    pre_scores, pre_breakdown = [], []
    post_scores, post_breakdown = [], []
    for i in range(len(originals)):
        s, b = fixed_score(pre_jmean, pre[i], originals[i], target_wcs[i], emb_orig[i], emb_pre[i])
        pre_scores.append(s); pre_breakdown.append(b)
        s, b = fixed_score(post_jmean, post[i], originals[i], target_wcs[i], emb_orig[i], emb_post[i])
        post_scores.append(s); post_breakdown.append(b)

    pre_mean = sum(pre_scores) / len(pre_scores)
    post_mean = sum(post_scores) / len(post_scores)
    delta = post_mean - pre_mean
    # Also report raw in-panel judge-only delta (no length/fidelity penalties) — so we can distinguish
    # "fixed-eval improvement" from "judge-only improvement" when comparing pilots.
    judge_only_delta = post_jmean - pre_jmean
    print(f"\n=== cross-pilot comparison metrics ===")
    print(f"  [fixed_eval_score] pre={pre_mean:.2f} post={post_mean:.2f} delta={delta:+.2f}")
    print(f"  [judge_only_delta] pre={pre_jmean:.2f} post={post_jmean:.2f} delta={judge_only_delta:+.2f}")
    print(f"\nbreakdown averages (pre → post):")
    keys = ["len_ratio", "len_penalty", "cos_sim", "fidelity_penalty"]
    for k in keys:
        pm = sum(b[k] for b in pre_breakdown) / len(pre_breakdown)
        qm = sum(b[k] for b in post_breakdown) / len(post_breakdown)
        print(f"  {k:>18}: {pm:.3f}  →  {qm:.3f}")

    out = {
        "formula": f"judge_mean − {LEN_ALPHA}·(r−1)² − {FIDELITY_BETA}·max(0, {FIDELITY_THRESHOLD} − cos_sim)",
        "embed_model": EMBED_MODEL,
        "in_panel_judges": in_panel,
        "pre_scores": pre_scores,
        "post_scores": post_scores,
        "pre_breakdown": pre_breakdown,
        "post_breakdown": post_breakdown,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "delta": delta,
        "pre_judge_only_mean": pre_jmean,
        "post_judge_only_mean": post_jmean,
        "judge_only_delta": judge_only_delta,
    }
    out_path = summary_path.parent / "fixed_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
