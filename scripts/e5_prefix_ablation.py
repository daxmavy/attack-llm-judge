"""Measure the impact of e5-large-v2 prefix choice on paragraph-pair cosine similarity.

Samples 500 random (original, rewrite) pairs from attack_rewrites × paragraphs,
embeds each pair under both "query: " and "passage: " prefixes, reports the
distribution of cos_sim values and the per-pair delta.

Fits on a shared GPU (~1.5GB for e5-large-v2).
"""
import argparse
import os
import random
import sqlite3
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
DB = "/home/max/attack-llm-judge/data/paragraphs.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel

    # Sample 500 diverse (original, rewrite) pairs: cover different methods/criteria/rewriters
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT r.text AS rewrite_text, p.text AS orig_text, r.method, r.criterion, r.rewriter_model
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE r.method != 'bon_candidate' AND r.method != 'original'
    """).fetchall()
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, min(args.n, len(rows)))
    originals = [r[1] for r in sampled]
    rewrites = [r[0] for r in sampled]
    print(f"sampled {len(sampled)} pairs from {len(rows)} non-candidate non-original rewrites")

    tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2", cache_dir="/workspace/hf_cache")
    mdl = AutoModel.from_pretrained("intfloat/e5-large-v2", cache_dir="/workspace/hf_cache",
                                     torch_dtype=torch.float32)
    mdl.to(args.device).eval()

    @torch.no_grad()
    def embed(texts, prefix, batch=32, max_len=512):
        embs = []
        for i in range(0, len(texts), batch):
            b = [prefix + t for t in texts[i:i+batch]]
            enc = tok(b, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(args.device)
            out = mdl(**enc)
            h = out.last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-6)
            embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0)

    for label, prefix in [("query", "query: "), ("passage", "passage: ")]:
        print(f"\n[embedding with prefix={prefix!r}]")
        eo = embed(originals, prefix)
        er = embed(rewrites, prefix)
        sims = np.sum(eo * er, axis=1)
        print(f"  n={len(sims)}  mean={sims.mean():.4f}  std={sims.std():.4f}")
        print(f"  min={sims.min():.4f}  p05={np.percentile(sims,5):.4f}  p25={np.percentile(sims,25):.4f}  "
              f"median={np.median(sims):.4f}  p75={np.percentile(sims,75):.4f}  p95={np.percentile(sims,95):.4f}  max={sims.max():.4f}")
        below_85 = (sims < 0.85).sum()
        print(f"  fraction < 0.85: {below_85}/{len(sims)} ({100*below_85/len(sims):.1f}%)")
        if label == "query":
            query_sims = sims
        else:
            passage_sims = sims

    # Per-pair delta
    delta = passage_sims - query_sims
    print(f"\n[per-pair delta = passage - query]")
    print(f"  mean={delta.mean():+.4f}  std={delta.std():.4f}")
    print(f"  min={delta.min():+.4f}  p05={np.percentile(delta,5):+.4f}  median={np.median(delta):+.4f}  p95={np.percentile(delta,95):+.4f}  max={delta.max():+.4f}")
    # Pearson/Spearman
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(query_sims, passage_sims)
    r_s, _ = spearmanr(query_sims, passage_sims)
    print(f"  pearson(query_sim, passage_sim) = {r_p:.4f}")
    print(f"  spearman(query_sim, passage_sim) = {r_s:.4f}")

    # Threshold behaviour
    print(f"\n[fidelity threshold analysis]")
    for thr in [0.80, 0.85, 0.87, 0.90]:
        q_below = (query_sims < thr).sum()
        p_below = (passage_sims < thr).sum()
        agree_below = ((query_sims < thr) & (passage_sims < thr)).sum()
        disagree = ((query_sims < thr) != (passage_sims < thr)).sum()
        print(f"  thr={thr}: query flags {q_below}, passage flags {p_below}, both agree below {agree_below}, "
              f"disagree {disagree} ({100*disagree/len(sims):.1f}%)")


if __name__ == "__main__":
    main()
