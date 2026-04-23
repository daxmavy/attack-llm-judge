"""Score ModernBERT NLI on (original, rewrite) pairs from a base-model JSONL.

Companion to probe_rollouts_nli.py: same model, same 4 directions, but the
pairs come from `gen_base_rewrites.py` output (one JSON per line with
`original`, `rewrite`, `proposition`). This gives the "pre-GRPO baseline"
distribution for comparison.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    ap.add_argument("--jsonl",
                    default="/data/shil6647/attack-llm-judge/tmp/base_rewrites/base_rewrites.jsonl")
    ap.add_argument("--out",
                    default="/home/shil6647/attack-llm-judge/notebooks/base_rewrites_nli.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).open()]
    # drop anything truncated to <20 chars (still-running/borked reasoning)
    rows = [r for r in rows if len(r["rewrite"]) >= 20]
    print(f"[{time.strftime('%H:%M:%S')}] loaded {len(rows)} valid pairs from {args.jsonl}",
          flush=True)

    originals = [r["original"] for r in rows]
    rewrites = [r["rewrite"] for r in rows]
    props = [r["proposition"] for r in rows]

    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE, dtype=torch.bfloat16
    ).to("cuda").eval()

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] scoring 4 directions", flush=True)
    fwd = batched_entail(tok, model, rewrites, originals)
    bwd = batched_entail(tok, model, originals, rewrites)
    prop_fwd = batched_entail(tok, model, rewrites, props)
    orig_prop = batched_entail(tok, model, originals, props)
    print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.1f}s", flush=True)

    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append({
            "document_id": r.get("document_id"),
            "proposition": r["proposition"][:120],
            "word_count": r.get("word_count"),
            "rewrite_chars": len(r["rewrite"]),
            "fwd_rew_to_orig": fwd[i],
            "bwd_orig_to_rew": bwd[i],
            "rewrite_to_prop": prop_fwd[i],
            "orig_to_prop": orig_prop[i],
            "stance_gap": abs(prop_fwd[i] - orig_prop[i]),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"model": MODEL_ID, "n": len(out_rows), "rows": out_rows}, f, indent=2)

    def stat(xs, name):
        a = np.asarray(xs, dtype=np.float64)
        print(f"  {name:24s} N={len(a):3d} mean={a.mean():.3f} p10={np.percentile(a,10):.3f} "
              f"med={np.median(a):.3f} p90={np.percentile(a,90):.3f} "
              f"min={a.min():.3f} max={a.max():.3f}")

    print(f"\n=== base-model rewrites N={len(out_rows)} ===")
    stat(fwd, "fwd (rewrite→orig)")
    stat(bwd, "bwd (orig→rewrite)")
    stat(prop_fwd, "rewrite→prop")
    stat(orig_prop, "orig→prop")
    stat([r["stance_gap"] for r in out_rows], "|stance_gap|")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
