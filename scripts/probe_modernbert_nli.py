"""Probe MoritzLaurer/ModernBERT-large-zeroshot-v2.0 on (original, rewrite) pairs.

Goal: characterize the NLI entailment-probability distribution so we can decide
how (and whether) to feed it into the GRPO reward as a semantic-fidelity term.

For each pair we compute entailment probability in two directions:
  fwd = P(entail | premise=rewrite,  hypothesis=original)   — does the attack paragraph support the original?
  bwd = P(entail | premise=original, hypothesis=rewrite)    — does the original support the attack?

Stratified sample across the 6 attack methods in `attack_rewrites`.
"""
import argparse
import json
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove full <think>...</think> blocks plus any orphan prefix (common with truncated outputs)."""
    if text is None:
        return ""
    t = _THINK_RE.sub("", text)
    # Orphan <think> with no closing tag — drop everything up to the first double-newline we find
    # (keeps the post-thought paragraph). If no break, drop everything after <think>.
    low = t.lower()
    if "<think>" in low:
        # keep only text after "</think>" if present, else drop up to first \n\n
        idx = low.find("</think>")
        if idx != -1:
            t = t[idx + len("</think>"):]
        else:
            # no closing tag — unsafe, return empty to avoid measuring reasoning text as rewrite
            t = ""
    return t.strip()

MODEL_ID = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
CACHE = "/data/shil6647/attack-llm-judge/hf_cache"
DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
# Entailment is label 0 per config.id2label.
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
        # float32 for stable softmax
        p = torch.softmax(logits.float(), dim=-1)[:, ENT_IDX].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-method", type=int, default=100)
    ap.add_argument("--out", type=str,
                    default="/home/shil6647/attack-llm-judge/notebooks/modernbert_nli_probe.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"[{time.strftime('%H:%M:%S')}] loading model {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, cache_dir=CACHE, dtype=torch.bfloat16
    ).to("cuda").eval()
    print(f"[{time.strftime('%H:%M:%S')}] model loaded, VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB",
          flush=True)

    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT r.method, r.rewrite_id, r.criterion, r.fold,
               p.proposition, p.text AS original, r.text AS rewrite
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
    """).fetchall()
    conn.close()

    by_method = defaultdict(list)
    for row in rows:
        by_method[row[0]].append(row)
    print(f"[{time.strftime('%H:%M:%S')}] method counts: "
          + ", ".join(f"{m}={len(v)}" for m, v in sorted(by_method.items())), flush=True)

    sample = []
    for m, rs in sorted(by_method.items()):
        rng.shuffle(rs)
        sample.extend(rs[: args.per_method])
    print(f"[{time.strftime('%H:%M:%S')}] sampled {len(sample)} pairs", flush=True)

    methods = [r[0] for r in sample]
    rids = [r[1] for r in sample]
    criteria = [r[2] for r in sample]
    folds = [r[3] for r in sample]
    props = [r[4] for r in sample]
    originals = [r[5] for r in sample]
    rewrites_raw = [r[6] for r in sample]
    rewrites = [strip_think(t) for t in rewrites_raw]

    # Report how many rewrites had reasoning traces (nontrivial difference after stripping)
    raw_lens = np.array([len(r or "") for r in rewrites_raw])
    stripped_lens = np.array([len(r) for r in rewrites])
    had_think = int(np.sum(stripped_lens < raw_lens - 5))
    empty_after = int(np.sum(stripped_lens < 20))
    print(f"[{time.strftime('%H:%M:%S')}] strip_think: {had_think}/{len(sample)} had a think block "
          f"(mean raw {raw_lens.mean():.0f} → stripped {stripped_lens.mean():.0f} chars); "
          f"{empty_after} are <20 chars after strip (dropped from stats)", flush=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] scoring fwd: P(entail | rewrite -> original)", flush=True)
    fwd = batched_entail(tok, model, rewrites, originals, batch=args.batch)
    print(f"[{time.strftime('%H:%M:%S')}] scoring bwd: P(entail | original -> rewrite)", flush=True)
    bwd = batched_entail(tok, model, originals, rewrites, batch=args.batch)
    # Plus: does the rewrite entail the bare proposition claim?
    print(f"[{time.strftime('%H:%M:%S')}] scoring prop: P(entail | rewrite -> proposition)", flush=True)
    prop_fwd = batched_entail(tok, model, rewrites, props, batch=args.batch)
    # And does the original paragraph entail the proposition (baseline)?
    print(f"[{time.strftime('%H:%M:%S')}] scoring orig_prop: P(entail | original -> proposition)", flush=True)
    orig_prop = batched_entail(tok, model, originals, props, batch=args.batch)
    print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.1f}s", flush=True)

    out = {
        "model": MODEL_ID,
        "per_method": args.per_method,
        "n": len(sample),
        "rows": [
            dict(
                method=methods[i],
                rewrite_id=rids[i],
                criterion=criteria[i],
                fold=folds[i],
                proposition=props[i],
                original=originals[i],
                rewrite=rewrites[i],
                fwd_rewrite_to_original=fwd[i],
                bwd_original_to_rewrite=bwd[i],
                rewrite_to_proposition=prop_fwd[i],
                original_to_proposition=orig_prop[i],
            )
            for i in range(len(sample))
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # summary
    def stat(xs):
        a = np.asarray(xs, dtype=np.float64)
        return dict(mean=float(a.mean()), median=float(np.median(a)),
                    p10=float(np.percentile(a, 10)),
                    p90=float(np.percentile(a, 90)),
                    min=float(a.min()), max=float(a.max()))

    print("\n=== overall distribution (N={}) ===".format(len(fwd)))
    for name, xs in [("fwd(rewrite->original)", fwd),
                     ("bwd(original->rewrite)", bwd),
                     ("rewrite->proposition", prop_fwd),
                     ("original->proposition", orig_prop)]:
        s = stat(xs)
        print(f"  {name:28s} mean={s['mean']:.3f} p10={s['p10']:.3f} "
              f"med={s['median']:.3f} p90={s['p90']:.3f}")

    print("\n=== by method (fwd = rewrite entails original) ===")
    for m in sorted(set(methods)):
        vals = [fwd[i] for i in range(len(methods)) if methods[i] == m]
        s = stat(vals)
        print(f"  {m:22s} N={len(vals):3d} mean={s['mean']:.3f} "
              f"p10={s['p10']:.3f} med={s['median']:.3f} p90={s['p90']:.3f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
