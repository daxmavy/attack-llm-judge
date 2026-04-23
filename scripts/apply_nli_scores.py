"""Apply bidirectional-entailment NLI scoring to every row in attack_rewrites.

Model: MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (binary entail/not_entail, long-context).

For each rewrite, computes:
  nli_fwd = P(entail | premise=rewrite,  hypothesis=original)   — does the attack support the original?
  nli_bwd = P(entail | premise=original, hypothesis=rewrite)    — does the original support the attack?

Raw probabilities (no thresholding). Persisted to new table attack_nli_scores
(rewrite_id PRIMARY KEY). Idempotent — resumes by skipping rows already scored
unless --reset is passed.

Excludes bon_candidate by default (11424 pool rows per rewriter × criterion are
used only to pick bon_panel winners; pool-level NLI would waste ~3 GPU hours).
"""
import argparse
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
MODEL_ID = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
CACHE = "/workspace/hf_cache"


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attack_nli_scores (
            rewrite_id TEXT PRIMARY KEY,
            nli_fwd REAL NOT NULL,
            nli_bwd REAL NOT NULL,
            model_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


@torch.no_grad()
def entail_probs(tok, model, premises, hypotheses, entail_idx, batch=32, max_length=512):
    probs = []
    for i in range(0, len(premises), batch):
        p_b = premises[i:i + batch]
        h_b = hypotheses[i:i + batch]
        enc = tok(p_b, h_b, truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(model.device)
        logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1)[:, entail_idx].cpu().numpy()
        probs.extend(p.tolist())
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="re-score everything (drops the table first)")
    ap.add_argument("--include-candidates", action="store_true",
                    help="also score bon_candidate rows (default: skip; costs ~3 extra GPU hours)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--model", type=str, default=MODEL_ID)
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    if args.reset:
        conn.execute("DROP TABLE IF EXISTS attack_nli_scores")
        conn.commit()
        print("dropped attack_nli_scores", flush=True)
    ensure_table(conn)

    # Build the universe of rewrites to consider
    exclude_cand = "AND r.method != 'bon_candidate'" if not args.include_candidates else ""
    all_rows = conn.execute(f"""
        SELECT r.rewrite_id, r.text, p.text AS original_text
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE 1=1 {exclude_cand}
    """).fetchall()
    scored = {r[0] for r in conn.execute("SELECT rewrite_id FROM attack_nli_scores").fetchall()}
    todo = [(rid, rew, orig) for rid, rew, orig in all_rows if rid not in scored]
    print(f"[{time.strftime('%H:%M:%S')}] universe={len(all_rows)}  already_scored={len(scored)}  "
          f"todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    print(f"[{time.strftime('%H:%M:%S')}] loading NLI backbone {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, cache_dir=CACHE, dtype=torch.bfloat16,
    ).to("cuda").eval()
    _id2label = getattr(model.config, "id2label", {0: "entailment"})
    entail_idx = next((i for i, lbl in _id2label.items()
                        if str(lbl).lower().startswith("entail")), 0)
    print(f"  entail_idx={entail_idx}  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    CHUNK = 1024
    t0 = time.time()
    for chunk_start in range(0, len(todo), CHUNK):
        chunk = todo[chunk_start:chunk_start + CHUNK]
        rids = [r[0] for r in chunk]
        rews = [r[1] for r in chunk]
        origs = [r[2] for r in chunk]
        fwd = entail_probs(tok, model, rews, origs, entail_idx,
                           batch=args.batch, max_length=args.max_length)
        bwd = entail_probs(tok, model, origs, rews, entail_idx,
                           batch=args.batch, max_length=args.max_length)
        conn.executemany(
            "INSERT OR REPLACE INTO attack_nli_scores (rewrite_id, nli_fwd, nli_bwd, model_id) "
            "VALUES (?, ?, ?, ?)",
            [(rid, float(f), float(b), args.model) for rid, f, b in zip(rids, fwd, bwd)]
        )
        conn.commit()
        done = chunk_start + len(chunk)
        rate = done / max(1e-6, time.time() - t0)
        eta_min = (len(todo) - done) / max(1e-6, rate) / 60
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(todo)}  "
              f"fwd_mean={np.mean(fwd):.3f}  bwd_mean={np.mean(bwd):.3f}  "
              f"rate={rate:.1f}/s  eta={eta_min:.1f}min", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
