"""Apply the v2 agreement_score regressor to every row in attack_rewrites.

v2 model: /workspace/agreement_model_v2/final
  Trained with the new hold-out strategy (5 contr + 5 non-contr held-out
  propositions for val; 10% of remaining non-contr held out for test_prop_ood;
  10% paragraphs per (prop × origin_kind) for test_para_ood).

Writes to NEW table attack_agreement_scores_v2 (rewrite_id PRIMARY KEY) so the
existing v1 attack_agreement_scores table is untouched. Idempotent — resumes
by skipping rows already scored unless --reset is passed.
"""
import argparse
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
MODEL_DIR = "/workspace/agreement_model_v2/final"
MAX_LEN = 256
BATCH = 64


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attack_agreement_scores_v2 (
            rewrite_id TEXT PRIMARY KEY,
            score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="re-score everything")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    ensure_table(conn)

    if args.reset:
        conn.execute("DELETE FROM attack_agreement_scores_v2")
        conn.commit()

    rows = conn.execute("""
        SELECT r.rewrite_id, p.proposition, r.text
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE NOT EXISTS (
            SELECT 1 FROM attack_agreement_scores_v2 a WHERE a.rewrite_id = r.rewrite_id
        )
    """).fetchall()
    print(f"[{time.strftime('%H:%M:%S')}] to score: {len(rows)}", flush=True)
    if not rows:
        return

    print(f"[{time.strftime('%H:%M:%S')}] loading v2 regressor from {MODEL_DIR}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(args.device).eval()

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start : start + BATCH]
            rids = [r[0] for r in chunk]
            props = [r[1] for r in chunk]
            texts = [r[2] for r in chunk]
            enc = tok(props, texts, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(args.device)
            logits = model(**enc).logits.squeeze(-1).detach().cpu().numpy()
            scores = np.clip(logits.astype(np.float64), 0.0, 1.0)
            for rid, sc in zip(rids, scores):
                conn.execute(
                    "INSERT OR REPLACE INTO attack_agreement_scores_v2 (rewrite_id, score) VALUES (?, ?)",
                    (rid, float(sc)))
            done += len(chunk)
            if done % 2048 == 0 or done == len(rows):
                conn.commit()
                rate = done / (time.time() - t0)
                eta = (len(rows) - done) / max(rate, 1e-6) / 60
                print(f"  {done}/{len(rows)} ({rate:.1f}/s, ETA {eta:.1f} min)", flush=True)

    conn.commit()
    conn.close()
    print(f"[{time.strftime('%H:%M:%S')}] done", flush=True)


if __name__ == "__main__":
    main()
