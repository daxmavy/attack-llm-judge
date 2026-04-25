"""BoN single-judge attack: argmax over the existing 11,424 bon_candidate pool
using ONE in-panel judge per fold, write winners as method='bon_panel_single'.

Fold rotation (encodes which single judge is the in-panel one):
  fold 1 → qwen95b   (held = llama8b + gemma9b)
  fold 2 → llama8b   (held = qwen95b + gemma9b)
  fold 3 → gemma9b   (held = qwen95b + llama8b)

Pure-SQL: no model inference. Reuses already-cached candidate scores from
attack_judge_scores. Idempotent (INSERT OR IGNORE; safe to re-run).

rewrite_id format: bon_single_{short}_f{fold}_{crit}_{doc_id}
(distinct prefix from existing bon_q95only_*, bon_panel hash-based, etc.)
"""
import argparse
import hashlib
import json
import sqlite3
import time

DB = "/home/max/attack-llm-judge/data/paragraphs.db"

REWRITERS = {
    "qwen25-15b": "Qwen/Qwen2.5-1.5B-Instruct",
    "lfm25-12b":  "LiquidAI/LFM2.5-1.2B-Instruct",
    "gemma3-1b":  "google/gemma-3-1b-it",
}

# fold → in-panel single judge slug (the JUDGE_SCORES.judge_slug value)
FOLD_TO_JUDGE = {
    1: "judge_qwen95b",
    2: "judge_llama8b",
    3: "judge_gemma9b",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be inserted, don't write")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    total_inserted = 0

    for short, rewriter_id in REWRITERS.items():
        for cri in ["clarity", "informativeness"]:
            for fold, judge_slug in FOLD_TO_JUDGE.items():
                # For each source_doc_id, pick the candidate with highest score
                # from this rewriter × criterion's pool.
                rows = cur.execute("""
                    SELECT
                      r.source_doc_id,
                      r.rewrite_id    AS cand_rid,
                      r.text          AS cand_text,
                      r.word_count    AS cand_wc,
                      s.score         AS cand_score
                    FROM attack_rewrites r
                    JOIN attack_judge_scores s
                      ON s.rewrite_id = r.rewrite_id
                     AND s.judge_slug  = ?
                     AND s.criterion   = ?
                    WHERE r.method = 'bon_candidate'
                      AND r.rewriter_model = ?
                      AND r.criterion = ?
                """, (judge_slug, cri, rewriter_id, cri)).fetchall()

                # Group by source_doc_id, take argmax
                best = {}
                for src, cand_rid, text, wc, score in rows:
                    if src not in best or score > best[src][3]:
                        best[src] = (cand_rid, text, wc, score)

                if not best:
                    print(f"[{short} × {cri} × fold{fold}] no candidates found — skipping",
                          flush=True)
                    continue

                config = {
                    "method": "bon_panel_single",
                    "fold": fold,
                    "criterion": cri,
                    "rewriter": rewriter_id,
                    "in_panel_single_judge": judge_slug,
                    "selection": "argmax_one_judge_over_bon_candidate_pool",
                }
                config_json = json.dumps(config, separators=(",", ":"))
                judge_panel = json.dumps([judge_slug], separators=(",", ":"))

                inserted = 0
                for src, (cand_rid, text, wc, score) in best.items():
                    rid = f"bon_single_{short}_f{fold}_{cri}_{src}"
                    cur.execute("""
                        INSERT OR IGNORE INTO attack_rewrites
                        (rewrite_id, source_doc_id, method, fold, criterion,
                         config_json, rewriter_model, judge_panel_json,
                         text, word_count, run_metadata_json)
                        VALUES (?, ?, 'bon_panel_single', ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rid, src, fold, cri, config_json, rewriter_id, judge_panel,
                          text, wc,
                          json.dumps({"selected_candidate_rid": cand_rid,
                                      "selected_candidate_score": score})))
                    inserted += cur.rowcount
                if not args.dry_run:
                    conn.commit()
                total_inserted += inserted
                print(f"[{short} × {cri} × fold{fold}] judge={judge_slug}  "
                      f"inserted={inserted}  best_mean={sum(b[3] for b in best.values())/len(best):.2f}",
                      flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] total inserted: {total_inserted} "
          f"({'DRY RUN' if args.dry_run else 'committed'})", flush=True)


if __name__ == "__main__":
    main()
