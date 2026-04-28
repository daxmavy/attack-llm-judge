"""BoN selection variant: pick the candidate with the highest mean across all
three in-sample judges (qwen95b, llama8b, gemma9b) at the candidate's own
criterion, per (rewriter, criterion, source_doc).

Uses pre-computed candidate-level judge scores. Pure SQL — no API calls, no GPU.

NOTE: this rule is information-leaky for held-out evaluation — the judge that
would otherwise be held-out for a given fold is one of the three judges used
to pick the winner. Treat results as a *ceiling* on what BoN selection can
achieve given full panel knowledge, not as held-out generalisation.

For analysis-pipeline consistency we replicate the same fold-independent winner
across folds 1, 2, 3.

Method tag: 'bon_panel_mean3'. rewrite_id prefix: 'bon_m3_'.
"""
import argparse
import json
import sqlite3
import time

DB = "/home/max/attack-llm-judge/data/paragraphs.db"

REWRITERS = {
    "qwen25-15b": "Qwen/Qwen2.5-1.5B-Instruct",
    "lfm25-12b":  "LiquidAI/LFM2.5-1.2B-Instruct",
    "gemma3-1b":  "google/gemma-3-1b-it",
}
FOLDS = (1, 2, 3)
PANEL = ("judge_qwen95b", "judge_llama8b", "judge_gemma9b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    total_inserted = 0

    for short, rewriter_id in REWRITERS.items():
        for cri in ("clarity", "informativeness"):
            # Per-candidate mean of the 3 in-sample judges at own criterion.
            # HAVING COUNT(*) = 3 ensures all three judges scored it.
            rows = cur.execute(f"""
                SELECT
                    r.source_doc_id,
                    r.rewrite_id    AS cand_rid,
                    r.text          AS cand_text,
                    r.word_count    AS cand_wc,
                    AVG(s.score)    AS mean3,
                    GROUP_CONCAT(s.score) AS scores_csv
                FROM attack_rewrites r
                JOIN attack_judge_scores s
                  ON s.rewrite_id = r.rewrite_id
                 AND s.judge_slug IN ({','.join('?'*len(PANEL))})
                 AND s.criterion  = r.criterion
                WHERE r.method = 'bon_candidate'
                  AND r.rewriter_model = ?
                  AND r.criterion = ?
                GROUP BY r.source_doc_id, r.rewrite_id
                HAVING COUNT(*) = 3
            """, (*PANEL, rewriter_id, cri)).fetchall()

            best = {}
            for src, cand_rid, text, wc, mean3, scores_csv in rows:
                if src not in best or mean3 > best[src][4]:
                    best[src] = (cand_rid, text, wc, scores_csv, mean3)

            if not best:
                print(f"[{short} × {cri}] no candidates with full 3-judge coverage — skipping", flush=True)
                continue

            inserted = 0
            for fold in FOLDS:
                config = {
                    "method": "bon_panel_mean3",
                    "fold": fold,
                    "criterion": cri,
                    "rewriter": rewriter_id,
                    "panel": list(PANEL),
                    "selection": "argmax(mean(qwen95b, llama8b, gemma9b) at own criterion)",
                    "note": "all-3-judges; information-leaky for held-out eval; fold-replicated",
                }
                config_json = json.dumps(config, separators=(",", ":"))
                judge_panel = json.dumps(list(PANEL), separators=(",", ":"))

                for src, (cand_rid, text, wc, scores_csv, mean3) in best.items():
                    rid = f"bon_m3_{short}_f{fold}_{cri}_{src}"
                    meta = json.dumps({
                        "selected_candidate_rid": cand_rid,
                        "selected_mean3": mean3,
                        "selected_scores_csv": scores_csv,
                    })
                    cur.execute("""
                        INSERT OR IGNORE INTO attack_rewrites
                        (rewrite_id, source_doc_id, method, fold, criterion,
                         config_json, rewriter_model, judge_panel_json,
                         text, word_count, run_metadata_json)
                        VALUES (?, ?, 'bon_panel_mean3', ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rid, src, fold, cri, config_json, rewriter_id, judge_panel,
                          text, wc, meta))
                    inserted += cur.rowcount

            if not args.dry_run:
                conn.commit()
            total_inserted += inserted
            mean_score = sum(b[4] for b in best.values()) / len(best)
            print(f"[{short} × {cri}] inserted={inserted} (3 folds × {len(best)} sources)  "
                  f"mean(mean3)={mean_score:.2f}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] total inserted: {total_inserted} "
          f"({'DRY RUN' if args.dry_run else 'committed'})", flush=True)


if __name__ == "__main__":
    main()
