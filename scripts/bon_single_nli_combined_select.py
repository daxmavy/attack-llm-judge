"""BoN selection variant: single-judge × NLI combined, equally weighted.

Mirrors `bon_nli_combined_select.py` (which uses the 2-judge in-panel mean) but
substitutes the single-judge rotation used by `bon_single_judge_select.py`:

    combined = 0.5 * single_judge_score + 0.5 * 100 * (nli_fwd + nli_bwd) / 2

Fold rotation:
  fold 1 → qwen95b   (held = llama8b + gemma9b)
  fold 2 → llama8b   (held = qwen95b + gemma9b)
  fold 3 → gemma9b   (held = qwen95b + llama8b)

Pure-SQL: reuses cached candidate scores + 100% NLI coverage. Idempotent.

Method tag: 'bon_panel_single_nli'. rewrite_id prefix: 'bon_snli_*'.
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

FOLD_TO_JUDGE = {
    1: "judge_qwen95b",
    2: "judge_llama8b",
    3: "judge_gemma9b",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    total_inserted = 0

    for short, rewriter_id in REWRITERS.items():
        for cri in ["clarity", "informativeness"]:
            for fold, judge_slug in FOLD_TO_JUDGE.items():
                rows = cur.execute("""
                    SELECT
                      r.source_doc_id, r.rewrite_id, r.text, r.word_count,
                      s.score AS j_score,
                      n.nli_fwd, n.nli_bwd
                    FROM attack_rewrites r
                    JOIN attack_judge_scores s
                      ON s.rewrite_id = r.rewrite_id
                     AND s.judge_slug  = ?
                     AND s.criterion   = ?
                    JOIN attack_nli_scores n
                      ON n.rewrite_id = r.rewrite_id
                    WHERE r.method = 'bon_candidate'
                      AND r.rewriter_model = ?
                      AND r.criterion = ?
                """, (judge_slug, cri, rewriter_id, cri)).fetchall()

                best = {}
                for src, cand_rid, text, wc, js, fwd, bwd in rows:
                    nli_score = 100.0 * (fwd + bwd) / 2.0
                    combined = 0.5 * js + 0.5 * nli_score
                    if src not in best or combined > best[src][6]:
                        best[src] = (cand_rid, text, wc, js, nli_score, fwd, combined, bwd)

                if not best:
                    print(f"[{short} × {cri} × fold{fold}] no candidates — skipping", flush=True)
                    continue

                config_template = {
                    "method": "bon_panel_single_nli",
                    "fold": fold,
                    "criterion": cri,
                    "rewriter": rewriter_id,
                    "in_panel_single_judge": judge_slug,
                    "selection": "argmax(0.5*single_judge_score + 0.5*100*(nli_fwd+nli_bwd)/2)",
                }
                config_json = json.dumps(config_template, separators=(",", ":"))
                judge_panel = json.dumps([judge_slug], separators=(",", ":"))

                inserted = 0
                for src, (cand_rid, text, wc, js, nli_score, fwd, combined, bwd) in best.items():
                    rid = f"bon_snli_{short}_f{fold}_{cri}_{src}"
                    meta = json.dumps({
                        "selected_candidate_rid": cand_rid,
                        "selected_judge_score": js,
                        "selected_nli_score": nli_score,
                        "selected_combined": combined,
                        "selected_nli_fwd": fwd,
                        "selected_nli_bwd": bwd,
                    })
                    cur.execute("""
                        INSERT OR IGNORE INTO attack_rewrites
                        (rewrite_id, source_doc_id, method, fold, criterion,
                         config_json, rewriter_model, judge_panel_json,
                         text, word_count, run_metadata_json)
                        VALUES (?, ?, 'bon_panel_single_nli', ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rid, src, fold, cri, config_json, rewriter_id, judge_panel,
                          text, wc, meta))
                    inserted += cur.rowcount
                if not args.dry_run:
                    conn.commit()
                total_inserted += inserted
                m_combined = sum(b[6] for b in best.values()) / len(best)
                m_j = sum(b[3] for b in best.values()) / len(best)
                m_nli = sum(b[4] for b in best.values()) / len(best)
                print(f"[{short} × {cri} × fold{fold}] judge={judge_slug.replace('judge_',''):<8} "
                      f"inserted={inserted}  mean(combined)={m_combined:.2f}  mean(judge)={m_j:.2f}  mean(nli×100)={m_nli:.2f}",
                      flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] total inserted: {total_inserted} "
          f"({'DRY RUN' if args.dry_run else 'committed'})", flush=True)


if __name__ == "__main__":
    main()
