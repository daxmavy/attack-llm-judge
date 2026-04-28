"""BoN selection variant: pick the candidate with the highest bidirectional
NLI sum (nli_fwd + nli_bwd) per (rewriter, criterion, source_doc).

Uses pre-computed candidate-level NLI scores in attack_nli_scores (100% covered
on bon_candidate). Pure SQL — no API calls, no GPU.

For analysis-pipeline consistency we replicate the same fold-independent
winner across folds 1, 2, 3 (the rule does not depend on fold; replicating
keeps row-level joins with existing fold-aware methods straightforward).

Method tag: 'bon_panel_highnli'. rewrite_id prefix: 'bon_hnli_'.
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
            rows = cur.execute("""
                SELECT
                    r.source_doc_id,
                    r.rewrite_id   AS cand_rid,
                    r.text         AS cand_text,
                    r.word_count   AS cand_wc,
                    n.nli_fwd, n.nli_bwd
                FROM attack_rewrites r
                JOIN attack_nli_scores n ON n.rewrite_id = r.rewrite_id
                WHERE r.method = 'bon_candidate'
                  AND r.rewriter_model = ?
                  AND r.criterion = ?
            """, (rewriter_id, cri)).fetchall()

            best = {}
            for src, cand_rid, text, wc, fwd, bwd in rows:
                nli_sum = fwd + bwd
                if src not in best or nli_sum > best[src][4]:
                    best[src] = (cand_rid, text, wc, fwd, nli_sum, bwd)

            if not best:
                print(f"[{short} × {cri}] no candidates — skipping", flush=True)
                continue

            inserted = 0
            for fold in FOLDS:
                config = {
                    "method": "bon_panel_highnli",
                    "fold": fold,
                    "criterion": cri,
                    "rewriter": rewriter_id,
                    "selection": "argmax(nli_fwd + nli_bwd)",
                    "note": "fold-independent rule; row replicated per fold for analysis consistency",
                }
                config_json = json.dumps(config, separators=(",", ":"))
                # judge_panel_json left empty — this rule doesn't use judges
                judge_panel = json.dumps([], separators=(",", ":"))

                for src, (cand_rid, text, wc, fwd, nli_sum, bwd) in best.items():
                    rid = f"bon_hnli_{short}_f{fold}_{cri}_{src}"
                    meta = json.dumps({
                        "selected_candidate_rid": cand_rid,
                        "selected_nli_fwd": fwd,
                        "selected_nli_bwd": bwd,
                        "selected_nli_sum": nli_sum,
                    })
                    cur.execute("""
                        INSERT OR IGNORE INTO attack_rewrites
                        (rewrite_id, source_doc_id, method, fold, criterion,
                         config_json, rewriter_model, judge_panel_json,
                         text, word_count, run_metadata_json)
                        VALUES (?, ?, 'bon_panel_highnli', ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rid, src, fold, cri, config_json, rewriter_id, judge_panel,
                          text, wc, meta))
                    inserted += cur.rowcount

            if not args.dry_run:
                conn.commit()
            total_inserted += inserted
            mean_nli = sum(b[4] for b in best.values()) / len(best)
            print(f"[{short} × {cri}] inserted={inserted} (3 folds × {len(best)} sources)  "
                  f"mean(nli_sum)={mean_nli:.3f}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] total inserted: {total_inserted} "
          f"({'DRY RUN' if args.dry_run else 'committed'})", flush=True)


if __name__ == "__main__":
    main()
