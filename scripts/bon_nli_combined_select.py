"""BoN selection variant: pick the candidate that maximises a combined score
of (in-panel judge mean) and (NLI bidirectional mean), equally weighted.

For each (rewriter × criterion × fold), each candidate's combined score is:

    combined = 0.5 * ej_in_panel + 0.5 * 100 * (nli_fwd + nli_bwd) / 2

where ej_in_panel = mean of the 2 in-panel judges' scores for that fold (per
the standard rotation: fold1=q+l, fold2=q+g, fold3=l+g). Both terms are on a
0-100 scale, so the equal weighting is also equal in magnitude.

Pure SQL — reuses the cached candidate scores and the (already 100%-complete)
NLI scores on bon_candidate rows. Idempotent.

Method tag: 'bon_panel_nli'. rewrite_id prefix: 'bon_nli_*' (collision-checked).
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

# fold → in-panel pair (the 2 judges whose scores feed the ej component)
FOLD_TO_IN_PANEL = {
    1: ("judge_qwen95b", "judge_llama8b"),
    2: ("judge_qwen95b", "judge_gemma9b"),
    3: ("judge_llama8b", "judge_gemma9b"),
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
            for fold, in_panel in FOLD_TO_IN_PANEL.items():
                j1, j2 = in_panel
                rows = cur.execute("""
                    SELECT
                      r.source_doc_id,
                      r.rewrite_id    AS cand_rid,
                      r.text          AS cand_text,
                      r.word_count    AS cand_wc,
                      s1.score        AS s1,
                      s2.score        AS s2,
                      n.nli_fwd       AS nli_fwd,
                      n.nli_bwd       AS nli_bwd
                    FROM attack_rewrites r
                    JOIN attack_judge_scores s1
                      ON s1.rewrite_id = r.rewrite_id
                     AND s1.judge_slug  = ?
                     AND s1.criterion   = ?
                    JOIN attack_judge_scores s2
                      ON s2.rewrite_id = r.rewrite_id
                     AND s2.judge_slug  = ?
                     AND s2.criterion   = ?
                    JOIN attack_nli_scores n
                      ON n.rewrite_id = r.rewrite_id
                    WHERE r.method = 'bon_candidate'
                      AND r.rewriter_model = ?
                      AND r.criterion = ?
                """, (j1, cri, j2, cri, rewriter_id, cri)).fetchall()

                # Group by source_doc_id, take argmax of combined score
                best = {}
                for src, cand_rid, text, wc, s1, s2, fwd, bwd in rows:
                    ej = (s1 + s2) / 2.0
                    nli_score = 100.0 * (fwd + bwd) / 2.0
                    combined = 0.5 * ej + 0.5 * nli_score
                    if src not in best or combined > best[src][6]:
                        best[src] = (cand_rid, text, wc, ej, nli_score, s1, combined, fwd, bwd, s2)

                if not best:
                    print(f"[{short} × {cri} × fold{fold}] no candidates — skipping", flush=True)
                    continue

                config_template = {
                    "method": "bon_panel_nli",
                    "fold": fold,
                    "criterion": cri,
                    "rewriter": rewriter_id,
                    "in_panel": list(in_panel),
                    "selection": "argmax(0.5*mean(in_panel)+0.5*100*(nli_fwd+nli_bwd)/2)",
                }
                config_json = json.dumps(config_template, separators=(",", ":"))
                judge_panel = json.dumps(list(in_panel), separators=(",", ":"))

                inserted = 0
                for src, (cand_rid, text, wc, ej, nli_score, s1, combined, fwd, bwd, s2) in best.items():
                    rid = f"bon_nli_{short}_f{fold}_{cri}_{src}"
                    meta = json.dumps({
                        "selected_candidate_rid": cand_rid,
                        "selected_ej": ej,
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
                        VALUES (?, ?, 'bon_panel_nli', ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (rid, src, fold, cri, config_json, rewriter_id, judge_panel,
                          text, wc, meta))
                    inserted += cur.rowcount
                if not args.dry_run:
                    conn.commit()
                total_inserted += inserted
                mean_combined = sum(b[6] for b in best.values()) / len(best)
                mean_ej = sum(b[3] for b in best.values()) / len(best)
                mean_nli = sum(b[4] for b in best.values()) / len(best)
                print(f"[{short} × {cri} × fold{fold}] in-panel={j1.replace('judge_',''):<8}+{j2.replace('judge_',''):<8} "
                      f"inserted={inserted}  mean(combined)={mean_combined:.2f}  mean(ej)={mean_ej:.2f}  mean(nli×100)={mean_nli:.2f}",
                      flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] total inserted: {total_inserted} "
          f"({'DRY RUN' if args.dry_run else 'committed'})", flush=True)


if __name__ == "__main__":
    main()
