"""Copy judge / NLI / agreement scores from prior BoN winners to a new BoN
selection method's winners — when both methods picked the same underlying
candidate.

Why
---
Each BoN selection script writes new attack_rewrites rows whose `text` is
copied from a chosen `bon_candidate` row. The original candidate's id is
recorded in `run_metadata_json.selected_candidate_rid`. Local OOS judges and
OpenRouter judges only score the WINNER rows (bon_candidate is excluded by
default), so winners' scores are keyed by the winner's rewrite_id, not the
candidate's. If a NEW BoN rule picks a candidate that some prior rule
already picked, the prior rule's winner already has full judge / NLI /
agreement coverage — we can copy scores instead of re-scoring.

This script is purely SQL — no API calls, no GPU. Idempotent (INSERT OR
IGNORE). Verifies texts match before copying so we never silently mis-attribute
scores when metadata is wrong.

Usage
-----
    # After your new selection script has populated attack_rewrites with
    # method='bon_my_new_rule_xyz' rows, run:
    python3 scripts/bon_reuse_scores.py --new-method bon_my_new_rule_xyz

    # Optional --source-methods to restrict which prior methods to copy from
    # (default: all 4 existing BoN methods).
    # Optional --dry-run to print plan without writing.

Expectation
-----------
A new winner row gets full coverage if and only if a prior winner on the same
(rewriter_model, criterion, fold, source_doc_id) picked the same candidate
(same selected_candidate_rid). Otherwise the new winner is left without
copied scores and the caller should run their normal scoring pass to fill
the gap.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
DEFAULT_SOURCE_METHODS = (
    "bon_panel", "bon_panel_single", "bon_panel_nli", "bon_panel_single_nli",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-method", required=True,
                    help="method tag of the new BoN selection (rows already inserted into attack_rewrites)")
    ap.add_argument("--source-methods", nargs="+", default=list(DEFAULT_SOURCE_METHODS),
                    help="prior BoN methods to copy from (default: all 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be copied without writing")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    n_new = cur.execute(
        "SELECT COUNT(*) FROM attack_rewrites WHERE method = ?",
        (args.new_method,)
    ).fetchone()[0]
    if n_new == 0:
        print(f"[{time.strftime('%H:%M:%S')}] no rows with method='{args.new_method}' — nothing to do",
              flush=True)
        sys.exit(0)
    print(f"[{time.strftime('%H:%M:%S')}] new method '{args.new_method}': {n_new} winner rows", flush=True)

    placeholders_src = ",".join("?" * len(args.source_methods))

    # Match new winners to prior winners on the (rewriter_model, criterion, fold,
    # source_doc_id, selected_candidate_rid) tuple. We pick at most one prior
    # winner per new winner (deterministic — first by source method order,
    # then by rewrite_id) and verify the text matches before copying.
    matches = cur.execute(f"""
        WITH new_w AS (
            SELECT
                rewrite_id      AS new_rid,
                source_doc_id, fold, criterion, rewriter_model, text,
                json_extract(run_metadata_json, '$.selected_candidate_rid') AS cand_rid
            FROM attack_rewrites
            WHERE method = ?
        ),
        prior_w AS (
            SELECT
                rewrite_id      AS prior_rid,
                method          AS prior_method,
                source_doc_id, fold, criterion, rewriter_model, text,
                json_extract(run_metadata_json, '$.selected_candidate_rid') AS cand_rid
            FROM attack_rewrites
            WHERE method IN ({placeholders_src})
              AND json_extract(run_metadata_json, '$.selected_candidate_rid') IS NOT NULL
        )
        SELECT
            n.new_rid, n.cand_rid, p.prior_rid, p.prior_method,
            (n.text = p.text)         AS text_match,
            n.source_doc_id, n.fold, n.criterion, n.rewriter_model
        FROM new_w n
        JOIN prior_w p
          ON  p.source_doc_id   = n.source_doc_id
          AND p.fold            IS n.fold       -- IS-equal handles NULL=NULL
          AND p.criterion       = n.criterion
          AND p.rewriter_model  = n.rewriter_model
          AND p.cand_rid        = n.cand_rid
        ORDER BY n.new_rid, p.prior_method, p.prior_rid
    """, (args.new_method, *args.source_methods)).fetchall()

    # Collapse to one prior per new (first match)
    seen = set()
    pairs = []
    text_mismatches = 0
    for new_rid, cand_rid, prior_rid, prior_method, text_match, *rest in matches:
        if new_rid in seen:
            continue
        seen.add(new_rid)
        if not text_match:
            text_mismatches += 1
            print(f"  [TEXT MISMATCH] new={new_rid} prior={prior_rid} (method={prior_method}) "
                  f"— skipping; would have copied wrong scores", flush=True)
            continue
        pairs.append((new_rid, prior_rid, prior_method))

    n_matched = len(pairs)
    n_unmatched = n_new - n_matched - text_mismatches
    print(f"  matched: {n_matched}/{n_new}  ({100*n_matched/n_new:.1f}%)", flush=True)
    print(f"  text mismatches (skipped): {text_mismatches}", flush=True)
    print(f"  unmatched (will need fresh scoring): {n_unmatched}", flush=True)

    if not pairs:
        print("  nothing to copy", flush=True)
        return

    by_source_method = {}
    for _, _, m in pairs:
        by_source_method[m] = by_source_method.get(m, 0) + 1
    print("  source-method distribution:", flush=True)
    for m, n in sorted(by_source_method.items(), key=lambda x: -x[1]):
        print(f"    {m:<25}  {n}", flush=True)

    if args.dry_run:
        # Estimate row counts that WOULD be copied.
        prior_rids = [p for _, p, _ in pairs]
        placeholders = ",".join("?" * len(prior_rids))
        n_judge = cur.execute(
            f"SELECT COUNT(*) FROM attack_judge_scores WHERE rewrite_id IN ({placeholders})",
            prior_rids
        ).fetchone()[0]
        n_nli = cur.execute(
            f"SELECT COUNT(*) FROM attack_nli_scores WHERE rewrite_id IN ({placeholders})",
            prior_rids
        ).fetchone()[0]
        n_agr = cur.execute(
            f"SELECT COUNT(*) FROM attack_agreement_scores WHERE rewrite_id IN ({placeholders})",
            prior_rids
        ).fetchone()[0]
        print(f"\n  [DRY RUN] would copy:", flush=True)
        print(f"    attack_judge_scores rows:     {n_judge}", flush=True)
        print(f"    attack_nli_scores rows:       {n_nli}", flush=True)
        print(f"    attack_agreement_scores rows: {n_agr}", flush=True)
        return

    # Real copy. INSERT OR IGNORE so re-runs are safe and the caller can also
    # score the row directly if desired (existing rows are not overwritten).
    n_judge_inserted = 0
    n_nli_inserted = 0
    n_agr_inserted = 0
    for new_rid, prior_rid, _prior_method in pairs:
        # judge scores (per-criterion key already includes the rewrite_id, so
        # multiple judges/criteria per (new_rid, prior_rid) all carry over)
        before = cur.execute(
            "SELECT COUNT(*) FROM attack_judge_scores WHERE rewrite_id=?", (new_rid,)
        ).fetchone()[0]
        cur.execute("""
            INSERT OR IGNORE INTO attack_judge_scores
                (rewrite_id, judge_slug, criterion, score, reasoning)
            SELECT ?, judge_slug, criterion, score, reasoning
            FROM attack_judge_scores WHERE rewrite_id = ?
        """, (new_rid, prior_rid))
        after = cur.execute(
            "SELECT COUNT(*) FROM attack_judge_scores WHERE rewrite_id=?", (new_rid,)
        ).fetchone()[0]
        n_judge_inserted += (after - before)

        # nli scores (rewrite_id is PK so at most 1 row)
        cur.execute("""
            INSERT OR IGNORE INTO attack_nli_scores
                (rewrite_id, nli_fwd, nli_bwd, model_id)
            SELECT ?, nli_fwd, nli_bwd, model_id
            FROM attack_nli_scores WHERE rewrite_id = ?
        """, (new_rid, prior_rid))
        n_nli_inserted += cur.rowcount

        # agreement scores (rewrite_id is PK)
        cur.execute("""
            INSERT OR IGNORE INTO attack_agreement_scores
                (rewrite_id, score)
            SELECT ?, score
            FROM attack_agreement_scores WHERE rewrite_id = ?
        """, (new_rid, prior_rid))
        n_agr_inserted += cur.rowcount

    conn.commit()
    print(f"\n[{time.strftime('%H:%M:%S')}] copied:", flush=True)
    print(f"  attack_judge_scores rows inserted:     {n_judge_inserted}", flush=True)
    print(f"  attack_nli_scores rows inserted:       {n_nli_inserted}", flush=True)
    print(f"  attack_agreement_scores rows inserted: {n_agr_inserted}", flush=True)
    print(f"\n  remaining unmatched winners ({n_unmatched}) need fresh scoring.", flush=True)


if __name__ == "__main__":
    main()
