"""Insert paragraphs.origin_kind='original_model' rows into attack_rewrites
with method='original_ai', mirroring the existing method='original' shape used
for human-written originals.

Filtered to the 40 controversial propositions used in the mission. Idempotent
(INSERT OR IGNORE on a deterministic rewrite_id derived from document_id).

After insertion, score with the 6-judge panel via score_all_missing.py
--methods original_ai (or no filter — score_all_missing's default already
includes everything except bon_candidate).

rewrite_id format: 'origai_{document_id}'  (distinct from any other prefix —
collision-checked).
"""
import argparse
import json
import sqlite3

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
DATASET = "/workspace/grpo_run/controversial_40_3fold.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 40 controversial proposition_ids
    ds = json.load(open(DATASET))
    prop_ids = {p["pid"] for p in ds["propositions"]}
    print(f"target propositions: {len(prop_ids)}")

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Pull all original_model paragraphs for those 40 props
    rows = cur.execute("""
        SELECT document_id, proposition_id, paul_data_model_name, text, word_count
        FROM paragraphs
        WHERE origin_kind = 'original_model'
        AND proposition_id IN ({})
    """.format(",".join("?" for _ in prop_ids)), list(prop_ids)).fetchall()
    print(f"original_model rows for those props: {len(rows)}")

    # Group by paul_data_model_name to see the rewriter inventory
    from collections import Counter
    by_model = Counter(r[2] for r in rows)
    print(f"\nrewriter (paul_data_model_name) breakdown:")
    for m, n in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {m or 'NULL':<35} n={n}")

    inserted = 0
    skipped_existing = 0
    config = {"method": "original_ai",
              "source": "paragraphs.origin_kind='original_model'",
              "note": "AI-authored paragraphs from paul_data; complement of method='original' for humans"}
    config_json = json.dumps(config, separators=(",", ":"))

    for doc_id, _prop, model_name, text, wc in rows:
        rid = f"origai_{doc_id}"
        cur.execute("""
            INSERT OR IGNORE INTO attack_rewrites
            (rewrite_id, source_doc_id, method, fold, criterion, config_json,
             rewriter_model, judge_panel_json, text, word_count, run_metadata_json)
            VALUES (?, ?, 'original_ai', NULL, 'clarity', ?, ?, NULL, ?, ?, NULL)
        """, (rid, doc_id, config_json, model_name, text, wc))
        if cur.rowcount:
            inserted += 1
        else:
            skipped_existing += 1
    if not args.dry_run:
        conn.commit()
    print(f"\ninserted: {inserted}, skipped (already-existing): {skipped_existing}")
    n_total = cur.execute("SELECT COUNT(*) FROM attack_rewrites WHERE method='original_ai'").fetchone()[0]
    print(f"total method='original_ai' rows now: {n_total}")


if __name__ == "__main__":
    main()
