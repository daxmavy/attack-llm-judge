"""Backfill GRPO rewrites into attack_rewrites + held-out judge scores.

Usage: python3 backfill_grpo_rewriter.py --short lfm25-12b --fold 1 --criterion clarity \
       --rewriter LiquidAI/LFM2.5-1.2B-Instruct --held-out gemma9b

Idempotent (INSERT OR IGNORE).
"""
import argparse
import json
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True, help="rewriter short name, e.g. lfm25-12b")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--criterion", required=True, choices=["clarity", "informativeness"])
    ap.add_argument("--rewriter", required=True)
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--method-tag", default="grpo_400step",
                    help="method tag for attack_rewrites (e.g. grpo_400step or grpo_nli_400step)")
    ap.add_argument("--name-prefix", default="grpo",
                    help="rewrite_id prefix + pilot dir prefix (e.g. 'grpo' or 'grpo_nli')")
    ap.add_argument("--db", default="/home/max/attack-llm-judge/data/paragraphs.db")
    args = ap.parse_args()

    eval_path = f"/workspace/grpo_run/pilot_{args.name_prefix}_{args.short}_fold{args.fold}_{args.criterion}/eval_summary.json"
    p = json.load(open(eval_path))
    doc_ids = p["eval_document_ids"]
    post = p["post_rewrites"]
    heldout_scores = p.get("heldout_post_scores", [])
    assert len(doc_ids) == len(post) == 714, f"lengths: {len(doc_ids)}/{len(post)}"

    config = {
        "method": args.method_tag,
        "fold": args.fold,
        "criterion": args.criterion,
        "rewriter": args.rewriter,
        "held_out": args.held_out,
        "summary": p["summary"],
    }
    config_json = json.dumps(config, separators=(",", ":"))
    judge_panel = json.dumps([f"judge_{args.held_out}"], separators=(",", ":"))

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    inserted = 0
    for doc_id, text in zip(doc_ids, post):
        rid = f"{args.name_prefix}_{args.short}_f{args.fold}_{args.criterion}_{doc_id}"
        wc = len(text.split())
        cur.execute(
            """INSERT OR IGNORE INTO attack_rewrites
               (rewrite_id, source_doc_id, method, fold, criterion, config_json, rewriter_model,
                judge_panel_json, text, word_count, run_metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (rid, doc_id, args.method_tag, args.fold, args.criterion, config_json, args.rewriter, judge_panel,
             text, wc),
        )
        inserted += cur.rowcount

    score_inserted = 0
    if heldout_scores and len(heldout_scores) == len(doc_ids):
        for doc_id, sc in zip(doc_ids, heldout_scores):
            rid = f"{args.name_prefix}_{args.short}_f{args.fold}_{args.criterion}_{doc_id}"
            cur.execute(
                """INSERT OR IGNORE INTO attack_judge_scores
                   (rewrite_id, judge_slug, criterion, score, reasoning)
                   VALUES (?, ?, ?, ?, NULL)""",
                (rid, f"judge_{args.held_out}", args.criterion, float(sc)),
            )
            score_inserted += cur.rowcount
    conn.commit()
    conn.close()
    print(f"backfilled {inserted} rewrites, {score_inserted} held-out scores for "
          f"{args.name_prefix} {args.short} fold {args.fold} {args.criterion} "
          f"(method_tag={args.method_tag})")


if __name__ == "__main__":
    main()
