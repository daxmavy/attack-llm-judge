"""Backfill GRPO post_rewrites into attack_rewrites + per-sample scores into attack_judge_scores.

Source: /workspace/grpo_run/pilot_grpo_400step_fold{N}/eval_summary.json
  - post_rewrites[714]: the GRPO-trained rewriter's outputs on the eval set
  - eval_document_ids[714]: aligned doc_ids
  - heldout_pre_scores / heldout_post_scores[714]: held-out judge per-sample scores
  - fold 1 only: gemma9b_pre_scores_posthoc / gemma9b_post_scores_posthoc[714]
  - summary[judge_X].{pre_mean, post_mean, delta, role}

Note: in-panel judges have only means in the JSON, not per-sample arrays. This script
inserts only the rewrite rows + per-sample scores we have; a follow-up score pass is
needed to fill in in-panel judge scores (and informativeness scores) uniformly.

rewrite_id scheme: grpo_f{fold}_{criterion}_{doc_id} (stable, idempotent)
method: 'grpo_400step'
"""
import json
import sqlite3
import time

import sys
DB = "/home/max/attack-llm-judge/data/paragraphs.db"
CRITERION = sys.argv[1] if len(sys.argv) > 1 else "clarity"
# Accept per-fold list via argv[2:], default to all 3
FOLDS = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [1, 2, 3]

HELDOUT_BY_FOLD = {1: "judge_gemma9b", 2: "judge_llama8b", 3: "judge_qwen95b"}
# Fold 1 was originally run with qwen7b as held-out; heldout_*_scores in fold1 JSON is qwen7b.
# The correct held-out for the mission is gemma9b, which was filled in posthoc arrays.
FOLD1_HELDOUT_JSON_FIELD = "judge_qwen7b"

PANEL = [
    "judge_qwen95b",
    "judge_llama8b",
    "judge_gemma9b",
]


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    total_rewrites = 0
    total_scores = 0

    for fold in FOLDS:
        suffix = "" if CRITERION == "clarity" else "_informativeness"
        path = f"/workspace/grpo_run/pilot_grpo_400step_fold{fold}{suffix}/eval_summary.json"
        print(f"[{time.strftime('%H:%M:%S')}] loading fold {fold}: {path}")
        data = json.load(open(path))
        doc_ids = data["eval_document_ids"]
        post_rewrites = data["post_rewrites"]
        assert len(doc_ids) == len(post_rewrites) == 714, f"fold {fold} length mismatch"

        # Build config_json describing the GRPO run
        summary = data["summary"]
        panel_info = {j: {"role": s.get("role"), "post_mean": s["post_mean"], "pre_mean": s["pre_mean"]}
                      for j, s in summary.items()}
        config = {
            "method": "grpo_400step",
            "fold": fold,
            "criterion": CRITERION,
            "held_out_judge": HELDOUT_BY_FOLD[fold],
            "max_steps": 400,
            "alpha": 100,
            "lr": 5e-6,
            "penalty_shape": "asymm_cubic",
            "penalty_gamma": 1000,
            "penalty_over_tol": 0.15,
            "embed_sim": True,
            "embed_beta": 200,
            "embed_threshold": 0.85,
            "summary": panel_info,
        }
        config_json = json.dumps(config, separators=(",", ":"))
        judge_panel_json = json.dumps(PANEL, separators=(",", ":"))
        rewriter_model = "Qwen/Qwen2.5-1.5B-Instruct"

        # Insert rewrite rows
        inserted = 0
        for doc_id, text in zip(doc_ids, post_rewrites):
            rw_id = f"grpo_f{fold}_{CRITERION}_{doc_id}"
            wc = len(text.split())
            cur.execute(
                """INSERT OR IGNORE INTO attack_rewrites
                   (rewrite_id, source_doc_id, method, fold, config_json, rewriter_model,
                    judge_panel_json, text, word_count, run_metadata_json, criterion)
                   VALUES (?, ?, 'grpo_400step', ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (rw_id, doc_id, fold, config_json, rewriter_model, judge_panel_json,
                 text, wc, CRITERION),
            )
            inserted += cur.rowcount
        conn.commit()
        print(f"  fold {fold}: {inserted}/714 new rewrite rows inserted")
        total_rewrites += inserted

        # Insert per-sample scores we have
        score_arrays = []
        if CRITERION == "clarity" and fold == 1:
            # Legacy clarity fold 1: held-out was auto-picked qwen7b (wrong); gemma9b_posthoc arrays
            # carry the mission-correct held-out scores. Keep qwen7b for provenance.
            score_arrays.append(("judge_qwen7b", data["heldout_post_scores"]))
            if "gemma9b_post_scores_posthoc" in data:
                score_arrays.append(("judge_gemma9b", data["gemma9b_post_scores_posthoc"]))
        else:
            # Correct held-out mapping used by every other fold/criterion combo
            score_arrays.append((HELDOUT_BY_FOLD[fold], data["heldout_post_scores"]))

        for judge_slug, scores in score_arrays:
            assert len(scores) == 714
            score_inserted = 0
            for doc_id, score in zip(doc_ids, scores):
                rw_id = f"grpo_f{fold}_{CRITERION}_{doc_id}"
                cur.execute(
                    """INSERT OR IGNORE INTO attack_judge_scores
                       (rewrite_id, judge_slug, criterion, score, reasoning)
                       VALUES (?, ?, ?, ?, NULL)""",
                    (rw_id, judge_slug, CRITERION, float(score)),
                )
                score_inserted += cur.rowcount
            conn.commit()
            print(f"  fold {fold}: {score_inserted}/714 {judge_slug} scores inserted")
            total_scores += score_inserted

    print()
    print(f"=== totals ===")
    print(f"  rewrites inserted: {total_rewrites}")
    print(f"  scores inserted: {total_scores}")

    # Verification
    print()
    print("=== verification ===")
    for row in cur.execute(
        "SELECT method, fold, criterion, COUNT(*) FROM attack_rewrites "
        "WHERE method='grpo_400step' GROUP BY method, fold, criterion ORDER BY fold"
    ):
        print(" ", row)
    print()
    for row in cur.execute(
        "SELECT r.fold, s.judge_slug, s.criterion, COUNT(*), ROUND(AVG(s.score),2) "
        "FROM attack_judge_scores s JOIN attack_rewrites r ON s.rewrite_id = r.rewrite_id "
        "WHERE r.method='grpo_400step' "
        "GROUP BY r.fold, s.judge_slug, s.criterion ORDER BY r.fold, s.judge_slug"
    ):
        print(" ", row)

    conn.close()


if __name__ == "__main__":
    main()
