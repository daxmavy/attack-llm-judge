"""Comprehensive gap-filling scorer.

For each judge (3) × criterion (2), finds rows in attack_rewrites that don't yet
have a score in attack_judge_scores, and scores them via vLLM. Idempotent — resuming
skips completed cells.

Scope: every attack_rewrites row EXCEPT bon_candidate (which has 11424 rows per criterion
and is not itself part of the eval panel — only bon_panel winners are). bon_candidate
clarity is already fully scored anyway; bon_candidate informativeness isn't needed for
eval comparison (we only report panel-level numbers).

Usage:
  python3 score_all_missing.py                      # all judges, both criteria
  python3 score_all_missing.py --judges qwen95b     # single judge
  python3 score_all_missing.py --criteria clarity   # single criterion
  python3 score_all_missing.py --include-candidates # also score bon_candidate

vLLM engines loaded one at a time to avoid OOM. Each engine scores both criteria
before unloading.
"""
import argparse
import gc
import sqlite3
import sys
import time

sys.path.insert(0, "/workspace/grpo_run")

DB = "/home/max/attack-llm-judge/data/paragraphs.db"
JUDGE_KEYS = ["qwen95b", "llama8b", "gemma9b"]
CRITERIA = ["clarity", "informativeness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=JUDGE_KEYS)
    ap.add_argument("--criteria", nargs="+", default=CRITERIA)
    ap.add_argument("--include-candidates", action="store_true",
                    help="also score bon_candidate rows (default: skip)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to specific methods (default: all except bon_candidate)")
    args = ap.parse_args()

    import torch
    from run_pilot_len_pen import JUDGE_REGISTRY, JudgeVLLM, RUBRICS

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Build the universe of rewrites to consider
    exclude_cand = "AND r.method != 'bon_candidate'" if not args.include_candidates else ""
    method_filter = ""
    params = []
    if args.methods:
        method_filter = f"AND r.method IN ({','.join('?' for _ in args.methods)})"
        params = list(args.methods)
    all_rows = cur.execute(f"""
        SELECT r.rewrite_id, p.proposition, r.text
        FROM attack_rewrites r
        JOIN paragraphs p ON r.source_doc_id = p.document_id
        WHERE 1=1 {exclude_cand} {method_filter}
    """, params).fetchall()
    print(f"[{time.strftime('%H:%M:%S')}] universe size: {len(all_rows)} rewrites", flush=True)

    for judge_key in args.judges:
        spec = JUDGE_REGISTRY[judge_key]  # (slug, model_id)
        slug = spec[0]

        # Determine todo per criterion BEFORE loading the judge
        cri_todo = {}
        for cri in args.criteria:
            scored = {r[0] for r in cur.execute(
                "SELECT rewrite_id FROM attack_judge_scores WHERE judge_slug=? AND criterion=?",
                (slug, cri)).fetchall()}
            todo = [r for r in all_rows if r[0] not in scored]
            cri_todo[cri] = todo
            print(f"  {slug} × {cri}: {len(todo)}/{len(all_rows)} remaining", flush=True)

        if not any(cri_todo.values()):
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: nothing to do", flush=True)
            continue

        # Load judge ONCE, swap rubric in-place across criteria (avoids vLLM reload)
        first_cri = next((c for c, t in cri_todo.items() if t), None)
        if first_cri is None:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] loading {slug} (initial rubric={first_cri})",
              flush=True)
        j = JudgeVLLM(*spec, rubric=first_cri)

        for cri, todo in cri_todo.items():
            if not todo:
                continue
            j.rubric_name = cri
            j.rubric_text = RUBRICS[cri]
            print(f"[{time.strftime('%H:%M:%S')}] {slug} @ {cri} — {len(todo)} items", flush=True)
            props = [r[1] for r in todo]
            texts = [r[2] for r in todo]
            t0 = time.time()
            scores = j.score(props, texts)
            dt = (time.time() - t0) / 60
            rate = len(scores) / dt / 60 if dt > 0 else 0
            print(f"  scored {len(scores)} in {dt:.1f} min ({rate:.1f}/s)", flush=True)
            for (rid, _, _), sc in zip(todo, scores):
                cur.execute("""
                    INSERT OR REPLACE INTO attack_judge_scores
                    (rewrite_id, judge_slug, criterion, score, reasoning)
                    VALUES (?, ?, ?, ?, NULL)
                """, (rid, slug, cri, sc))
            conn.commit()

        del j
        gc.collect()
        torch.cuda.empty_cache()

    conn.close()
    print(f"[{time.strftime('%H:%M:%S')}] done", flush=True)


if __name__ == "__main__":
    main()
