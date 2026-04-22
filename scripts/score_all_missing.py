"""Comprehensive gap-filling scorer.

For each judge (3) × criterion (2), finds rows in attack_rewrites that don't yet
have a score in attack_judge_scores, and scores them via the HTTP judge server.
Idempotent — resuming skips completed cells.

Scope: every attack_rewrites row EXCEPT bon_candidate (which has 11424 rows per criterion
and is not itself part of the eval panel — only bon_panel winners are). bon_candidate
clarity is already fully scored anyway; bon_candidate informativeness isn't needed for
eval comparison (we only report panel-level numbers).

Usage:
  python3 score_all_missing.py                      # all judges, both criteria
  python3 score_all_missing.py --judges judgeA      # single judge
  python3 score_all_missing.py --criteria clarity   # single criterion
  python3 score_all_missing.py --include-candidates # also score bon_candidate

Judges are loaded onto the shared HTTP judge server one at a time and unloaded
between slugs, so GPU 0 only holds one engine at any moment. Rubrics are
swapped in place (no reload) between criteria for the same judge.
"""
import argparse
import gc
import sqlite3
import sys
import time

sys.path.insert(0, "/data/shil6647/attack-llm-judge/grpo_run")
sys.path.insert(0, "/home/shil6647/attack-llm-judge")

from config.models import JUDGE_REGISTRY, require_config
from judge.http_client import JudgeHTTP, spawn_judge_server

DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
# Default to scoring with every judge in the canonical registry.
JUDGE_KEYS = list(JUDGE_REGISTRY.keys())
CRITERIA = ["clarity", "informativeness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=JUDGE_KEYS)
    ap.add_argument("--criteria", nargs="+", default=CRITERIA)
    ap.add_argument("--include-candidates", action="store_true",
                    help="also score bon_candidate rows (default: skip)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to specific methods (default: all except bon_candidate)")
    ap.add_argument("--judge-endpoint", type=str, default=None,
                    help="Reuse an already-running judge server.")
    ap.add_argument("--judge-port", type=int, default=8127)
    ap.add_argument("--judge-gpu", type=int, default=0,
                    help="Physical GPU for the spawned judge server.")
    args = ap.parse_args()
    require_config()

    # Judge server — spawn or reuse.
    if args.judge_endpoint:
        judge_endpoint = args.judge_endpoint
        judge_server_proc = None
        print(f"[{time.strftime('%H:%M:%S')}] reusing judge server at {judge_endpoint}",
              flush=True)
    else:
        log_path = "/data/shil6647/attack-llm-judge/grpo_run/score_all_judge_server.log"
        judge_server_proc, judge_endpoint = spawn_judge_server(
            port=args.judge_port, gpu=args.judge_gpu, log_path=log_path,
        )

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
        wandb_name, _hf_id = JUDGE_REGISTRY[judge_key]
        # judge_slug column historically stores the wandb_name (e.g.
        # "judge_<slug>"), preserved here so existing aggregation queries keep
        # working across the rewriter/refactor boundary.
        slug_column = wandb_name

        # Determine todo per criterion BEFORE loading the judge
        cri_todo = {}
        for cri in args.criteria:
            scored = {r[0] for r in cur.execute(
                "SELECT rewrite_id FROM attack_judge_scores WHERE judge_slug=? AND criterion=?",
                (slug_column, cri)).fetchall()}
            todo = [r for r in all_rows if r[0] not in scored]
            cri_todo[cri] = todo
            print(f"  {slug_column} × {cri}: {len(todo)}/{len(all_rows)} remaining",
                  flush=True)

        if not any(cri_todo.values()):
            print(f"[{time.strftime('%H:%M:%S')}] {slug_column}: nothing to do",
                  flush=True)
            continue

        # Load judge ONCE server-side, swap rubric in-place between criteria.
        first_cri = next((c for c, t in cri_todo.items() if t), None)
        if first_cri is None:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] loading {judge_key} via HTTP "
              f"(initial rubric={first_cri})", flush=True)
        j = JudgeHTTP(name=judge_key, rubric=first_cri, endpoint=judge_endpoint)

        try:
            for cri, todo in cri_todo.items():
                if not todo:
                    continue
                if j.rubric_name != cri:
                    j.set_rubric(cri)
                print(f"[{time.strftime('%H:%M:%S')}] {judge_key} @ {cri} — "
                      f"{len(todo)} items", flush=True)
                props = [r[1] for r in todo]
                texts = [r[2] for r in todo]
                t0 = time.time()
                scores = j.score(props, texts)
                dt = (time.time() - t0) / 60
                rate = len(scores) / dt / 60 if dt > 0 else 0
                print(f"  scored {len(scores)} in {dt:.1f} min ({rate:.1f}/s)",
                      flush=True)
                for (rid, _, _), sc in zip(todo, scores):
                    cur.execute("""
                        INSERT OR REPLACE INTO attack_judge_scores
                        (rewrite_id, judge_slug, criterion, score, reasoning)
                        VALUES (?, ?, ?, ?, NULL)
                    """, (rid, slug_column, cri, sc))
                conn.commit()
        finally:
            try:
                j.unload()
            except Exception as e:
                print(f"  warn: unload({judge_key}) failed: {e!r}", flush=True)
        gc.collect()

    conn.close()

    if judge_server_proc is not None:
        try:
            JudgeHTTP.request_shutdown(judge_endpoint, timeout=10.0)
        except Exception:
            try:
                judge_server_proc.terminate()
            except Exception:
                pass
        try:
            judge_server_proc.wait(timeout=60)
        except Exception:
            try:
                judge_server_proc.kill()
            except Exception:
                pass
    print(f"[{time.strftime('%H:%M:%S')}] done", flush=True)


if __name__ == "__main__":
    main()
