"""Parameterised judge runner.

Scores any iterable of paragraphs with any panel and any criterion,
writing results straight into the `evaluations` table (long format).
Idempotent: PRIMARY KEY (paragraph_id, metric, criterion, source)
means re-running skips rows already present unless --force is passed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from judge.client import call_judge, load_api_key
from judge.rubrics import SYSTEM_PROMPT as JUDGE_SYSTEM, build_prompt
from eval_suite.panels import JudgeConfig, panel as get_panel
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC_NAME = "judge_score"


def existing_eval_keys(con: sqlite3.Connection, criterion: str, panel_name: str) -> set:
    cur = con.execute(
        """SELECT paragraph_id, source FROM evaluations
           WHERE metric=? AND criterion=? AND panel=?""",
        (METRIC_NAME, criterion, panel_name))
    return {(r[0], r[1]) for r in cur.fetchall()}


def load_paragraphs_for_judging(con: sqlite3.Connection, where: str | None = None) -> pd.DataFrame:
    sql = "SELECT document_id, proposition, text FROM paragraphs"
    if where:
        sql += f" WHERE {where}"
    return pd.read_sql_query(sql, con)


INSERT_EVAL_SQL = """
INSERT OR REPLACE INTO evaluations(paragraph_id, metric, criterion, source, panel, value, extra_json)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def run_panel(
    criterion: str,
    panel_name: str,
    where: str | None = None,
    max_workers: int = 8,
    force: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
    dry_run: bool = False,
) -> dict:
    """Score every paragraph matching `where` with every judge in the panel.

    Returns a small summary dict (n_calls_queued, n_calls_done, cost estimate).
    """
    api_key = load_api_key()
    judges = get_panel(panel_name)
    con = connect(db_path)
    try:
        paragraphs = load_paragraphs_for_judging(con, where)
        skip_keys = set() if force else existing_eval_keys(con, criterion, panel_name)
        tasks = []
        for _, row in paragraphs.iterrows():
            for j in judges:
                if (row["document_id"], j.slug) in skip_keys:
                    continue
                tasks.append((row["document_id"], row["proposition"], row["text"], j))
        print(f"[{panel_name}/{criterion}] paragraphs={len(paragraphs)} judges={len(judges)} "
              f"calls_queued={len(tasks)} (skipped {len(skip_keys)} existing)")
        if dry_run:
            return {"queued": len(tasks), "done": 0, "cost_usd": 0.0}
        if not tasks:
            return {"queued": 0, "done": 0, "cost_usd": 0.0}

        def _go(t):
            doc_id, prop, text, j = t
            prompt = build_prompt(criterion, prop, text)
            r = call_judge(j.model_id, JUDGE_SYSTEM, prompt, api_key, max_tokens=250, temperature=0.0)
            extra = {
                "ok": bool(r.ok),
                "error": r.error,
                "reasoning": r.reasoning,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "judge_model_id": j.model_id,
            }
            return (doc_id, j, r, extra)

        t0 = time.time()
        done = 0
        total_cost = 0.0
        # Commit in batches.
        batch = []
        BATCH = 50
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for doc_id, j, r, extra in ex.map(_go, tasks):
                done += 1
                score_val = float(r.score) if (r.ok and r.score is not None) else None
                batch.append((doc_id, METRIC_NAME, criterion, j.slug, j.panel, score_val,
                              json.dumps(extra)))
                # Cost accounting.
                p_tokens = r.prompt_tokens or 0
                c_tokens = r.completion_tokens or 0
                total_cost += (p_tokens / 1e6) * j.prompt_in_per_1m_usd \
                            + (c_tokens / 1e6) * j.prompt_out_per_1m_usd
                if len(batch) >= BATCH:
                    con.executemany(INSERT_EVAL_SQL, batch)
                    con.commit()
                    batch = []
                if done % 50 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    print(f"  [{panel_name}/{criterion}] {done}/{len(tasks)} "
                          f"elapsed={elapsed:.0f}s rate={rate:.1f}/s cost=${total_cost:.4f}")
        if batch:
            con.executemany(INSERT_EVAL_SQL, batch)
            con.commit()
        return {"queued": len(tasks), "done": done, "cost_usd": float(total_cost)}
    finally:
        con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--criterion", default="clarity")
    p.add_argument("--panel", default="attack", choices=["attack", "gold", "all"])
    p.add_argument("--where", default=None)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.panel == "all":
        for pnl in ("attack", "gold"):
            print(run_panel(args.criterion, pnl, args.where, args.max_workers, args.force,
                            dry_run=args.dry_run))
    else:
        print(run_panel(args.criterion, args.panel, args.where, args.max_workers, args.force,
                        dry_run=args.dry_run))
