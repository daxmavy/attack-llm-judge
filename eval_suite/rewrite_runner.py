"""DB-native rewriter driver.

For a given method slug and a set of source paragraphs, generate
rewrites and insert them into the `paragraphs` table. Idempotent:
rewrite row IDs are deterministic (`rw_<method>_<base_id>`) so rerunning
skips existing rows unless --force is passed.

Registers methods in the `methods` table with their config so the DB is
self-describing.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from eval_suite.schema import DEFAULT_DB_PATH, connect
from judge.client import load_api_key
from rewriters.orchestrators import (
    run_bon_panel,
    run_icir_single,
    run_injection_leadin,
    run_scaffolded,
    run_simple,
)
from rewriters.rewriter_client import REWRITER_LABEL, REWRITER_MODEL


# Method slug -> registration metadata. 2026-04-17: operator pruned
# `lit_informed`, `naive_tight`, `rules_explicit`, `scaffolded_cot_distill`
# from the minimum set. Remaining 6 methods.
METHOD_SPEC = {
    "naive":              {"calls_pp": 1.3,  "t": 0.7, "simple": True},
    "lit_informed_tight": {"calls_pp": 1.3,  "t": 0.7, "simple": True},
    "injection_leadin":   {"calls_pp": 1.3,  "t": 0.4, "simple": False},
    "rubric_aware":       {"calls_pp": 1.3,  "t": 0.5, "simple": True},
    # icir_single now scores with BOTH in-panel judges (Qwen-2.5-7B + Llama-3.1-8B)
    # and averages; plan-of-record per MODELS.md is that the training signal is
    # mean of the two in-panel judges, so the ICIR feedback loop should match.
    "icir_single":        {"calls_pp": 15.0, "t": 0.5, "simple": False},
    # bon_panel uses the 2-judge attack panel (was 5 before the minimum-set change).
    "bon_panel":          {"calls_pp": 10.0, "t": 1.0, "simple": False},
}


def rewrite_id(method: str, base_id: str) -> str:
    return f"rw_{method}_{base_id}"


INSERT_METHOD_SQL = """
INSERT OR REPLACE INTO methods(slug, description, rewriter_model, rewriter_label, prompt_version,
    temperature, max_tokens, n_samples, iterations, uses_length_retry, notes, config_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def register_method(con: sqlite3.Connection, method: str) -> None:
    spec = METHOD_SPEC[method]
    con.execute(INSERT_METHOD_SQL, (
        method,
        f"Rewriter method '{method}' — see background_docs/methods_5_10_design.md and rewriters/rewrite_prompts.py",
        REWRITER_MODEL,
        REWRITER_LABEL,
        "v3",
        float(spec["t"]),
        400,
        8 if method == "bon_panel" else 1,
        4 if method == "icir_single" else 1,
        1,
        None,
        json.dumps({"estimated_calls_per_paragraph": spec["calls_pp"]}),
    ))
    con.commit()


INSERT_PARAGRAPH_SQL = """
INSERT OR REPLACE INTO paragraphs(
    document_id, origin_kind, base_document_id, method_slug, proposition_id, proposition,
    writer_id, paul_data_model_name, text, word_count,
    writer_is_top_decile, writer_agreement_quintile,
    n_human_ratings, human_mean_clarity, human_mean_informativeness, human_agreement_score
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def load_source_paragraphs(con: sqlite3.Connection, origin_kind: str = "original_writer",
                            limit: int | None = None,
                            only_top_decile: bool = False,
                            sample_tag: str | None = None) -> pd.DataFrame:
    if sample_tag:
        sql = f"""SELECT p.document_id, p.proposition_id, p.proposition, p.text
                   FROM paragraphs p
                   JOIN sampled_writers s ON s.document_id = p.document_id
                   WHERE s.tag = ? AND p.origin_kind = ?"""
        params = (sample_tag, origin_kind)
    else:
        sql = f"SELECT document_id, proposition_id, proposition, text FROM paragraphs WHERE origin_kind=?"
        params = (origin_kind,)
    if only_top_decile and origin_kind == "original_writer":
        sql += " AND p.writer_is_top_decile=1" if sample_tag else " AND writer_is_top_decile=1"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql_query(sql, con, params=params)


def existing_rewrite_ids(con: sqlite3.Connection, method: str, base_ids: list[str]) -> set:
    if not base_ids:
        return set()
    placeholders = ",".join(["?"] * len(base_ids))
    cur = con.execute(
        f"SELECT base_document_id FROM paragraphs WHERE method_slug=? AND base_document_id IN ({placeholders})",
        [method] + base_ids,
    )
    return {r[0] for r in cur.fetchall()}


def do_one(method: str, row: dict, api_key: str, rot_index: int) -> dict:
    prop = row["proposition"]
    para = row["text"]
    if method in ("naive", "lit_informed", "naive_tight", "lit_informed_tight",
                  "rules_explicit", "rubric_aware"):
        return run_simple(method, prop, para, api_key,
                          temperature=METHOD_SPEC[method]["t"])
    if method == "injection_leadin":
        return run_injection_leadin(prop, para, api_key, rot_index)
    if method == "scaffolded_cot_distill":
        return run_scaffolded(prop, para, api_key)
    if method == "icir_single":
        return run_icir_single(prop, para, api_key)
    if method == "bon_panel":
        return run_bon_panel(prop, para, api_key)
    raise ValueError(method)


def run_method(method: str,
               origin_kind: str = "original_writer",
               limit: int | None = None,
               max_workers: int = 4,
               force: bool = False,
               only_top_decile: bool = False,
               sample_tag: str | None = None,
               db_path: Path = DEFAULT_DB_PATH) -> dict:
    api_key = load_api_key()
    con = connect(db_path)
    try:
        register_method(con, method)
        src = load_source_paragraphs(con, origin_kind=origin_kind, limit=limit,
                                      only_top_decile=only_top_decile,
                                      sample_tag=sample_tag)
        if not force:
            existing = existing_rewrite_ids(con, method, src["document_id"].tolist())
            src = src[~src["document_id"].isin(existing)].reset_index(drop=True)
        print(f"[{method}] source paragraphs to rewrite: {len(src)}")
        if len(src) == 0:
            return {"method": method, "n_done": 0, "cost_rewriter_usd": 0.0}

        # Pre-load proposition_id for back-reference.
        prop_ids = dict(zip(src["document_id"], src["proposition_id"]))

        def _go(i_row):
            i, row = i_row
            return (row["document_id"], do_one(method, row, api_key, rot_index=i))

        t0 = time.time()
        done = 0
        batch = []
        BATCH_N = 25
        total_p = 0
        total_c = 0
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for base_id, res in ex.map(_go, list(src.iterrows())):
                done += 1
                if not res["ok"] or not res["text"]:
                    continue
                total_p += res.get("prompt_tokens") or 0
                total_c += res.get("completion_tokens") or 0
                text = res["text"]
                wc = len(text.split())
                rw_id = rewrite_id(method, base_id)
                prop = src.loc[src["document_id"] == base_id].iloc[0]
                batch.append((
                    rw_id, "rewrite", base_id, method, int(prop_ids[base_id]),
                    prop["proposition"], None, None, text, wc,
                    None, None, None, None, None, None,
                ))
                if len(batch) >= BATCH_N:
                    con.executemany(INSERT_PARAGRAPH_SQL, batch)
                    con.commit()
                    batch = []
                if done % 20 == 0 or done == len(src):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    print(f"  [{method}] {done}/{len(src)} elapsed={elapsed:.0f}s rate={rate:.2f}/s")
        if batch:
            con.executemany(INSERT_PARAGRAPH_SQL, batch)
            con.commit()
        # Qwen 2.5 72B Instruct list price on OpenRouter: ~$0.13 / $0.40 per 1M.
        cost = total_p / 1e6 * 0.13 + total_c / 1e6 * 0.40
        return {"method": method, "n_done": done, "cost_rewriter_usd": float(cost)}
    finally:
        con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHOD_SPEC))
    p.add_argument("--origin-kind", default="original_writer")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only-top-decile", action="store_true")
    p.add_argument("--sample-tag", default=None)
    args = p.parse_args()
    print(json.dumps(run_method(**vars(args)), indent=2))
