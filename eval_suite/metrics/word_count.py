"""Metric 3: word-count change.

For originals: just stores word_count as absolute.
For rewrites: stores both absolute word_count AND delta/ratio vs. the base human paragraph.

Benchmark: (not really applicable — word count is deterministic). Instead, we report
distribution stats across paragraph types in paul_data as sanity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC_ABSOLUTE = "word_count"
METRIC_DELTA = "word_count_delta_vs_base"
METRIC_RATIO = "word_count_ratio_vs_base"
SOURCE = "tokenizer_whitespace"


def score_all(db_path: Path = DEFAULT_DB_PATH) -> int:
    con = connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT p.document_id, p.word_count AS wc, p.base_document_id, p2.word_count AS base_wc
            FROM paragraphs p
            LEFT JOIN paragraphs p2 ON p2.document_id = p.base_document_id
        """, con)
        rows = []
        for _, r in df.iterrows():
            rows.append(MetricRow(r["document_id"], METRIC_ABSOLUTE, None, SOURCE, None,
                                   float(r["wc"]), None).to_tuple())
            if pd.notna(r["base_wc"]):
                rows.append(MetricRow(r["document_id"], METRIC_DELTA, None, SOURCE, None,
                                       float(r["wc"] - r["base_wc"]), None).to_tuple())
                rows.append(MetricRow(r["document_id"], METRIC_RATIO, None, SOURCE, None,
                                       float(r["wc"] / r["base_wc"]) if r["base_wc"] else None,
                                       None).to_tuple())
        write_rows(con, rows)
        return len(rows)
    finally:
        con.close()


def benchmark(db_path: Path = DEFAULT_DB_PATH) -> dict:
    con = connect(db_path)
    try:
        df = pd.read_sql_query("""SELECT origin_kind, word_count FROM paragraphs""", con)
        out = df.groupby("origin_kind")["word_count"].describe().to_dict(orient="index")
        return {"word_count_by_origin_kind": out}
    finally:
        con.close()


if __name__ == "__main__":
    print("wrote rows:", score_all())
    from pprint import pprint; pprint(benchmark())
