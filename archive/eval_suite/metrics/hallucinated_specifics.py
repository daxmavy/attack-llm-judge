"""Metric: hallucinated specifics (addition 'c' approved by operator).

Definition: a "specific" is a numeric expression, a percentage, a year,
a proper-noun NER entity (ORG/PER/GPE/NORP/EVENT/LAW/WORK_OF_ART), or a
quoted phrase. A specific is "introduced" if it appears in the rewrite
but is not present (by substring or fuzzy match) in the base paragraph.

We record two metrics per rewrite:
- hallucinated_specifics_count (absolute count of new specifics)
- hallucinated_specifics_rate  (count / max(1, rewrite_words) * 100)

Benchmark on paul_data:
- Compute the same metric on the `original_edited` vs `original_writer`
  parallel pairs (treating edited as the "rewrite" and writer as the
  "base"). Human edits should rarely introduce novel entities; a small
  mean + tight tail here is what we expect.
- Compute on `original_model` vs the writer paragraph that shares the
  same (proposition_id) — this is NOT the same "base" relationship, so
  we interpret the number as a topical-drift baseline rather than a
  hallucination rate, and report it for contrast.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC_COUNT = "hallucinated_specifics_count"
METRIC_RATE = "hallucinated_specifics_rate_per100w"
SOURCE = "regex+spacy_en_core_web_sm"

NUM_RE = re.compile(r"""
    \b(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)%?       # numbers or percentages (1, 23.5, 3,000, 70%)
    | \b(?:19|20)\d{2}\b                           # years
    | \$\d+(?:\.\d+)?                              # $-prefixed
""", re.VERBOSE)
QUOTE_RE = re.compile(r'"([^"]{6,})"')


_NLP = None


def _nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        # If the model is not present, download it.
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


ENT_TYPES = {"PERSON", "ORG", "GPE", "NORP", "EVENT", "LAW", "WORK_OF_ART", "MONEY", "PERCENT",
             "QUANTITY", "CARDINAL", "DATE"}


def extract_specifics(text: str) -> set[str]:
    """Return a set of normalised specifics found in the text."""
    specs: set[str] = set()
    for m in NUM_RE.finditer(text):
        specs.add(m.group(0).lower().replace(",", ""))
    for m in QUOTE_RE.finditer(text):
        specs.add(("q:" + m.group(1).lower().strip())[:120])
    doc = _nlp()(text)
    for ent in doc.ents:
        if ent.label_ in ENT_TYPES and len(ent.text.strip()) >= 2:
            specs.add((ent.label_ + ":" + ent.text.lower().strip())[:80])
    return specs


def _normalise_reference(text: str) -> str:
    return text.lower().replace(",", "")


def count_introduced(rewrite_text: str, base_text: str) -> int:
    """Count specifics in rewrite_text that don't appear (substring) in base_text."""
    rw_specs = extract_specifics(rewrite_text)
    ref = _normalise_reference(base_text)
    count = 0
    for s in rw_specs:
        if ":" in s:
            # entity or quote form
            label, value = s.split(":", 1)
            if value not in ref:
                count += 1
        else:
            if s not in ref:
                count += 1
    return count


def _apply_df(df: pd.DataFrame) -> pd.DataFrame:
    counts = []
    rates = []
    for _, r in df.iterrows():
        n = count_introduced(r["rewrite_text"], r["base_text"])
        wc = max(1, len(r["rewrite_text"].split()))
        counts.append(n)
        rates.append(100.0 * n / wc)
    out = df.copy()
    out["count"] = counts
    out["rate_per_100w"] = rates
    return out


def score_rewrites(db_path: Path = DEFAULT_DB_PATH) -> int:
    con = connect(db_path)
    try:
        pairs = pd.read_sql_query("""
            SELECT p.document_id AS rewrite_id, p.text AS rewrite_text, b.text AS base_text
            FROM paragraphs p
            JOIN paragraphs b ON b.document_id = p.base_document_id
            WHERE p.base_document_id IS NOT NULL
        """, con)
        if len(pairs) == 0:
            return 0
        res = _apply_df(pairs)
        rows = []
        for _, r in res.iterrows():
            rows.append(MetricRow(r["rewrite_id"], METRIC_COUNT, None, SOURCE, None,
                                   float(r["count"]), None).to_tuple())
            rows.append(MetricRow(r["rewrite_id"], METRIC_RATE, None, SOURCE, None,
                                   float(r["rate_per_100w"]), None).to_tuple())
        write_rows(con, rows)
        return len(rows)
    finally:
        con.close()


def benchmark(db_path: Path = DEFAULT_DB_PATH, n_model_sample: int = 400) -> dict:
    con = connect(db_path)
    try:
        # parallel edited vs writer (true "rewrite of same base")
        edited = pd.read_sql_query("""
            SELECT w.text AS base_text, e.text AS rewrite_text
            FROM paragraphs w
            JOIN paragraphs e ON e.writer_id = w.writer_id AND e.proposition_id = w.proposition_id
            WHERE w.origin_kind='original_writer' AND e.origin_kind='original_edited'
        """, con)
        edited_res = _apply_df(edited) if len(edited) else pd.DataFrame()
        # model vs same-writer-proposition paragraph (topical baseline, NOT a true rewrite)
        model = pd.read_sql_query(f"""
            SELECT w.text AS base_text, m.text AS rewrite_text
            FROM paragraphs w
            JOIN paragraphs m ON m.writer_id = w.writer_id AND m.proposition_id = w.proposition_id
            WHERE w.origin_kind='original_writer' AND m.origin_kind='original_model'
            ORDER BY RANDOM()
            LIMIT {int(n_model_sample)}
        """, con)
        model_res = _apply_df(model) if len(model) else pd.DataFrame()
        def stats(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"n": 0}
            return {
                "n": int(len(df)),
                "count_mean": float(df["count"].mean()),
                "count_median": float(df["count"].median()),
                "count_p95": float(df["count"].quantile(0.95)),
                "rate_mean": float(df["rate_per_100w"].mean()),
                "rate_p95": float(df["rate_per_100w"].quantile(0.95)),
            }
        return {
            "parallel_edited_vs_writer": stats(edited_res),
            "model_vs_writer_topical_baseline": stats(model_res),
        }
    finally:
        con.close()


if __name__ == "__main__":
    import json
    print(json.dumps(benchmark(), indent=2))
