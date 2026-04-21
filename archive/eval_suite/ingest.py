"""Ingest paul_data originals into the paragraphs DB.

Populates:
- criteria rows (currently only clarity; informativeness disabled but
  declared so the schema is ready).
- paragraphs rows for every row in paul_data/prepared/documents.csv
  (writer, model, edited).
- Human rating aggregates (mean clarity / informativeness, n_ratings,
  agreement_score) from main_phase_2/annotations.csv + the prepared
  table's own agreement_score column.
- The top-10%-within-quintile flag for writer paragraphs.

Flag interpretation (see plan.md + CLAUDE.md dialogue):
  For each proposition, rank its writer paragraphs by agreement_score
  and bin into 5 equal-size quintiles (1 = lowest, 5 = highest). Within
  each (proposition x agreement_quintile) cell, take the top 10% of
  documents ranked by human_mean_clarity. This "top decile" flag marks
  high-clarity exemplars within each stance tier of each proposition.
  Rationale: we want high-quality writing exemplars AT EACH STANCE
  LEVEL, so the flag is stance-stratified rather than a raw top-10%
  over the whole corpus (which would concentrate on one side of each
  proposition).

Because clarity is the ONLY criterion in scope right now, we use
human_mean_clarity as the within-cell ranker. When informativeness is
added later, the flag stays the same (or we add a second flag column —
see TODO).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_suite.schema import DEFAULT_DB_PATH, connect, initialise


REPO = Path(__file__).resolve().parent.parent
DOCS_CSV = REPO / "paul_data" / "prepared" / "documents.csv"
ANN_CSV = REPO / "paul_data" / "main_phase_2" / "annotations.csv"


CRITERIA = [
    {"name": "clarity", "description": "How well-written, grammatical, and easy to read the paragraph is", "in_scope": 1},
    {"name": "informativeness", "description": "How much substantive content the paragraph provides on the proposition", "in_scope": 0},
]


def _assign_quintile(s: pd.Series) -> pd.Series:
    """Per-proposition equal-count quintiles of agreement_score (1..5)."""
    if len(s) < 5:
        # Few docs — collapse to single-bucket; still label 1..n to avoid NaNs.
        ranks = s.rank(method="first")
        return pd.Series(np.ceil(ranks / (len(s) / 5)).clip(1, 5).astype(int), index=s.index)
    try:
        q = pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        return q.astype(int)
    except ValueError:
        # All identical values — fall back to a single bucket labelled 3 (middle).
        return pd.Series([3] * len(s), index=s.index)


def compute_writer_flags(docs: pd.DataFrame, human_agg: pd.DataFrame) -> pd.DataFrame:
    """Return docs df with two extra cols for writer rows: writer_agreement_quintile, writer_is_top_decile."""
    w = docs[docs["paragraph_type"] == "writer"].copy()
    w = w.merge(human_agg, on=["writer_id", "proposition_id", "paragraph_type", "model_name",
                                "model_input_condition"], how="left")
    # Quintile by agreement_score, within proposition.
    w["writer_agreement_quintile"] = (
        w.groupby("proposition_id")["agreement_score"].transform(_assign_quintile)
    )
    # Top 10% by mean_human_clarity, within (proposition, quintile).
    def _flag_top_decile(g: pd.DataFrame) -> pd.Series:
        if g["mean_clarity"].isna().all() or len(g) == 0:
            return pd.Series([0] * len(g), index=g.index)
        # int(ceil(0.10 * n)) winners per cell; minimum 1 if cell has any docs.
        k = max(1, int(np.ceil(0.10 * len(g))))
        top_idx = g["mean_clarity"].nlargest(k).index
        out = pd.Series([0] * len(g), index=g.index)
        out.loc[top_idx] = 1
        return out

    w["writer_is_top_decile"] = (
        w.groupby(["proposition_id", "writer_agreement_quintile"], dropna=False, group_keys=False)
         .apply(_flag_top_decile)
    )
    return w[["writer_id", "proposition_id", "writer_agreement_quintile", "writer_is_top_decile"]]


def load_human_aggregates() -> pd.DataFrame:
    ann = pd.read_csv(ANN_CSV)
    key = ["writer_id", "proposition_id", "paragraph_type", "model_name", "model_input_condition"]
    agg = ann.groupby(key, dropna=False).agg(
        n_ratings=("paragraph_clarity", "size"),
        mean_clarity=("paragraph_clarity", "mean"),
        mean_informativeness=("paragraph_informativeness", "mean"),
    ).reset_index()
    return agg


def build_paragraph_rows() -> list[dict]:
    docs = pd.read_csv(DOCS_CSV)
    agg = load_human_aggregates()
    flags = compute_writer_flags(docs, agg)

    joined = docs.merge(
        agg, on=["writer_id", "proposition_id", "paragraph_type", "model_name", "model_input_condition"],
        how="left",
    ).merge(flags, on=["writer_id", "proposition_id"], how="left")

    rows = []
    for _, r in joined.iterrows():
        pt = r["paragraph_type"]
        origin_kind = {
            "writer": "original_writer",
            "model":  "original_model",
            "edited": "original_edited",
        }[pt]
        text = str(r["document_text"])
        wc = len(text.split())
        rows.append({
            "document_id": r["document_id"],
            "origin_kind": origin_kind,
            "base_document_id": None,
            "method_slug": None,
            "proposition_id": int(r["proposition_id"]),
            "proposition": r["proposition"],
            "writer_id": r["writer_id"],
            "paul_data_model_name": None if pd.isna(r.get("model_name")) else r["model_name"],
            "text": text,
            "word_count": wc,
            "writer_is_top_decile": (int(r["writer_is_top_decile"]) if pt == "writer" and not pd.isna(r.get("writer_is_top_decile")) else None),
            "writer_agreement_quintile": (int(r["writer_agreement_quintile"]) if pt == "writer" and not pd.isna(r.get("writer_agreement_quintile")) else None),
            "n_human_ratings": (int(r["n_ratings"]) if not pd.isna(r.get("n_ratings")) else None),
            "human_mean_clarity": (float(r["mean_clarity"]) if not pd.isna(r.get("mean_clarity")) else None),
            "human_mean_informativeness": (float(r["mean_informativeness"]) if not pd.isna(r.get("mean_informativeness")) else None),
            "human_agreement_score": (float(r["agreement_score"]) if not pd.isna(r.get("agreement_score")) else None),
        })
    return rows


INSERT_CRITERION_SQL = """
INSERT OR REPLACE INTO criteria(name, description, in_scope) VALUES(?, ?, ?);
"""
INSERT_PARAGRAPH_SQL = """
INSERT OR REPLACE INTO paragraphs(
    document_id, origin_kind, base_document_id, method_slug, proposition_id, proposition,
    writer_id, paul_data_model_name, text, word_count,
    writer_is_top_decile, writer_agreement_quintile,
    n_human_ratings, human_mean_clarity, human_mean_informativeness, human_agreement_score
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def ingest(db_path=DEFAULT_DB_PATH) -> None:
    initialise(db_path)
    rows = build_paragraph_rows()
    con = connect(db_path)
    try:
        con.executemany(INSERT_CRITERION_SQL,
                        [(c["name"], c["description"], c["in_scope"]) for c in CRITERIA])
        con.executemany(INSERT_PARAGRAPH_SQL, [(
            r["document_id"], r["origin_kind"], r["base_document_id"], r["method_slug"],
            r["proposition_id"], r["proposition"], r["writer_id"], r["paul_data_model_name"],
            r["text"], r["word_count"],
            r["writer_is_top_decile"], r["writer_agreement_quintile"],
            r["n_human_ratings"], r["human_mean_clarity"], r["human_mean_informativeness"],
            r["human_agreement_score"],
        ) for r in rows])
        con.commit()
        # Print a summary.
        cur = con.execute("SELECT origin_kind, COUNT(*) FROM paragraphs GROUP BY origin_kind;")
        print("paragraphs by origin_kind:", dict(cur.fetchall()))
        cur = con.execute("""
            SELECT writer_agreement_quintile, COUNT(*) AS n, SUM(writer_is_top_decile) AS top
            FROM paragraphs WHERE origin_kind='original_writer'
            GROUP BY writer_agreement_quintile ORDER BY writer_agreement_quintile;""")
        print("writer quintiles (n, top_decile):", cur.fetchall())
    finally:
        con.close()


if __name__ == "__main__":
    ingest()
