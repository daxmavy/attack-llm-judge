"""Initialise paragraphs.db for a fresh replication run.

- Builds `paragraphs` (the master paragraph table) from the 1805-row dataset JSON.
- Creates empty `attack_rewrites` / `attack_judge_scores` / `attack_agreement_scores`
  tables with the schema the pipeline scripts expect.

Idempotent: uses CREATE IF NOT EXISTS + INSERT OR REPLACE.
Run once after cloning the repo on a new host.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DEFAULT_DB = "/home/shil6647/attack-llm-judge/data/paragraphs.db"
DEFAULT_DATASET = "/home/shil6647/attack-llm-judge/data/controversial_40_3fold.json"


SCHEMA = """
CREATE TABLE IF NOT EXISTS paragraphs (
    document_id                TEXT PRIMARY KEY,
    proposition_id             INTEGER,
    proposition                TEXT NOT NULL,
    text                       TEXT NOT NULL,
    word_count                 INTEGER,
    human_mean_clarity         REAL,
    human_mean_informativeness REAL,
    human_agreement_score      REAL,
    origin_kind                TEXT NOT NULL,
    writer_is_top_decile       INTEGER,
    split                      TEXT
);
CREATE INDEX IF NOT EXISTS idx_paragraphs_proposition_id ON paragraphs(proposition_id);

CREATE TABLE IF NOT EXISTS attack_rewrites (
    rewrite_id           TEXT PRIMARY KEY,
    source_doc_id        TEXT NOT NULL,
    method               TEXT NOT NULL,
    fold                 INTEGER,
    criterion            TEXT,
    config_json          TEXT,
    rewriter_model       TEXT,
    judge_panel_json     TEXT,
    text                 TEXT NOT NULL,
    word_count           INTEGER,
    run_metadata_json    TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_doc_id) REFERENCES paragraphs(document_id)
);
CREATE INDEX IF NOT EXISTS idx_rewrites_method    ON attack_rewrites(method);
CREATE INDEX IF NOT EXISTS idx_rewrites_criterion ON attack_rewrites(criterion);
CREATE INDEX IF NOT EXISTS idx_rewrites_fold      ON attack_rewrites(fold);
CREATE INDEX IF NOT EXISTS idx_rewrites_rewriter  ON attack_rewrites(rewriter_model);
CREATE INDEX IF NOT EXISTS idx_rewrites_doc       ON attack_rewrites(source_doc_id);

CREATE TABLE IF NOT EXISTS attack_judge_scores (
    rewrite_id   TEXT NOT NULL,
    judge_slug   TEXT NOT NULL,
    criterion    TEXT NOT NULL,
    score        REAL NOT NULL,
    reasoning    TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rewrite_id, judge_slug, criterion),
    FOREIGN KEY (rewrite_id) REFERENCES attack_rewrites(rewrite_id)
);
CREATE INDEX IF NOT EXISTS idx_judge_scores_judge     ON attack_judge_scores(judge_slug);
CREATE INDEX IF NOT EXISTS idx_judge_scores_criterion ON attack_judge_scores(criterion);

CREATE TABLE IF NOT EXISTS attack_agreement_scores (
    rewrite_id   TEXT PRIMARY KEY,
    score        REAL NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)
    conn.commit()

    payload = json.loads(Path(args.dataset).read_text())
    rows = payload["rows"]
    print(f"ingesting {len(rows)} paragraph rows from {args.dataset}")

    insert_sql = """
        INSERT OR REPLACE INTO paragraphs
        (document_id, proposition_id, proposition, text, word_count,
         human_mean_clarity, human_mean_informativeness, human_agreement_score,
         origin_kind, writer_is_top_decile, split)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for r in rows:
        conn.execute(insert_sql, (
            r["document_id"],
            r.get("proposition_id"),
            r["proposition"],
            r["text"],
            r.get("word_count") or len(r["text"].split()),
            r.get("human_mean_clarity"),
            r.get("human_mean_informativeness"),
            r.get("human_agreement_score"),
            "original_writer",
            1,  # every row in the subset is top-decile by construction
            r.get("split"),
        ))
    conn.commit()

    counts = {}
    for t in ("paragraphs", "attack_rewrites", "attack_judge_scores", "attack_agreement_scores"):
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    conn.close()

    print("row counts:")
    for t, n in counts.items():
        print(f"  {t:30s} {n}")


if __name__ == "__main__":
    main()
