"""SQLite schema for the rewriter evaluation suite.

Design choices:
- `paragraphs` is the central table. One row per paragraph — original OR
  rewrite. `origin_kind` says which, `base_document_id` links a rewrite
  back to the original it was derived from (NULL for originals).
- `methods` records the rewriter configuration (prompt version,
  rewriter model, sampling params). Rewrite paragraphs point to their
  method via `method_slug`.
- `evaluations` is long-format: one row per (paragraph, metric, judge
  or model) so new metrics / new criteria / new judges all append
  without schema churn. `criterion` is a string ("clarity",
  "informativeness", or NULL for metrics that aren't criterion-specific
  like word_count or ai_detector).
- `criteria` just records which criteria are in scope — currently only
  "clarity"; adding "informativeness" is one row + re-running.

Read-heavy, append-only. We WAL-journal to let concurrent writers
append evaluations without stepping on each other.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("/home/max/attack-llm-judge/data/paragraphs.db")


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS criteria (
    name           TEXT PRIMARY KEY,         -- "clarity", "informativeness"
    description    TEXT NOT NULL,
    in_scope       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS methods (
    slug                  TEXT PRIMARY KEY,   -- "naive", "lit_informed", ...
    description           TEXT NOT NULL,
    rewriter_model        TEXT NOT NULL,      -- "qwen/qwen-2.5-72b-instruct"
    rewriter_label        TEXT NOT NULL,
    prompt_version        TEXT NOT NULL,      -- e.g. "v1"
    temperature           REAL,
    max_tokens            INTEGER,
    n_samples             INTEGER DEFAULT 1,  -- N for BoN-style methods
    iterations            INTEGER DEFAULT 1,  -- rounds for iterative methods
    uses_length_retry     INTEGER DEFAULT 0,
    notes                 TEXT,
    config_json           TEXT                -- any extra config
);

CREATE TABLE IF NOT EXISTS paragraphs (
    document_id           TEXT PRIMARY KEY,   -- paul_data hex for originals, rw_<method>_<base> for rewrites
    origin_kind           TEXT NOT NULL,      -- "original_writer" | "original_model" | "original_edited" | "rewrite"
    base_document_id      TEXT,               -- originating human paragraph (NULL for AI-written originals and NULL for original_writer if it IS the base)
    method_slug           TEXT,               -- NULL for originals; FK-ish to methods.slug
    proposition_id        INTEGER NOT NULL,
    proposition           TEXT NOT NULL,
    writer_id             TEXT,               -- paul_data writer for originals; NULL for rewrites
    paul_data_model_name  TEXT,               -- for original_model/original_edited
    text                  TEXT NOT NULL,
    word_count            INTEGER NOT NULL,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    -- for original_writer only: the human top-10%-within-quintile flag (see plan.md)
    writer_is_top_decile  INTEGER,            -- 0/1, NULL if not an original_writer
    writer_agreement_quintile INTEGER,        -- 1..5 per-proposition quintile of agreement_score, NULL otherwise
    -- convenience columns aggregated from paul_data annotations (originals only)
    n_human_ratings       INTEGER,
    human_mean_clarity    REAL,
    human_mean_informativeness REAL,
    human_agreement_score REAL                -- [0,1], only for originals (aggregated)
);
CREATE INDEX IF NOT EXISTS idx_para_method ON paragraphs(method_slug);
CREATE INDEX IF NOT EXISTS idx_para_base ON paragraphs(base_document_id);
CREATE INDEX IF NOT EXISTS idx_para_kind ON paragraphs(origin_kind);
CREATE INDEX IF NOT EXISTS idx_para_prop ON paragraphs(proposition_id);

CREATE TABLE IF NOT EXISTS evaluations (
    paragraph_id      TEXT NOT NULL,
    metric            TEXT NOT NULL,          -- "judge_score" | "word_count" | "embed_sim_to_base" | "ai_detector_prob_machine" | "agreement_pred" | "clarity_regressor_pred" | "hallucinated_specifics_rate" | "hallucinated_specifics_count" | ...
    criterion         TEXT,                   -- "clarity" / "informativeness" / NULL
    source            TEXT,                   -- judge model slug for judge_score; "e5-large-v2" for embed; "binoculars" for ai_detector; etc.
    panel             TEXT,                   -- "attack" | "gold" | NULL
    value             REAL,
    extra_json        TEXT,                   -- reasoning, tokens, retries, etc.
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (paragraph_id, metric, criterion, source)
);
CREATE INDEX IF NOT EXISTS idx_eval_metric ON evaluations(metric);
CREATE INDEX IF NOT EXISTS idx_eval_panel ON evaluations(panel);
CREATE INDEX IF NOT EXISTS idx_eval_source ON evaluations(source);
"""


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=60.0)
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def initialise(db_path: Path = DEFAULT_DB_PATH) -> None:
    con = connect(db_path)
    try:
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    initialise()
    print(f"Initialised {DEFAULT_DB_PATH}")
