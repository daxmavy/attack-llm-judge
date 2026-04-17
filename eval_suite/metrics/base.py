"""Common interface for DB-native metrics.

Every metric exposes:
- `METRIC_NAME` / `SOURCE`: keys into the evaluations table.
- `score(paragraphs_df) -> DataFrame`: produce values for each row (the
  caller is responsible for picking which rows to score).
- `write_to_db(con, rows)`: append rows to the evaluations table.
- `benchmark(db_path)` (optional): run on paul_data's labelled human /
  AI / edited paragraphs and report headline numbers. Required for any
  metric where the benchmark is defensible (detector AUROC, regressor
  Pearson vs human ratings, embedding-similarity human-vs-AI separation,
  etc.) per operator's instruction.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


INSERT_EVAL_SQL = """
INSERT OR REPLACE INTO evaluations(paragraph_id, metric, criterion, source, panel, value, extra_json)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def write_rows(con: sqlite3.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    con.executemany(INSERT_EVAL_SQL, rows)
    con.commit()
    return len(rows)


@dataclass
class MetricRow:
    paragraph_id: str
    metric: str
    criterion: str | None
    source: str
    panel: str | None
    value: float | None
    extra: dict | None = None

    def to_tuple(self) -> tuple:
        return (self.paragraph_id, self.metric, self.criterion, self.source, self.panel,
                self.value, json.dumps(self.extra) if self.extra else None)
