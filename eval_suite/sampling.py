"""Stratified sampling of human-written paragraphs.

Per operator (2026-04-17): evaluation is restricted to human-written
paragraphs (writer), and we evaluate on a stratified subsample rather
than the full corpus. The sampler stratifies by (agreement_quintile x
top_decile) which is the structure we set up in ingest.py:

- agreement_quintile  : per-proposition quintile of agreement_score (1..5)
- writer_is_top_decile: 1 if this doc is in the top 10% by
                        mean_human_clarity within its (proposition,
                        quintile) cell

Strata: 5 quintiles x 2 top-decile classes = 10. We sample
proportionally to each stratum's size so the subsample mirrors the
population distribution. Top-decile docs are ~10% per quintile so they
stay ~10% of the subsample, not over-represented.

Records the sampled document_ids + their stratum to a tagged table
(`sampled_writers`) so the downstream judge/metric runners can join
without re-sampling.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eval_suite.schema import DEFAULT_DB_PATH, connect


CREATE_SAMPLE_SQL = """
CREATE TABLE IF NOT EXISTS sampled_writers (
    tag                TEXT NOT NULL,
    document_id        TEXT NOT NULL,
    stratum            TEXT NOT NULL,
    writer_agreement_quintile INTEGER,
    writer_is_top_decile INTEGER,
    PRIMARY KEY (tag, document_id)
);
CREATE INDEX IF NOT EXISTS idx_sample_tag ON sampled_writers(tag);
"""


def _stratify_and_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = []
    for (q, d), g in df.groupby(["writer_agreement_quintile", "writer_is_top_decile"]):
        n = int(math.ceil(frac * len(g)))
        n = max(1, min(n, len(g)))
        chosen = g.sample(n=n, random_state=rng.integers(0, 2**31 - 1))
        chosen = chosen.assign(stratum=f"q{q}_top{int(d)}")
        groups.append(chosen)
    return pd.concat(groups, ignore_index=True)


def create_sample(tag: str, frac: float = 0.20, seed: int = 17,
                   db_path: Path = DEFAULT_DB_PATH) -> dict:
    con = connect(db_path)
    try:
        con.executescript(CREATE_SAMPLE_SQL)
        df = pd.read_sql_query("""
            SELECT document_id, proposition_id, writer_agreement_quintile,
                   writer_is_top_decile, human_mean_clarity, human_agreement_score
            FROM paragraphs
            WHERE origin_kind='original_writer'
              AND writer_agreement_quintile IS NOT NULL
        """, con)
        sampled = _stratify_and_sample(df, frac, seed)
        # Clear prior rows with this tag.
        con.execute("DELETE FROM sampled_writers WHERE tag=?", (tag,))
        rows = [(tag, r["document_id"], r["stratum"],
                  int(r["writer_agreement_quintile"]),
                  int(r["writer_is_top_decile"]))
                for _, r in sampled.iterrows()]
        con.executemany("INSERT OR REPLACE INTO sampled_writers VALUES (?,?,?,?,?)", rows)
        con.commit()
        by_stratum = sampled.groupby("stratum").size().to_dict()
        return {
            "tag": tag,
            "seed": seed,
            "frac": frac,
            "n_total_population": int(len(df)),
            "n_sampled": int(len(sampled)),
            "by_stratum": {k: int(v) for k, v in by_stratum.items()},
        }
    finally:
        con.close()


def get_sample_ids(tag: str, db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    con = connect(db_path)
    try:
        cur = con.execute("SELECT document_id FROM sampled_writers WHERE tag=?", (tag,))
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main_20pct")
    p.add_argument("--frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    import json
    print(json.dumps(create_sample(args.tag, args.frac, args.seed), indent=2))
