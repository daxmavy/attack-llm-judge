"""Semantic-fidelity metric (addition 'a' approved by operator).

Cosine similarity between a rewrite and its base human paragraph, using
a sentence-transformers model. We default to `intfloat/e5-large-v2`
because it's strong and currently ubiquitous in text-similarity
benchmarks; swap to `sentence-transformers/all-mpnet-base-v2` if VRAM
or download is an issue.

Only applies to rewrites (they have a `base_document_id`).

Benchmark on paul_data:
- Pair every `original_edited` paragraph with its corresponding
  `original_writer` paragraph (shared writer_id + proposition_id),
  compute similarity. This is the closest thing to a ground-truth
  "rewrite of the same paragraph" in paul_data, and we report the
  distribution as a sanity anchor for rewrite similarities.
- Also report distribution on random unrelated (writer_i, writer_j)
  pairs as a lower-bound anchor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC = "embed_cosine_sim_to_base"
SOURCE_DEFAULT = "e5-large-v2"


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _encode_texts(model, texts: list[str], prefix: str = "passage: ") -> np.ndarray:
    # e5 wants a "passage: " / "query: " prefix; for a symmetric "passage-passage"
    # similarity we use "passage: " on both sides. (mpnet doesn't need a prefix;
    # harmless to include so we keep one code path.)
    return model.encode([prefix + t for t in texts], batch_size=32,
                         convert_to_numpy=True, normalize_embeddings=True)


def score_rewrites(model_name: str = SOURCE_DEFAULT,
                   db_path: Path = DEFAULT_DB_PATH,
                   batch_size: int = 64) -> int:
    con = connect(db_path)
    try:
        pairs = pd.read_sql_query("""
            SELECT p.document_id AS rewrite_id, p.text AS rewrite_text,
                   b.text AS base_text
            FROM paragraphs p
            JOIN paragraphs b ON b.document_id = p.base_document_id
            WHERE p.base_document_id IS NOT NULL
        """, con)
        if len(pairs) == 0:
            return 0
        model = _load_model(model_name)
        emb_rw = _encode_texts(model, pairs["rewrite_text"].tolist())
        emb_base = _encode_texts(model, pairs["base_text"].tolist())
        sims = (emb_rw * emb_base).sum(axis=1)
        rows = []
        for sim, (_, r) in zip(sims, pairs.iterrows()):
            rows.append(MetricRow(r["rewrite_id"], METRIC, None, model_name, None,
                                   float(sim), None).to_tuple())
        write_rows(con, rows)
        return len(rows)
    finally:
        con.close()


def benchmark(model_name: str = SOURCE_DEFAULT,
              db_path: Path = DEFAULT_DB_PATH,
              seed: int = 1) -> dict:
    """Compute two anchors: human-edited vs human-original (parallel), and random pairs."""
    con = connect(db_path)
    try:
        # parallel pairs: same writer + proposition, original_writer vs original_edited
        df = pd.read_sql_query("""
            SELECT w.text AS writer_text, e.text AS edited_text, w.writer_id, w.proposition_id
            FROM paragraphs w
            JOIN paragraphs e ON e.writer_id = w.writer_id AND e.proposition_id = w.proposition_id
            WHERE w.origin_kind='original_writer' AND e.origin_kind='original_edited'
        """, con)
        if len(df) == 0:
            return {"error": "no parallel writer/edited pairs found"}
        model = _load_model(model_name)
        emb_w = _encode_texts(model, df["writer_text"].tolist())
        emb_e = _encode_texts(model, df["edited_text"].tolist())
        parallel_sims = (emb_w * emb_e).sum(axis=1)
        # random pairs anchor
        rng = np.random.default_rng(seed)
        n_rand = min(1000, len(df))
        ix1 = rng.integers(0, len(df), n_rand)
        ix2 = rng.integers(0, len(df), n_rand)
        rand_sims = (emb_w[ix1] * emb_w[ix2]).sum(axis=1)
        return {
            "parallel_edited_vs_writer": {
                "n": int(len(df)),
                "mean": float(parallel_sims.mean()),
                "median": float(np.median(parallel_sims)),
                "std": float(parallel_sims.std()),
                "min": float(parallel_sims.min()),
                "max": float(parallel_sims.max()),
            },
            "random_writer_pairs": {
                "n": int(n_rand),
                "mean": float(rand_sims.mean()),
                "median": float(np.median(rand_sims)),
                "std": float(rand_sims.std()),
                "p05": float(np.quantile(rand_sims, 0.05)),
                "p95": float(np.quantile(rand_sims, 0.95)),
            },
            "model_name": model_name,
        }
    finally:
        con.close()


if __name__ == "__main__":
    import json
    print(json.dumps(benchmark(), indent=2))
