"""Metric 6: AI-generated-text probability via Binoculars.

Binoculars (Hans et al. ICML 2024): ratio of log-perplexity under a base
model to cross-perplexity under an instruct model, with the two models
from the same family (Falcon-7B / Falcon-7B-Instruct). Lower raw score
= more machine-like.

We do NOT use the author-shipped global threshold. We store both the raw
score AND a calibrated P(machine) that is fit on paul_data's labelled
human (writer) vs. AI (model) paragraphs via a logistic regression.

Benchmark: AUROC of raw score and calibrated P(machine) at
distinguishing original_writer vs. original_model on paul_data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC_RAW = "ai_detector_raw_score"
METRIC_P_MACHINE = "ai_detector_p_machine"
SOURCE = "binoculars_falcon7b"

CALIBRATOR_PATH = Path("/home/max/attack-llm-judge/data/binoculars_calibrator.json")


class BinocularsRunner:
    def __init__(self, batch_size: int = 32):
        from binoculars import Binoculars
        self.bino = Binoculars()
        self.batch_size = batch_size

    def compute_raw(self, texts: list[str]) -> np.ndarray:
        # Binoculars.compute_score handles batching internally; we chunk to
        # keep VRAM stable on a shared A100.
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            scores = self.bino.compute_score(chunk)
            out.extend(list(scores))
        return np.array(out, dtype=np.float64)


def fit_calibrator(raw_scores: np.ndarray, labels: np.ndarray,
                    calibrator_path: Path = CALIBRATOR_PATH) -> dict:
    """Fit a univariate logistic regression P(machine=1 | raw_score)."""
    from sklearn.linear_model import LogisticRegression
    # Lower raw = more machine, so we feed -raw as the feature so positive coef
    # means "more likely machine" (not strictly necessary for logistic, but
    # keeps the stored weights human-readable).
    X = (-raw_scores).reshape(-1, 1)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, labels)
    calib = {
        "intercept": float(clf.intercept_[0]),
        "coef": float(clf.coef_[0][0]),
        "feature": "neg_raw_score",
        "n_human": int((labels == 0).sum()),
        "n_machine": int((labels == 1).sum()),
    }
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator_path.write_text(json.dumps(calib, indent=2))
    return calib


def load_calibrator(calibrator_path: Path = CALIBRATOR_PATH) -> dict | None:
    if not calibrator_path.exists():
        return None
    return json.loads(calibrator_path.read_text())


def apply_calibrator(raw: np.ndarray, calib: dict) -> np.ndarray:
    x = -raw
    z = calib["intercept"] + calib["coef"] * x
    return 1.0 / (1.0 + np.exp(-z))


def score_all_and_calibrate(db_path: Path = DEFAULT_DB_PATH,
                              calibrator_path: Path = CALIBRATOR_PATH) -> dict:
    """Score everything in the paragraphs table. Also fits the calibrator
    using original_writer (label 0) vs original_model (label 1) and
    applies it to produce P(machine) for every row."""
    con = connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT document_id, text, origin_kind FROM paragraphs", con)
    finally:
        con.close()
    runner = BinocularsRunner(batch_size=32)
    raw = runner.compute_raw(df["text"].tolist())
    df = df.assign(raw=raw)

    # Fit calibrator on writer vs model.
    calib_df = df[df["origin_kind"].isin(["original_writer", "original_model"])]
    labels = (calib_df["origin_kind"] == "original_model").astype(int).to_numpy()
    calib = fit_calibrator(calib_df["raw"].to_numpy(), labels, calibrator_path)
    p_machine = apply_calibrator(df["raw"].to_numpy(), calib)

    rows = []
    for (_, r), raw_v, p in zip(df.iterrows(), df["raw"], p_machine):
        rows.append(MetricRow(r["document_id"], METRIC_RAW, None, SOURCE, None,
                               float(raw_v), None).to_tuple())
        rows.append(MetricRow(r["document_id"], METRIC_P_MACHINE, None, SOURCE, None,
                               float(p), {"calib": calib}).to_tuple())
    con = connect(db_path)
    try:
        write_rows(con, rows)
    finally:
        con.close()
    return {"n_rows_scored": int(len(df)), "calibrator": calib}


def benchmark(db_path: Path = DEFAULT_DB_PATH,
               calibrator_path: Path = CALIBRATOR_PATH) -> dict:
    """Compute AUROC of raw score and calibrated P(machine) on
    original_writer (label 0) vs original_model (label 1)."""
    from sklearn.metrics import roc_auc_score
    con = connect(db_path)
    try:
        df = pd.read_sql_query(f"""
            SELECT p.document_id, p.origin_kind,
                   er.value AS raw, em.value AS p_machine
            FROM paragraphs p
            LEFT JOIN evaluations er ON er.paragraph_id=p.document_id
                AND er.metric='{METRIC_RAW}' AND er.source='{SOURCE}'
            LEFT JOIN evaluations em ON em.paragraph_id=p.document_id
                AND em.metric='{METRIC_P_MACHINE}' AND em.source='{SOURCE}'
            WHERE p.origin_kind IN ('original_writer','original_model')
        """, con)
    finally:
        con.close()
    df = df.dropna(subset=["raw", "p_machine"])
    y = (df["origin_kind"] == "original_model").astype(int).to_numpy()
    if len(df) < 10 or y.sum() == 0 or (1 - y).sum() == 0:
        return {"error": "insufficient labelled data", "n": int(len(df))}
    return {
        "n": int(len(df)),
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
        # Lower raw = more machine, so we pass -raw for ROC direction.
        "auroc_raw": float(roc_auc_score(y, -df["raw"].to_numpy())),
        "auroc_p_machine": float(roc_auc_score(y, df["p_machine"].to_numpy())),
        "calibrator": load_calibrator(calibrator_path),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["score", "benchmark"], default="score")
    args = p.parse_args()
    if args.mode == "score":
        print(json.dumps(score_all_and_calibrate(), indent=2))
    else:
        print(json.dumps(benchmark(), indent=2))
