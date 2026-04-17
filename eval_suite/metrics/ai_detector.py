"""Metric 6: AI-generated-text probability (Fast-DetectGPT-lite).

Falcon-7B (Binoculars) doesn't fit in available disk (needs ~28GB for
base+instruct). We fall back to the sub-agent's backup recommendation
(Fast-DetectGPT, Bao et al. 2023) in its single-model analytic form
using GPT2-XL (~3GB fp16). It computes the "sampling discrepancy":
for each token t, how far the scored model's log-likelihood of the
actual token deviates from the distribution-expected log-likelihood.
Higher = more likely machine-generated (machines pick near-modal tokens).

Reference implementation:
  github.com/baoguangsheng/fast-detect-gpt (MIT)

We store two metrics per paragraph:
- ai_detector_raw_score  (discrepancy, higher = more AI)
- ai_detector_p_machine  (logistic-regression calibrated on the
                           paul_data writer-vs-model split)

Benchmark: AUROC on the writer (human) vs model (AI) split of paul_data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC_RAW = "ai_detector_raw_score"
METRIC_P_MACHINE = "ai_detector_p_machine"
SOURCE = "fastdetectgpt_gpt2xl"

CALIBRATOR_PATH = Path("/home/max/attack-llm-judge/data/ai_detector_calibrator.json")
DEFAULT_MODEL = "gpt2-xl"


class FastDetectGPT:
    """Single-model analytic Fast-DetectGPT. Higher raw_score = more machine-like."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                  max_length: int = 512):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.model_name = model_name

    @torch.inference_mode()
    def _score_one(self, text: str) -> float | None:
        """Compute the analytic sampling-discrepancy for a single text."""
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=self.max_length, return_token_type_ids=False)
        if enc["input_ids"].size(1) < 8:
            return None
        ids = enc["input_ids"].to(self.device)
        logits = self.model(input_ids=ids).logits  # [1, L, V]
        # Shift: predict token t from position t-1
        logits_score = logits[:, :-1, :]         # [1, L-1, V]
        labels = ids[:, 1:]                       # [1, L-1]
        # For the analytic form: under the model's own distribution, compute
        # mean and variance of the log-likelihood contribution; then the
        # discrepancy is (actual - expected) / sqrt(var).
        lprobs = torch.log_softmax(logits_score.float(), dim=-1)
        probs = torch.softmax(logits_score.float(), dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        mean_ref = (probs * lprobs).sum(dim=-1)
        var_ref = (probs * torch.square(lprobs)).sum(dim=-1) - torch.square(mean_ref)
        actual_sum = log_likelihood.sum(dim=-1)
        expected_sum = mean_ref.sum(dim=-1)
        var_sum = var_ref.sum(dim=-1)
        if var_sum.item() <= 0:
            return None
        discrepancy = (actual_sum - expected_sum) / torch.sqrt(var_sum)
        return float(discrepancy.item())

    def compute_scores(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            try:
                s = self._score_one(t)
            except Exception as e:
                s = None
            out.append(s if s is not None else np.nan)
        return np.array(out, dtype=np.float64)


def fit_calibrator(raw_scores: np.ndarray, labels: np.ndarray,
                    calibrator_path: Path = CALIBRATOR_PATH) -> dict:
    from sklearn.linear_model import LogisticRegression
    mask = np.isfinite(raw_scores)
    X = raw_scores[mask].reshape(-1, 1)  # higher raw = more machine
    y = labels[mask]
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    calib = {
        "intercept": float(clf.intercept_[0]),
        "coef": float(clf.coef_[0][0]),
        "feature": "raw_score",
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
    }
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator_path.write_text(json.dumps(calib, indent=2))
    return calib


def load_calibrator(calibrator_path: Path = CALIBRATOR_PATH) -> dict | None:
    if not calibrator_path.exists():
        return None
    return json.loads(calibrator_path.read_text())


def apply_calibrator(raw: np.ndarray, calib: dict) -> np.ndarray:
    z = calib["intercept"] + calib["coef"] * raw
    return 1.0 / (1.0 + np.exp(-z))


def score_all_and_calibrate(model_name: str = DEFAULT_MODEL,
                              db_path: Path = DEFAULT_DB_PATH,
                              calibrator_path: Path = CALIBRATOR_PATH,
                              where: str | None = None) -> dict:
    con = connect(db_path)
    try:
        sql = "SELECT document_id, text, origin_kind FROM paragraphs"
        if where:
            sql += f" WHERE {where}"
        df = pd.read_sql_query(sql, con)
    finally:
        con.close()
    runner = FastDetectGPT(model_name=model_name)
    raw = runner.compute_scores(df["text"].tolist())
    df = df.assign(raw=raw)

    calib_df = df[df["origin_kind"].isin(["original_writer", "original_model"])].dropna(subset=["raw"])
    labels = (calib_df["origin_kind"] == "original_model").astype(int).to_numpy()
    calib = fit_calibrator(calib_df["raw"].to_numpy(), labels, calibrator_path)
    valid = np.isfinite(df["raw"].to_numpy())
    p_machine = np.where(valid, apply_calibrator(df["raw"].to_numpy(), calib), np.nan)

    rows = []
    for (_, r), raw_v, p in zip(df.iterrows(), df["raw"], p_machine):
        if np.isnan(raw_v):
            continue
        rows.append(MetricRow(r["document_id"], METRIC_RAW, None, SOURCE, None,
                               float(raw_v), None).to_tuple())
        rows.append(MetricRow(r["document_id"], METRIC_P_MACHINE, None, SOURCE, None,
                               float(p), {"calib": calib}).to_tuple())
    con = connect(db_path)
    try:
        write_rows(con, rows)
    finally:
        con.close()
    return {"n_rows_scored": int(len(df)), "n_valid": int(valid.sum()),
            "calibrator": calib, "model": model_name}


def benchmark(db_path: Path = DEFAULT_DB_PATH,
               calibrator_path: Path = CALIBRATOR_PATH) -> dict:
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
    if len(df) < 10:
        return {"error": "insufficient labelled data", "n": int(len(df))}
    y = (df["origin_kind"] == "original_model").astype(int).to_numpy()
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"error": "only one class present"}
    return {
        "n": int(len(df)),
        "n_human": int((y == 0).sum()),
        "n_machine": int((y == 1).sum()),
        "auroc_raw": float(roc_auc_score(y, df["raw"].to_numpy())),
        "auroc_p_machine": float(roc_auc_score(y, df["p_machine"].to_numpy())),
        "calibrator": load_calibrator(calibrator_path),
        "source": SOURCE,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["score", "benchmark"], default="score")
    p.add_argument("--where", default=None)
    args = p.parse_args()
    if args.mode == "score":
        print(json.dumps(score_all_and_calibrate(where=args.where), indent=2))
    else:
        print(json.dumps(benchmark(), indent=2))
