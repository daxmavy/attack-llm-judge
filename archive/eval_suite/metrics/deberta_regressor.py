"""Inference wrapper for DeBERTa-style regressors.

Shared by:
- agreement-score regressor (metric 4) — weights at
  agreement_model/runs/main/final/
- clarity criterion regressor (metric 5) — trained in
  criterion_model/clarity/final/ by criterion_model/train_clarity.py.

Both take (proposition, paragraph) -> scalar in [0,1] (agreement_score) or
[0,100] (clarity). Kept separate at the metric layer so callers can
swap criteria.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


class DebertaRegressor:
    def __init__(self, model_dir: Path, max_length: int = 256, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, propositions: list[str], paragraphs: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(paragraphs), batch_size):
            props = propositions[i:i + batch_size]
            paras = paragraphs[i:i + batch_size]
            enc = self.tokenizer(props, paras, truncation=True, padding=True,
                                  max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            out.append(logits.detach().cpu().numpy())
        return np.concatenate(out)


# ---- agreement_model wrapper (metric 4) ----
AGREEMENT_DEFAULT_DIR = Path("/home/max/attack-llm-judge/agreement_model/runs/main/final")
METRIC_AGREEMENT = "agreement_score_pred"
SOURCE_AGREEMENT = "agreement_model_deberta_v3_base"


def score_agreement(model_dir: Path = AGREEMENT_DEFAULT_DIR,
                    db_path: Path = DEFAULT_DB_PATH,
                    where: str | None = None) -> int:
    con = connect(db_path)
    try:
        sql = "SELECT document_id, proposition, text FROM paragraphs"
        if where:
            sql += f" WHERE {where}"
        df = pd.read_sql_query(sql, con)
        if len(df) == 0:
            return 0
        reg = DebertaRegressor(model_dir)
        preds = reg.predict(df["proposition"].tolist(), df["text"].tolist())
        # agreement regressor was trained in [0,1]; clip.
        preds = np.clip(preds, 0.0, 1.0)
        rows = [MetricRow(r["document_id"], METRIC_AGREEMENT, None, SOURCE_AGREEMENT, None,
                          float(p), None).to_tuple()
                for p, (_, r) in zip(preds, df.iterrows())]
        write_rows(con, rows)
        return len(rows)
    finally:
        con.close()


def benchmark_agreement(model_dir: Path = AGREEMENT_DEFAULT_DIR,
                         db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Run the agreement regressor on all paul_data originals and compute
    Pearson/Spearman/MAE vs the stored human agreement_score."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error
    con = connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT document_id, proposition, text, origin_kind, paul_data_model_name,
                   human_agreement_score
            FROM paragraphs
            WHERE origin_kind IN ('original_writer','original_model','original_edited')
              AND human_agreement_score IS NOT NULL
        """, con)
    finally:
        con.close()
    reg = DebertaRegressor(model_dir)
    preds = np.clip(reg.predict(df["proposition"].tolist(), df["text"].tolist()), 0.0, 1.0)
    df = df.assign(pred=preds)
    out = {"overall": _reg_stats(df)}
    for g, sub in df.groupby("origin_kind"):
        out[g] = _reg_stats(sub)
    return out


# ---- clarity regressor wrapper (metric 5) ----
CLARITY_DEFAULT_DIR = Path("/home/max/attack-llm-judge/criterion_model/clarity/final")
METRIC_CLARITY = "clarity_score_pred"
SOURCE_CLARITY = "clarity_regressor_deberta_v3_base"


def score_clarity(model_dir: Path = CLARITY_DEFAULT_DIR,
                  db_path: Path = DEFAULT_DB_PATH,
                  where: str | None = None) -> int:
    if not model_dir.exists():
        print(f"[clarity] model not found at {model_dir}; train first via "
              "criterion_model.train_clarity")
        return 0
    con = connect(db_path)
    try:
        sql = "SELECT document_id, proposition, text FROM paragraphs"
        if where:
            sql += f" WHERE {where}"
        df = pd.read_sql_query(sql, con)
        if len(df) == 0:
            return 0
        reg = DebertaRegressor(model_dir)
        preds = reg.predict(df["proposition"].tolist(), df["text"].tolist())
        # clarity regressor targets [0,100]; clip.
        preds = np.clip(preds, 0.0, 100.0)
        rows = [MetricRow(r["document_id"], METRIC_CLARITY, "clarity", SOURCE_CLARITY, None,
                          float(p), None).to_tuple()
                for p, (_, r) in zip(preds, df.iterrows())]
        write_rows(con, rows)
        return len(rows)
    finally:
        con.close()


def benchmark_clarity(model_dir: Path = CLARITY_DEFAULT_DIR,
                       db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Pearson/Spearman/MAE vs human_mean_clarity on originals, overall and by origin_kind."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error
    if not model_dir.exists():
        return {"error": f"{model_dir} not found"}
    con = connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT document_id, proposition, text, origin_kind, human_mean_clarity
            FROM paragraphs
            WHERE human_mean_clarity IS NOT NULL
        """, con)
    finally:
        con.close()
    reg = DebertaRegressor(model_dir)
    preds = np.clip(reg.predict(df["proposition"].tolist(), df["text"].tolist()), 0, 100)
    df = df.assign(pred=preds, human=df["human_mean_clarity"])
    out = {"overall": _reg_stats(df, human_col="human")}
    for g, sub in df.groupby("origin_kind"):
        out[g] = _reg_stats(sub, human_col="human")
    return out


def _reg_stats(df: pd.DataFrame, pred_col: str = "pred",
               human_col: str = "human_agreement_score") -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error
    if len(df) < 5:
        return {"n": int(len(df))}
    a = df[pred_col].to_numpy()
    b = df[human_col].to_numpy() if human_col in df.columns else df["human"].to_numpy()
    return {
        "n": int(len(df)),
        "pearson": float(pearsonr(a, b)[0]),
        "spearman": float(spearmanr(a, b).correlation),
        "mae": float(mean_absolute_error(b, a)),
    }
