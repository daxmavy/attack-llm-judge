"""Supervised AI-generated-text detector (metric 6 - revised).

Fast-DetectGPT-style zero-shot methods need a scoring model of similar
capability to the AIs being detected. Our AI paragraphs come from
Claude Sonnet 4 / DeepSeek V3 / GPT-4o-latest; no 1-3B scoring model
(what our disk fits) comes close. Qwen-2.5-1.5B Fast-DetectGPT gave
AUROC 0.57 — too weak to be meaningful.

Supervised alternative: train a logistic regression on TF-IDF features
of the 4503 human + 4503 AI paragraphs in paul_data. This is a much
better fit for this specific corpus (it learns the stylistic signature
of the three specific generating models), at the cost of being
corpus-specific rather than generally-applicable. Trade-off is the
right call here.

Benchmark: held-out (group-by-proposition) AUROC.

Stores:
- ai_detector_p_machine   (calibrated probability from the classifier)
- ai_detector_raw_score   (logit of the above, for consistency with the
                            Fast-DetectGPT version — callers can use
                            whichever)

Both under source="tfidf_logreg_paul_split".
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from eval_suite.metrics.base import MetricRow, write_rows
from eval_suite.schema import DEFAULT_DB_PATH, connect


METRIC_RAW = "ai_detector_raw_score"
METRIC_P_MACHINE = "ai_detector_p_machine"
SOURCE = "tfidf_logreg_paul_split"

MODEL_PATH = Path("/home/max/attack-llm-judge/data/ai_detector_tfidf.pkl")


def train(db_path: Path = DEFAULT_DB_PATH, seed: int = 17) -> dict:
    con = connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT document_id, proposition_id, origin_kind, text
            FROM paragraphs
            WHERE origin_kind IN ('original_writer', 'original_model')
        """, con)
    finally:
        con.close()
    # Label: 1 = machine (model), 0 = human (writer).
    df["label"] = (df["origin_kind"] == "original_model").astype(int)
    # Group-by-proposition split 80/20 so train and test cover different prompts.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx_train, idx_test = next(gss.split(df, groups=df["proposition_id"]))
    train_df = df.iloc[idx_train].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    # Features: word-level TF-IDF (1-2 grams) + char 3-5 gram TF-IDF.
    # Char n-grams catch stylistic signatures (punctuation spacing, em-dashes,
    # sentence-length regularity) that models consistently produce.
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=30000, sublinear_tf=True)
    X_train_w = word_vec.fit_transform(train_df["text"])
    X_train_c = char_vec.fit_transform(train_df["text"])
    X_train = hstack([X_train_w, X_train_c]).tocsr()
    X_test_w = word_vec.transform(test_df["text"])
    X_test_c = char_vec.transform(test_df["text"])
    X_test = hstack([X_test_w, X_test_c]).tocsr()

    clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1)
    clf.fit(X_train, train_df["label"])

    def stats(X, y_true):
        probs = clf.predict_proba(X)[:, 1]
        return {"n": int(len(y_true)), "auroc": float(roc_auc_score(y_true, probs))}

    metrics = {
        "train": stats(X_train, train_df["label"]),
        "test":  stats(X_test, test_df["label"]),
    }
    # Detailed test split stats.
    test_df["p_machine"] = clf.predict_proba(X_test)[:, 1]
    by_model = {}
    con = connect(db_path)
    try:
        mn = pd.read_sql_query(
            "SELECT document_id, paul_data_model_name FROM paragraphs WHERE origin_kind='original_model'",
            con)
    finally:
        con.close()
    enriched = test_df.merge(mn, on="document_id", how="left")
    for m, g in enriched.groupby("paul_data_model_name"):
        if len(g) < 5: continue
        # Build per-model AUROC using g vs all writers in test.
        writers_in_test = enriched[enriched["origin_kind"] == "original_writer"]
        if len(writers_in_test) < 5: continue
        y = np.concatenate([g["label"].values, writers_in_test["label"].values])
        p = np.concatenate([g["p_machine"].values, writers_in_test["p_machine"].values])
        by_model[str(m)] = {"n_model": int(len(g)), "n_human": int(len(writers_in_test)),
                             "auroc": float(roc_auc_score(y, p))}
    metrics["test_by_model_name"] = by_model

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf,
                      "seed": seed, "metrics": metrics}, f)
    return metrics


def _load():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def predict_proba(texts: list[str]) -> np.ndarray:
    bundle = _load()
    Xw = bundle["word_vec"].transform(texts)
    Xc = bundle["char_vec"].transform(texts)
    X = hstack([Xw, Xc]).tocsr()
    return bundle["clf"].predict_proba(X)[:, 1]


def score_all(db_path: Path = DEFAULT_DB_PATH, where: str | None = None) -> int:
    con = connect(db_path)
    try:
        sql = "SELECT document_id, text FROM paragraphs"
        if where:
            sql += f" WHERE {where}"
        df = pd.read_sql_query(sql, con)
    finally:
        con.close()
    if len(df) == 0: return 0
    p = predict_proba(df["text"].tolist())
    # logit of p as raw_score
    eps = 1e-7
    raw = np.log((p + eps) / (1 - p + eps))
    rows = []
    for (_, r), prob, rv in zip(df.iterrows(), p, raw):
        rows.append(MetricRow(r["document_id"], METRIC_P_MACHINE, None, SOURCE, None,
                               float(prob), None).to_tuple())
        rows.append(MetricRow(r["document_id"], METRIC_RAW, None, SOURCE, None,
                               float(rv), None).to_tuple())
    con = connect(db_path)
    try:
        write_rows(con, rows)
    finally:
        con.close()
    return len(rows)


def benchmark(db_path: Path = DEFAULT_DB_PATH) -> dict:
    con = connect(db_path)
    try:
        df = pd.read_sql_query(f"""
            SELECT p.document_id, p.origin_kind,
                   em.value AS p_machine
            FROM paragraphs p
            LEFT JOIN evaluations em ON em.paragraph_id=p.document_id
                AND em.metric='{METRIC_P_MACHINE}' AND em.source='{SOURCE}'
            WHERE p.origin_kind IN ('original_writer','original_model','original_edited')
        """, con)
    finally:
        con.close()
    df = df.dropna(subset=["p_machine"])
    out: dict = {"source": SOURCE, "overall": {}, "by_origin_kind": {}}
    wk = df[df["origin_kind"].isin(["original_writer", "original_model"])]
    y = (wk["origin_kind"] == "original_model").astype(int).to_numpy()
    if y.sum() > 0 and (1 - y).sum() > 0:
        out["overall"] = {
            "n": int(len(wk)),
            "auroc_writer_vs_model": float(roc_auc_score(y, wk["p_machine"].to_numpy())),
            "mean_p_machine_writer": float(wk[wk["origin_kind"]=="original_writer"]["p_machine"].mean()),
            "mean_p_machine_model":  float(wk[wk["origin_kind"]=="original_model"]["p_machine"].mean()),
        }
    for ok, g in df.groupby("origin_kind"):
        out["by_origin_kind"][ok] = {
            "n": int(len(g)),
            "mean_p_machine": float(g["p_machine"].mean()),
            "median_p_machine": float(g["p_machine"].median()),
        }
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "score", "benchmark"], default="train")
    p.add_argument("--where", default=None)
    args = p.parse_args()
    if args.mode == "train":
        print(json.dumps(train(), indent=2))
    elif args.mode == "score":
        n = score_all(where=args.where)
        print(json.dumps({"rows_written": n}, indent=2))
    else:
        print(json.dumps(benchmark(), indent=2))
