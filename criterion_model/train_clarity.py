"""Train a DeBERTa regressor for (proposition, paragraph) -> mean_human_clarity.

Mirrors agreement_model/train.py so the two are directly comparable.
Target is the per-paragraph mean of rater-level paragraph_clarity scores
from paul_data/main_phase_2/annotations.csv. Score range [0, 100]; we
divide by 100 at train-time and multiply back at inference so the
objective lives in [0, 1] like the agreement regressor.

Split: group-by-proposition 80/10/10 so val/test propositions are unseen.
Reports overall + subgroup (paragraph_type, paul_data model_name) stats
on the test set, the numbers the operator asked for.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DOCS_CSV = "/home/max/attack-llm-judge/paul_data/prepared/documents.csv"
ANN_CSV = "/home/max/attack-llm-judge/paul_data/main_phase_2/annotations.csv"
DEFAULT_MODEL = "microsoft/deberta-v3-base"


def load_data() -> pd.DataFrame:
    docs = pd.read_csv(DOCS_CSV)
    ann = pd.read_csv(ANN_CSV)
    key = ["writer_id", "proposition_id", "paragraph_type", "model_name", "model_input_condition"]
    agg = ann.groupby(key, dropna=False).agg(
        n_ratings=("paragraph_clarity", "size"),
        mean_clarity=("paragraph_clarity", "mean"),
    ).reset_index()
    df = docs.merge(agg, on=key, how="inner")
    df = df.dropna(subset=["mean_clarity"]).reset_index(drop=True)
    return df


class CritDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.props = df["proposition"].tolist()
        self.paras = df["document_text"].tolist()
        # target in [0,1] for training stability; clamp.
        self.y = (df["mean_clarity"] / 100.0).clip(0, 1).astype("float32").tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.props[idx], self.paras[idx], truncation=True,
                              max_length=self.max_length, padding=False)
        enc["labels"] = float(self.y[idx])
        return enc


def metrics_fn(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).squeeze().astype(np.float64)
    labels = np.asarray(labels).squeeze().astype(np.float64)
    fallback = float(np.nanmean(labels)) if np.isfinite(labels).any() else 0.5
    preds = np.where(np.isfinite(preds), preds, fallback)
    preds_clipped = np.clip(preds, 0.0, 1.0)
    return {
        "mae": float(mean_absolute_error(labels, preds_clipped)),
        "rmse": float(np.sqrt(mean_squared_error(labels, preds_clipped))),
        "pearson": float(pearsonr(preds, labels)[0]) if np.std(preds) > 0 else 0.0,
        "spearman": float(spearmanr(preds, labels).correlation) if np.std(preds) > 0 else 0.0,
    }


def group_split(df: pd.DataFrame, seed: int = 42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx_train, idx_rest = next(gss.split(df, groups=df["proposition_id"]))
    rest = df.iloc[idx_rest].reset_index(drop=True)
    # split rest 50/50 into val/test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    idx_val, idx_test = next(gss2.split(rest, groups=rest["proposition_id"]))
    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = rest.iloc[idx_val].reset_index(drop=True)
    test_df = rest.iloc[idx_test].reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate_test(trainer: Trainer, test_ds: CritDataset, test_df: pd.DataFrame) -> dict:
    preds = trainer.predict(test_ds).predictions.squeeze()
    preds = np.clip(preds, 0.0, 1.0)
    y = np.array(test_ds.y)
    overall = {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "pearson": float(pearsonr(preds, y)[0]),
        "spearman": float(spearmanr(preds, y).correlation),
        "n": int(len(y)),
    }
    # subgroup stats
    test_df = test_df.copy()
    test_df["pred"] = preds
    test_df["y"] = y
    def _stats(sub):
        if len(sub) < 5:
            return {"n": int(len(sub))}
        return {
            "n": int(len(sub)),
            "mae": float(mean_absolute_error(sub["y"], sub["pred"])),
            "pearson": float(pearsonr(sub["pred"], sub["y"])[0]),
            "spearman": float(spearmanr(sub["pred"], sub["y"]).correlation),
        }
    by_type = {str(k): _stats(sub) for k, sub in test_df.groupby("paragraph_type")}
    by_model = {str(k): _stats(sub) for k, sub in test_df.groupby("model_name", dropna=False)}
    return {"overall": overall, "by_paragraph_type": by_type, "by_model_name": by_model}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="/home/max/attack-llm-judge/criterion_model/clarity")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    df = load_data()
    train_df, val_df, test_df = group_split(df, args.seed)
    print(f"sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression",
    )

    train_ds = CritDataset(train_df, tok, args.max_length)
    val_ds = CritDataset(val_df, tok, args.max_length)
    test_ds = CritDataset(test_df, tok, args.max_length)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn,
    )
    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = evaluate_test(trainer, test_ds, test_df)
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    report = {
        "val": val_metrics,
        "test": test_metrics,
        "args": vars(args),
        "target": "mean_human_clarity / 100",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
