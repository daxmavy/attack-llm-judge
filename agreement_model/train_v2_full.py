"""Train the FINAL agreement_score regressor on the full dataset.

Same validation strategy as train_v2.py (5 controversial + 5 non-controversial
propositions held out for early-stopping / best-checkpoint selection) but NO
TEST HOLDOUTS — every other paragraph trains.

Use case: this is the model that gets applied to attack_rewrites for downstream
analysis. The earlier train_v2 / train_v2_humanonly runs were diagnostic
(quantifying generalisation under various distribution shifts); now we use all
the labelled data we have.

Includes ALL origin_kinds (writer + model + edited).

Predictions written with model_run_id 'agreement_v2_full_<ts>' to
attack_agreement_score_predictions_v2 — only train + val for this run, since
there is no test split. After training, apply_agreement_score_v2.py uses the
saved model from /workspace/agreement_model_v2_full/final to score
attack_rewrites.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATA_CSV = "/home/max/attack-llm-judge/paul_data/prepared/documents.csv"
DB_PATH = "/home/max/attack-llm-judge/data/paragraphs.db"
CONTR_JSON = "/workspace/grpo_run/controversial_40_3fold.json"
DEFAULT_MODEL = "microsoft/deberta-v3-base"


class AgreementDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.propositions = df["proposition"].tolist()
        self.documents = df["document_text"].tolist()
        self.labels = df["agreement_score"].astype("float32").tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.propositions[idx], self.documents[idx],
                              truncation=True, max_length=self.max_length, padding=False)
        enc["labels"] = float(self.labels[idx])
        return enc


def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).squeeze().astype(np.float64)
    labels = np.asarray(labels).squeeze().astype(np.float64)
    fallback = float(np.nanmean(labels)) if np.isfinite(labels).any() else 0.5
    preds = np.where(np.isfinite(preds), preds, fallback)
    preds_clipped = np.clip(preds, 0.0, 1.0)
    pearson = float(pearsonr(preds, labels)[0]) if np.std(preds) > 0 else 0.0
    spearman = float(spearmanr(preds, labels)[0]) if np.std(preds) > 0 else 0.0
    return {
        "mae": float(mean_absolute_error(labels, preds_clipped)),
        "rmse": float(np.sqrt(mean_squared_error(labels, preds_clipped))),
        "pearson": pearson, "spearman": spearman,
    }


def make_splits_full(df, controversial_pids, seed):
    """Train = ALL paragraphs except those in val propositions.
    Val = same 5+5 propositions as train_v2.py for fair MAE comparison.
    """
    rng = random.Random(seed)

    df = df.copy()
    df["origin_kind"] = df["paragraph_type"].map({
        "writer": "original_writer", "model": "original_model", "edited": "original_edited",
    })
    df["is_controversial"] = df["proposition_id"].isin(controversial_pids).astype(int)

    all_pids = df["proposition_id"].unique().tolist()
    contr_pids = sorted(p for p in all_pids if p in controversial_pids)
    nonctr_pids = sorted(p for p in all_pids if p not in controversial_pids)

    val_contr = rng.sample(contr_pids, 5)
    val_nonctr = rng.sample(nonctr_pids, 5)
    val_pids = set(val_contr + val_nonctr)

    train_df = df[~df["proposition_id"].isin(val_pids)].reset_index(drop=True)
    val_df = df[df["proposition_id"].isin(val_pids)].reset_index(drop=True)

    print(f"  total props: {len(all_pids)}  controversial: {len(contr_pids)}  non-controversial: {len(nonctr_pids)}", flush=True)
    print(f"  val props: {len(val_pids)} ({len(val_contr)} contr + {len(val_nonctr)} nonctr)", flush=True)
    print(f"  paragraph counts: train={len(train_df)} val={len(val_df)}  sum={len(train_df)+len(val_df)}", flush=True)

    return train_df, val_df, val_pids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="/workspace/agreement_model_v2_full")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df_all = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df_all)} rows", flush=True)
    contr_pids = {p["pid"] for p in json.load(open(CONTR_JSON))["propositions"]}

    train_df, val_df, val_pids = make_splits_full(df_all, contr_pids, args.seed)

    for d in (train_df, val_df):
        d["is_human"] = (d["origin_kind"] == "original_writer").astype(int)

    print("\nTrain composition (origin_kind × is_controversial):", flush=True)
    print(train_df.groupby(["origin_kind","is_controversial"]).size().to_string(), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression",
        use_safetensors=True, torch_dtype=torch.float32,
    )
    model.to(torch.float32)

    train_ds = AgreementDataset(train_df, tokenizer, args.max_length)
    val_ds = AgreementDataset(val_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="steps", logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False,
        save_total_limit=1, report_to="none", dataloader_num_workers=2, seed=args.seed,
    )
    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    print(f"\nVal: {val_metrics}", flush=True)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nmodel saved to {final_dir}", flush=True)

    meta = {
        "run_id": f"agreement_v2_full_{int(time.time())}",
        "args": vars(args),
        "val_metrics": val_metrics,
        "splits": {"val_pids": sorted(val_pids),
                   "train_n": int(len(train_df)), "val_n": int(len(val_df))},
    }
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nmeta saved to {args.output_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
