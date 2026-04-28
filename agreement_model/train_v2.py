"""Train agreement_score regressor v2 with new hold-out strategy.

Splits (seed=42, fixed):
  - VAL: 5 random controversial + 5 random non-controversial propositions
         (entirely held out at proposition level)
  - TEST_PROP_OOD: 10% of the remaining non-controversial propositions
                   (entirely held out at proposition level)
  - For all other propositions (35 controversial + 49 non-controversial):
      For each (proposition × origin_kind), 10% of paragraphs → TEST_PARA_OOD,
      remaining 90% → TRAIN
  - Stratified by origin_kind so test_para_ood has matching writer/model/edited
    proportions per proposition.

Model recipe: same as previous (DeBERTa-v3-base, 4 epochs, lr=1e-5, batch=32,
max_length=256). Best-checkpoint selection on val MAE.

After training, predicts on train + val + test_para_ood + test_prop_ood,
writes to a NEW table attack_agreement_score_predictions_v2 (does not touch
attack_agreement_scores from the v1 model). Reports MAE and stance-flip
rates split by:
  - split
  - is_controversial × split
  - is_human × split
  - polar (true∈[<0.25 ∪ >0.75]) × split  (denom for stance-flip rate)

Stance-flip = (true<0.25 and pred>0.75) or (true>0.75 and pred<0.25); the rate
is reported only over paragraphs where the human-label ground truth itself is
in the polar regions ([<0.25] ∪ [>0.75]).

Method: 'agreement_v2' (model_run_id = 'agreement_v2_<timestamp>')
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from collections import defaultdict
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
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.propositions = df["proposition"].tolist()
        self.documents = df["document_text"].tolist()
        self.labels = df["agreement_score"].astype("float32").tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.propositions[idx], self.documents[idx],
            truncation=True, max_length=self.max_length, padding=False,
        )
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
        "pearson": pearson,
        "spearman": spearman,
    }


def make_splits(df: pd.DataFrame, controversial_pids: set, seed: int):
    """Produce train / val / test_para_ood / test_prop_ood DataFrames.

    Validation: 5 controversial + 5 non-controversial random propositions.
    Test (prop OOD): 10% of remaining non-controversial propositions.
    Remaining propositions: 90/10 paragraph split per (prop × origin_kind).
    """
    rng = random.Random(seed)

    # documents.csv has paragraph_type {writer, model, edited}; map to origin_kind
    df = df.copy()
    df["origin_kind"] = df["paragraph_type"].map({
        "writer": "original_writer",
        "model": "original_model",
        "edited": "original_edited",
    })
    assert df["origin_kind"].notna().all(), "unmapped paragraph_type"

    df["is_controversial"] = df["proposition_id"].isin(controversial_pids).astype(int)

    all_pids = df["proposition_id"].unique().tolist()
    contr_pids = sorted(p for p in all_pids if p in controversial_pids)
    nonctr_pids = sorted(p for p in all_pids if p not in controversial_pids)
    print(f"  total props: {len(all_pids)}  controversial: {len(contr_pids)}  non-controversial: {len(nonctr_pids)}",
          flush=True)

    # 5 controversial + 5 non-controversial → val
    val_contr = rng.sample(contr_pids, 5)
    val_nonctr = rng.sample(nonctr_pids, 5)
    val_pids = set(val_contr + val_nonctr)
    remaining_nonctr = [p for p in nonctr_pids if p not in val_pids]

    # 10% of remaining non-controversial → test_prop_ood
    n_test_prop = max(1, int(round(0.10 * len(remaining_nonctr))))
    test_prop_pids = set(rng.sample(remaining_nonctr, n_test_prop))

    train_props = [p for p in all_pids if p not in val_pids and p not in test_prop_pids]
    print(f"  val props: {len(val_pids)} ({len(val_contr)} contr + {len(val_nonctr)} nonctr)", flush=True)
    print(f"  test_prop_ood props: {len(test_prop_pids)} (all non-controversial)", flush=True)
    print(f"  train props: {len(train_props)}", flush=True)

    # Per-proposition × origin_kind paragraph split for train props
    train_indices = []
    test_para_indices = []
    for pid in train_props:
        sub = df[df["proposition_id"] == pid]
        for ok, grp in sub.groupby("origin_kind"):
            n = len(grp)
            n_test = max(1, int(round(0.10 * n))) if n >= 10 else max(0, n // 10)
            shuffled = grp.sample(frac=1.0, random_state=seed + pid).index.tolist()
            test_para_indices.extend(shuffled[:n_test])
            train_indices.extend(shuffled[n_test:])

    val_indices = df[df["proposition_id"].isin(val_pids)].index.tolist()
    test_prop_indices = df[df["proposition_id"].isin(test_prop_pids)].index.tolist()

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_para_df = df.loc[test_para_indices].reset_index(drop=True)
    test_prop_df = df.loc[test_prop_indices].reset_index(drop=True)

    # Sanity: disjoint
    seen = set()
    for label, d in [("train", train_df), ("val", val_df),
                     ("test_para_ood", test_para_df), ("test_prop_ood", test_prop_df)]:
        ids = set(d["document_id"].tolist())
        overlap = ids & seen
        assert not overlap, f"{label} overlaps prior splits: {len(overlap)}"
        seen.update(ids)
    print(f"  paragraph counts: train={len(train_df)} val={len(val_df)} "
          f"test_para_ood={len(test_para_df)} test_prop_ood={len(test_prop_df)}  "
          f"sum={len(train_df)+len(val_df)+len(test_para_df)+len(test_prop_df)} (full={len(df)})", flush=True)

    return train_df, val_df, test_para_df, test_prop_df, val_pids, test_prop_pids


def stance_flip_rate(true, pred):
    """Polar threshold: true must be <0.25 or >0.75. Flip = goes to opposite polar."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    polar_mask = (true < 0.25) | (true > 0.75)
    n_polar = int(polar_mask.sum())
    if n_polar == 0:
        return 0.0, 0, 0
    flips = ((true < 0.25) & (pred > 0.75)) | ((true > 0.75) & (pred < 0.25))
    n_flips = int((flips & polar_mask).sum())
    return n_flips / n_polar, n_flips, n_polar


def report_splits(predictions_df: pd.DataFrame):
    """Print MAE and stance-flip rates split by various dimensions."""
    print("\n=== overall metrics by split ===", flush=True)
    for split, sub in predictions_df.groupby("split"):
        mae = mean_absolute_error(sub["true_score"], sub["predicted_score"])
        sf_rate, n_flips, n_polar = stance_flip_rate(sub["true_score"], sub["predicted_score"])
        print(f"  {split:<18} n={len(sub):<6}  MAE={mae:.4f}  "
              f"stance_flip={sf_rate:.4f} ({n_flips}/{n_polar} polar)", flush=True)

    print("\n=== by is_controversial × split ===", flush=True)
    for (controv, split), sub in predictions_df.groupby(["is_controversial", "split"]):
        if len(sub) == 0:
            continue
        mae = mean_absolute_error(sub["true_score"], sub["predicted_score"])
        sf_rate, n_flips, n_polar = stance_flip_rate(sub["true_score"], sub["predicted_score"])
        ctag = "controversial" if controv else "non-controversial"
        print(f"  {ctag:<18} {split:<18} n={len(sub):<6}  MAE={mae:.4f}  "
              f"stance_flip={sf_rate:.4f} ({n_flips}/{n_polar} polar)", flush=True)

    print("\n=== by is_human × split ===", flush=True)
    for (human, split), sub in predictions_df.groupby(["is_human", "split"]):
        if len(sub) == 0:
            continue
        mae = mean_absolute_error(sub["true_score"], sub["predicted_score"])
        sf_rate, n_flips, n_polar = stance_flip_rate(sub["true_score"], sub["predicted_score"])
        htag = "human" if human else "non-human"
        print(f"  {htag:<18} {split:<18} n={len(sub):<6}  MAE={mae:.4f}  "
              f"stance_flip={sf_rate:.4f} ({n_flips}/{n_polar} polar)", flush=True)


def predict_on(trainer, df, tokenizer, max_length, split_label):
    if len(df) == 0:
        return np.zeros(0)
    ds = AgreementDataset(df, tokenizer, max_length)
    out = trainer.predict(ds)
    return np.clip(out.predictions.squeeze().astype(np.float64), 0.0, 1.0)


def write_predictions_table(conn, run_id: str, predictions_df: pd.DataFrame):
    """Write predictions to attack_agreement_score_predictions_v2."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attack_agreement_score_predictions_v2 (
            paragraph_id TEXT NOT NULL,
            proposition_id INTEGER NOT NULL,
            is_controversial INTEGER NOT NULL,
            split TEXT NOT NULL,
            is_human INTEGER NOT NULL,
            origin_kind TEXT NOT NULL,
            true_score REAL NOT NULL,
            predicted_score REAL NOT NULL,
            abs_error REAL NOT NULL,
            stance_flip INTEGER NOT NULL,
            polar INTEGER NOT NULL,
            model_run_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (paragraph_id, model_run_id)
        )
    """)
    rows = []
    for _, r in predictions_df.iterrows():
        true = float(r["true_score"]); pred = float(r["predicted_score"])
        polar = int((true < 0.25) or (true > 0.75))
        flip = int(((true < 0.25) and (pred > 0.75)) or ((true > 0.75) and (pred < 0.25)))
        rows.append((
            r["document_id"], int(r["proposition_id"]), int(r["is_controversial"]),
            r["split"], int(r["is_human"]), r["origin_kind"],
            true, pred, abs(true - pred), flip, polar, run_id,
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO attack_agreement_score_predictions_v2
        (paragraph_id, proposition_id, is_controversial, split, is_human, origin_kind,
         true_score, predicted_score, abs_error, stance_flip, polar, model_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    print(f"  wrote {len(rows)} rows to attack_agreement_score_predictions_v2 (run_id={run_id})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="/workspace/agreement_model_v2",
                   help="checkpoint output dir; default on /workspace because root volume is small")
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_all = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df_all)} rows", flush=True)

    contr_pids = {p["pid"] for p in json.load(open(CONTR_JSON))["propositions"]}
    print(f"controversial proposition ids loaded: {len(contr_pids)}", flush=True)

    train_df, val_df, test_para_df, test_prop_df, val_pids, test_prop_pids = make_splits(
        df_all, contr_pids, args.seed
    )

    # Add is_human helper column for downstream reporting
    for d in (train_df, val_df, test_para_df, test_prop_df):
        d["is_human"] = (d["origin_kind"] == "original_writer").astype(int)

    print("\nTrain composition (origin_kind × is_controversial):")
    print(train_df.groupby(["origin_kind", "is_controversial"]).size().to_string(), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression",
        use_safetensors=True, torch_dtype=torch.float32,
    )
    model.to(torch.float32)

    train_ds = AgreementDataset(train_df, tokenizer, args.max_length)
    val_ds = AgreementDataset(val_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=2,
        seed=args.seed,
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

    # Predict on every split
    print("\n=== running predictions ===", flush=True)
    splits = [
        ("train", train_df), ("val", val_df),
        ("test_para_ood", test_para_df), ("test_prop_ood", test_prop_df),
    ]
    pred_records = []
    for label, d in splits:
        if len(d) == 0:
            continue
        preds = predict_on(trainer, d, tokenizer, args.max_length, label)
        sub = d.copy()
        sub["split"] = label
        sub["predicted_score"] = preds
        sub["true_score"] = sub["agreement_score"].astype(float)
        pred_records.append(sub[["document_id", "proposition_id", "is_controversial",
                                  "split", "is_human", "origin_kind",
                                  "true_score", "predicted_score"]])
        print(f"  {label}: {len(sub)} predictions", flush=True)
    predictions_df = pd.concat(pred_records, ignore_index=True)

    # Save model
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nmodel saved to {final_dir}", flush=True)

    # Write predictions table
    run_id = f"agreement_v2_{int(time.time())}"
    conn = sqlite3.connect(DB_PATH)
    write_predictions_table(conn, run_id, predictions_df)
    conn.close()

    # Reports
    report_splits(predictions_df)

    # Save config metadata
    meta = {
        "run_id": run_id,
        "args": vars(args),
        "val_metrics": val_metrics,
        "splits": {
            "val_pids": sorted(val_pids),
            "test_prop_pids": sorted(test_prop_pids),
            "train_n": int(len(train_df)),
            "val_n": int(len(val_df)),
            "test_para_ood_n": int(len(test_para_df)),
            "test_prop_ood_n": int(len(test_prop_df)),
        },
    }
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nmeta saved to {args.output_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
