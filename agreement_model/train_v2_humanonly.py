"""Train agreement_score regressor v2 — HUMAN-only training, AI as
distribution-shift test set.

Same hold-out strategy as train_v2.py:
  - VAL: 5 random controversial + 5 random non-controversial propositions
  - TEST_PROP_OOD: 10% of remaining non-controversial propositions
  - TEST_PARA_OOD: 10% of paragraphs per (prop × origin_kind) for remaining
                   propositions

DIFFERENCE: every split uses ONLY origin_kind='original_writer' (human-written)
paragraphs. AI-generated paragraphs (origin_kind='original_model') are
completely held out from training AND validation, then evaluated as a
distribution-shift test slice ('test_dist_shift_ai').

Edited paragraphs ('original_edited') are also held out and reported as a
secondary distribution-shift slice ('test_dist_shift_edited') for completeness.

Recipe matches v2 (DeBERTa-v3-base, 4 epochs, lr=1e-5, batch=32, max_length=256).
Best-checkpoint by val MAE on humans only.

Predictions written with model_run_id 'agreement_v2_humanonly_<ts>' to the same
attack_agreement_score_predictions_v2 table — distinguishable from the v2
(all-data) run by model_run_id.
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


def make_splits_humanonly(df, controversial_pids, seed):
    """Train splits use ONLY origin_kind='original_writer' rows."""
    rng = random.Random(seed)

    df = df.copy()
    df["origin_kind"] = df["paragraph_type"].map({
        "writer": "original_writer", "model": "original_model", "edited": "original_edited",
    })
    df["is_controversial"] = df["proposition_id"].isin(controversial_pids).astype(int)

    all_pids = df["proposition_id"].unique().tolist()
    contr_pids = sorted(p for p in all_pids if p in controversial_pids)
    nonctr_pids = sorted(p for p in all_pids if p not in controversial_pids)
    print(f"  total props: {len(all_pids)}  controversial: {len(contr_pids)}  non-controversial: {len(nonctr_pids)}", flush=True)

    val_contr = rng.sample(contr_pids, 5)
    val_nonctr = rng.sample(nonctr_pids, 5)
    val_pids = set(val_contr + val_nonctr)
    remaining_nonctr = [p for p in nonctr_pids if p not in val_pids]

    n_test_prop = max(1, int(round(0.10 * len(remaining_nonctr))))
    test_prop_pids = set(rng.sample(remaining_nonctr, n_test_prop))

    train_props = [p for p in all_pids if p not in val_pids and p not in test_prop_pids]
    print(f"  val props: {len(val_pids)} ({len(val_contr)} contr + {len(val_nonctr)} nonctr)", flush=True)
    print(f"  test_prop_ood props: {len(test_prop_pids)} (all non-controversial)", flush=True)
    print(f"  train props: {len(train_props)}", flush=True)

    # Filter to humans only for the in-distribution splits (train/val/test_para/test_prop)
    df_human = df[df["origin_kind"] == "original_writer"]

    train_indices = []
    test_para_indices = []
    for pid in train_props:
        sub = df_human[df_human["proposition_id"] == pid]
        n = len(sub)
        n_test = max(1, int(round(0.10 * n))) if n >= 10 else max(0, n // 10)
        shuffled = sub.sample(frac=1.0, random_state=seed + pid).index.tolist()
        test_para_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])
    val_indices = df_human[df_human["proposition_id"].isin(val_pids)].index.tolist()
    test_prop_indices = df_human[df_human["proposition_id"].isin(test_prop_pids)].index.tolist()

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_para_df = df.loc[test_para_indices].reset_index(drop=True)
    test_prop_df = df.loc[test_prop_indices].reset_index(drop=True)

    # Distribution-shift test slices: ALL AI paragraphs (any proposition)
    test_dist_shift_ai = df[df["origin_kind"] == "original_model"].reset_index(drop=True)
    test_dist_shift_edited = df[df["origin_kind"] == "original_edited"].reset_index(drop=True)

    print(f"  paragraph counts: train={len(train_df)} val={len(val_df)} "
          f"test_para_ood={len(test_para_df)} test_prop_ood={len(test_prop_df)}", flush=True)
    print(f"  dist-shift slices (held-out completely): "
          f"test_dist_shift_ai={len(test_dist_shift_ai)} "
          f"test_dist_shift_edited={len(test_dist_shift_edited)}", flush=True)

    return (train_df, val_df, test_para_df, test_prop_df,
            test_dist_shift_ai, test_dist_shift_edited,
            val_pids, test_prop_pids)


def stance_flip_rate(true, pred):
    true = np.asarray(true); pred = np.asarray(pred)
    polar = (true < 0.25) | (true > 0.75)
    n_polar = int(polar.sum())
    if n_polar == 0: return 0.0, 0, 0
    flips = ((true < 0.25) & (pred > 0.75)) | ((true > 0.75) & (pred < 0.25))
    return int((flips & polar).sum()) / n_polar, int((flips & polar).sum()), n_polar


def predict_on(trainer, df, tokenizer, max_length):
    if len(df) == 0: return np.zeros(0)
    out = trainer.predict(AgreementDataset(df, tokenizer, max_length))
    return np.clip(out.predictions.squeeze().astype(np.float64), 0.0, 1.0)


def report(predictions_df, p95_too=True):
    print("\n=== overall metrics by split ===", flush=True)
    print(f'  {"split":<25} {"n":>6}  {"MAE":>7} {"p95":>7} {"flip":>7}', flush=True)
    for split, sub in predictions_df.groupby("split"):
        true = sub["true_score"].to_numpy(); pred = sub["predicted_score"].to_numpy()
        mae = mean_absolute_error(true, pred)
        ae = np.abs(true - pred)
        p95 = float(np.percentile(ae, 95))
        sf, n_f, n_p = stance_flip_rate(true, pred)
        print(f"  {split:<25} {len(sub):>6}  {mae:.4f}  {p95:.4f}  {sf:.4f} ({n_f}/{n_p} polar)", flush=True)


def write_predictions(conn, run_id, predictions_df):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attack_agreement_score_predictions_v2 (
            paragraph_id TEXT NOT NULL, proposition_id INTEGER NOT NULL,
            is_controversial INTEGER NOT NULL, split TEXT NOT NULL,
            is_human INTEGER NOT NULL, origin_kind TEXT NOT NULL,
            true_score REAL NOT NULL, predicted_score REAL NOT NULL,
            abs_error REAL NOT NULL, stance_flip INTEGER NOT NULL, polar INTEGER NOT NULL,
            model_run_id TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    print(f"  wrote {len(rows)} rows (run_id={run_id})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="/workspace/agreement_model_v2_humanonly")
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

    (train_df, val_df, test_para_df, test_prop_df,
     test_ai_df, test_edited_df, val_pids, test_prop_pids) = make_splits_humanonly(
        df_all, contr_pids, args.seed
    )

    for d in (train_df, val_df, test_para_df, test_prop_df, test_ai_df, test_edited_df):
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

    print("\n=== running predictions on all splits incl distribution-shift slices ===", flush=True)
    splits = [
        ("train", train_df), ("val", val_df),
        ("test_para_ood", test_para_df), ("test_prop_ood", test_prop_df),
        ("test_dist_shift_ai", test_ai_df),
        ("test_dist_shift_edited", test_edited_df),
    ]
    pred_records = []
    for label, d in splits:
        if len(d) == 0: continue
        preds = predict_on(trainer, d, tokenizer, args.max_length)
        sub = d.copy()
        sub["split"] = label
        sub["predicted_score"] = preds
        sub["true_score"] = sub["agreement_score"].astype(float)
        pred_records.append(sub[["document_id","proposition_id","is_controversial","split",
                                  "is_human","origin_kind","true_score","predicted_score"]])
        print(f"  {label}: {len(sub)} predictions", flush=True)
    predictions_df = pd.concat(pred_records, ignore_index=True)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nmodel saved to {final_dir}", flush=True)

    run_id = f"agreement_v2_humanonly_{int(time.time())}"
    conn = sqlite3.connect(DB_PATH)
    write_predictions(conn, run_id, predictions_df)
    conn.close()

    report(predictions_df)

    meta = {
        "run_id": run_id, "args": vars(args),
        "val_metrics": val_metrics,
        "splits": {
            "val_pids": sorted(val_pids), "test_prop_pids": sorted(test_prop_pids),
            "train_n": int(len(train_df)), "val_n": int(len(val_df)),
            "test_para_ood_n": int(len(test_para_df)), "test_prop_ood_n": int(len(test_prop_df)),
            "test_dist_shift_ai_n": int(len(test_ai_df)),
            "test_dist_shift_edited_n": int(len(test_edited_df)),
        },
    }
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nmeta saved to {args.output_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
