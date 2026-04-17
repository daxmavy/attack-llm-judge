"""Train a regression model: (proposition, paragraph) -> agreement_score in [0, 1].

Uses aggregated rater `writer_stance` (= `avg_stance` / 100 = `agreement_score`
column in paul_data/prepared/documents.csv), not the writer's self-assessment.

Default split: group-by-proposition 80/10/10 so propositions in val/test are
unseen during training. Also reports subgroup metrics (paragraph_type,
model_name) on the test set.
"""

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


DATA_CSV = "/home/max/attack-llm-judge/paul_data/prepared/documents.csv"
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
            self.propositions[idx],
            self.documents[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = float(self.labels[idx])
        return enc


def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).squeeze().astype(np.float64)
    labels = np.asarray(labels).squeeze().astype(np.float64)
    # Replace NaN/inf (can occur early in training) with the label mean so metrics stay finite.
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


def group_split(df: pd.DataFrame, group_col: str, sizes=(0.8, 0.1, 0.1), seed=42):
    assert abs(sum(sizes) - 1.0) < 1e-6
    train_size, val_size, test_size = sizes
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    idx_train, idx_hold = next(gss1.split(df, groups=df[group_col]))
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_hold = df.iloc[idx_hold].reset_index(drop=True)
    rel_test = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    idx_val, idx_test = next(gss2.split(df_hold, groups=df_hold[group_col]))
    return df_train, df_hold.iloc[idx_val].reset_index(drop=True), df_hold.iloc[idx_test].reset_index(drop=True)


def subgroup_report(trainer, df_test: pd.DataFrame, tokenizer, max_length: int):
    report = {}
    preds_all = trainer.predict(AgreementDataset(df_test, tokenizer, max_length))
    y_pred = np.clip(preds_all.predictions.squeeze().astype(np.float64), 0.0, 1.0)
    y_true = df_test["agreement_score"].to_numpy()
    report["overall"] = compute_metrics_fn((y_pred, y_true))
    report["overall"]["n"] = int(len(y_true))

    df_test = df_test.copy()
    df_test["_pred"] = y_pred
    for col in ("paragraph_type", "model_name", "model_input_condition"):
        sub_report = {}
        for val, grp in df_test.groupby(col, dropna=False):
            if len(grp) < 20:
                continue
            m = compute_metrics_fn((grp["_pred"].to_numpy(), grp["agreement_score"].to_numpy()))
            m["n"] = int(len(grp))
            sub_report[str(val)] = m
        report[f"by_{col}"] = sub_report
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="/home/max/attack-llm-judge/agreement_model/runs/main")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    # Training-set filters. Evaluation is ALWAYS on the full test split so all runs
    # share the same held-out propositions and subgroup metrics are comparable.
    p.add_argument("--train_include_writer", type=int, default=1, help="1 to include human (writer) paragraphs in training")
    p.add_argument("--train_include_edited", type=int, default=1, help="1 to include human-edited AI paragraphs in training")
    p.add_argument(
        "--train_ai_models",
        default="all",
        help="'all', 'none', or comma-separated model_names to include in training (among paragraph_type=model rows)",
    )
    p.add_argument("--no_save_model", action="store_true", default=False, help="skip writing model weights (for experiment sweeps)")
    # DeBERTa-v3 can produce NaNs in bf16/fp16 due to disentangled-attention softmax; fp32 is reliable.
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_all = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df_all)} total rows")

    df_train_full, df_val_full, df_test = group_split(df_all, group_col="proposition_id", seed=args.seed)

    # Apply training-set filters AFTER splitting so test is always the same full set.
    def apply_train_filter(df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(False, index=df.index)
        if args.train_include_writer:
            mask |= df["paragraph_type"] == "writer"
        if args.train_include_edited:
            mask |= df["paragraph_type"] == "edited"
        ai_spec = args.train_ai_models.strip()
        if ai_spec == "all":
            mask |= df["paragraph_type"] == "model"
        elif ai_spec == "none":
            pass
        else:
            wanted = set(m.strip() for m in ai_spec.split(",") if m.strip())
            mask |= (df["paragraph_type"] == "model") & df["model_name"].isin(wanted)
        return df[mask].reset_index(drop=True)

    df_train = apply_train_filter(df_train_full)
    df_val = apply_train_filter(df_val_full)
    print(
        f"Full split: train_grp={len(df_train_full)} val_grp={len(df_val_full)} test={len(df_test)} "
        f"(propositions: {df_train_full.proposition_id.nunique()}/{df_val_full.proposition_id.nunique()}/{df_test.proposition_id.nunique()})"
    )
    print(
        f"After training filter (include_writer={args.train_include_writer}, include_edited={args.train_include_edited}, "
        f"ai_models={args.train_ai_models}): train={len(df_train)} val={len(df_val)}"
    )
    print("Train composition:")
    print(df_train.groupby(["paragraph_type", "model_name"], dropna=False).size().to_string())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="regression",
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    model.to(torch.float32)

    train_ds = AgreementDataset(df_train, tokenizer, args.max_length)
    val_ds = AgreementDataset(df_val, tokenizer, args.max_length)

    # Sweep runs (--no_save_model) skip per-epoch checkpoints entirely to avoid
    # filling disk with optimizer state (~2 GB/epoch for DeBERTa-v3-base).
    save_epoch_ckpts = not args.no_save_model
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch" if save_epoch_ckpts else "no",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=save_epoch_ckpts,
        metric_for_best_model="mae" if save_epoch_ckpts else None,
        greater_is_better=False if save_epoch_ckpts else None,
        save_total_limit=1 if save_epoch_ckpts else None,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        report_to="none",
        dataloader_num_workers=2,
        seed=args.seed,
    )

    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    print("Val:", val_metrics)

    test_report = subgroup_report(trainer, df_test, tokenizer, args.max_length)
    print("Test (overall):", test_report["overall"])

    if not args.no_save_model:
        final_dir = Path(args.output_dir) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

    train_composition = (
        df_train.groupby(["paragraph_type", "model_name"], dropna=False)
        .size()
        .reset_index(name="n")
        .to_dict(orient="records")
    )
    with open(Path(args.output_dir) / "metrics.json", "w") as f:
        json.dump(
            {
                "val": val_metrics,
                "test": test_report,
                "args": vars(args),
                "train_n": int(len(df_train)),
                "train_composition": train_composition,
            },
            f,
            indent=2,
            default=str,
        )

    for sub_key in ("by_paragraph_type", "by_model_name", "by_model_input_condition"):
        print(f"\n{sub_key}:")
        for k, v in test_report[sub_key].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
