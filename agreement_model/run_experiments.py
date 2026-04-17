"""Run a sweep of agreement-score regression experiments under different train-time
hold-out strategies, and emit a single summary table.

All runs share the same proposition-level 80/10/10 split — only the TRAINING
filter varies. Evaluation is always on the full test set plus subgroup
breakdowns, so each experiment tells us how the model generalises to
paragraph distributions that were partly or fully unseen during training.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/max/attack-llm-judge/agreement_model")
TRAIN = ROOT / "train.py"
RUNS = ROOT / "runs"


EXPERIMENTS = [
    # name, kwargs dict for train.py
    # Baseline: everything goes in.
    ("baseline_all", dict(train_include_writer=1, train_include_edited=1, train_ai_models="all")),
    # Plan item: human-only training.
    ("human_only", dict(train_include_writer=1, train_include_edited=0, train_ai_models="none")),
    # Human + edited (human-revised AI).
    ("human_plus_edited", dict(train_include_writer=1, train_include_edited=1, train_ai_models="none")),
    # Plan item: human + one AI model; evaluate on the other two AI models + humans.
    ("human_plus_claude", dict(train_include_writer=1, train_include_edited=0, train_ai_models="anthropic/claude-sonnet-4")),
    ("human_plus_gpt", dict(train_include_writer=1, train_include_edited=0, train_ai_models="openai/chatgpt-4o-latest")),
    ("human_plus_deepseek", dict(train_include_writer=1, train_include_edited=0, train_ai_models="deepseek/deepseek-chat-v3-0324")),
    # Reverse: train on AI only, test on humans.
    ("ai_only", dict(train_include_writer=0, train_include_edited=0, train_ai_models="all")),
    # Leave-one-AI-out: train on writer + 2 AI models, evaluate on the held-out one.
    ("loo_claude", dict(train_include_writer=1, train_include_edited=0, train_ai_models="openai/chatgpt-4o-latest,deepseek/deepseek-chat-v3-0324")),
    ("loo_gpt", dict(train_include_writer=1, train_include_edited=0, train_ai_models="anthropic/claude-sonnet-4,deepseek/deepseek-chat-v3-0324")),
    ("loo_deepseek", dict(train_include_writer=1, train_include_edited=0, train_ai_models="anthropic/claude-sonnet-4,openai/chatgpt-4o-latest")),
    # Only-one-model controls.
    ("claude_only", dict(train_include_writer=0, train_include_edited=0, train_ai_models="anthropic/claude-sonnet-4")),
    ("gpt_only", dict(train_include_writer=0, train_include_edited=0, train_ai_models="openai/chatgpt-4o-latest")),
    ("deepseek_only", dict(train_include_writer=0, train_include_edited=0, train_ai_models="deepseek/deepseek-chat-v3-0324")),
]


def run_one(name: str, kwargs: dict, epochs: float, seed: int, extra_args: list[str]) -> dict | None:
    out_dir = RUNS / "sweep" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN),
        "--output_dir", str(out_dir),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--no_save_model",
        "--train_include_writer", str(kwargs["train_include_writer"]),
        "--train_include_edited", str(kwargs["train_include_edited"]),
        "--train_ai_models", kwargs["train_ai_models"],
        *extra_args,
    ]
    log_file = out_dir / "train.log"
    t0 = time.time()
    print(f"[{name}] launching: {' '.join(cmd)}")
    with log_file.open("w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    dur = time.time() - t0
    print(f"[{name}] exit={rc} in {dur/60:.1f} min; log -> {log_file}")
    metrics_path = out_dir / "metrics.json"
    if rc != 0 or not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        return json.load(f)


def summarise(name: str, metrics: dict) -> list[dict]:
    """Flatten a metrics.json into one row per (experiment, test subgroup)."""
    rows = []
    test = metrics["test"]
    rows.append({"experiment": name, "subgroup_kind": "overall", "subgroup": "overall", **test["overall"]})
    for key in ("by_paragraph_type", "by_model_name", "by_model_input_condition"):
        for sub_name, m in test[key].items():
            rows.append({"experiment": name, "subgroup_kind": key, "subgroup": sub_name, **m})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="*", default=None, help="subset of experiment names to run")
    p.add_argument("--extra", nargs="*", default=[], help="additional args to forward to train.py")
    args = p.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    sweep_root = RUNS / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    selected = EXPERIMENTS
    if args.only:
        want = set(args.only)
        selected = [(n, k) for (n, k) in EXPERIMENTS if n in want]
        missing = want - {n for n, _ in selected}
        if missing:
            print(f"WARNING: unknown experiment names: {missing}", file=sys.stderr)

    all_rows = []
    summary = {}
    for name, kwargs in selected:
        metrics = run_one(name, kwargs, args.epochs, args.seed, list(args.extra))
        if metrics is None:
            print(f"[{name}] FAILED; continuing")
            continue
        summary[name] = metrics
        all_rows.extend(summarise(name, metrics))
        # Persist progressively so partial runs are usable.
        with (sweep_root / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        _write_csv(sweep_root / "summary.csv", all_rows)
    print(f"\nDone. Wrote {sweep_root}/summary.json and summary.csv")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
