"""Push all 3 fold models + eval summaries to HF.

Creates 3 private HF repos: daxmavy/attack-llm-judge-grpo-fold{1,2,3}-20260418
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

for line in open("/home/max/attack-llm-judge/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

from huggingface_hub import HfApi


FOLDS = [
    {
        "name": "fold1",
        "model_dir": "/workspace/grpo_run/final_fold1",
        "summary": "/workspace/grpo_run/pilot_fold1_heldout_gemma/eval_summary.json",
        "train_judges": ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        "heldout_judge": "google/gemma-2-9b-it",
    },
    {
        "name": "fold2",
        "model_dir": "/workspace/grpo_run/final_fold2",
        "summary": "/workspace/grpo_run/pilot_fold2_heldout_llama/eval_summary.json",
        "train_judges": ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"],
        "heldout_judge": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "fold3",
        "model_dir": "/workspace/grpo_run/final_fold3",
        "summary": "/workspace/grpo_run/pilot_fold3_heldout_qwen/eval_summary.json",
        "train_judges": ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"],
        "heldout_judge": "Qwen/Qwen2.5-7B-Instruct",
    },
]


def push_one(fold, api):
    name = fold["name"]
    model_dir = Path(fold["model_dir"])
    summary_path = Path(fold["summary"])
    assert model_dir.exists(), model_dir
    assert summary_path.exists(), summary_path

    summary = json.loads(summary_path.read_text())
    delta_lines = []
    if "summary" in summary:
        for jn, s in summary["summary"].items():
            delta_lines.append(
                f"| {jn} | {s.get('role','train')} | {s['pre_mean']:.2f} | {s['post_mean']:.2f} | {s['delta']:+.2f} |")

    date = time.strftime("%Y%m%d")
    repo_id = f"daxmavy/attack-llm-judge-grpo-{name}-{date}"
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    print(f"[{name}] creating repo {repo_id}", flush=True)

    readme = f"""# GRPO 3-fold mission — {name} ({date})

Mission: train a Qwen2.5-1.5B-Instruct rewriter with GRPO against a 2-judge
ensemble reward, with a quadratic length penalty (α=100, penalty = α·(ratio−1)²).
See `../STRUCTURE.md` and `../EXPERIMENT_NOTES.md` in the main project repo for
full context. The three folds rotate which of Qwen-7B / Llama-8B / Gemma-9B is
held-out; this fold is **{name}**.

- Rewriter base: `Qwen/Qwen2.5-1.5B-Instruct`
- Training judges: {fold['train_judges']}
- **Held-out judge**: `{fold['heldout_judge']}`
- Reward: `(mean of 2 training judges) − 100·(word_count / target − 1)²`
- Training data: 33 propositions × 5 top-decile writer paragraphs each; 130 train / 35 eval (disjoint paragraphs from the same proposition set).
- GRPO: G=4, bsz=8 prompts/step (grad_accum=4), lr=5e-6, β=0.01, T=1.0, 200 steps.
- Judges: vLLM bf16 enforce_eager on the same A100-80GB as training.

## Pre vs post on eval set

| Judge | Role | Pre | Post | Δ |
|---|---|---|---|---|
{chr(10).join(delta_lines)}

Detailed eval in `eval_summary.json`.
"""
    (model_dir / "README.md").write_text(readme)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(summary_path),
                     path_in_repo="eval_summary.json",
                     repo_id=repo_id, repo_type="model")
    print(f"[{name}] pushed to https://huggingface.co/{repo_id}", flush=True)


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    for fold in FOLDS:
        try:
            push_one(fold, api)
        except Exception as e:
            print(f"[{fold['name']}] push FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
