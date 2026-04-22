"""Push the Qwen3-14B QLoRA GRPO fold-1 clarity 100-step checkpoint to HF.

MISSION.md §10 requires this push with round-trip verification.
Adapter-only push (base = Qwen/Qwen3-14B); 138 MB upload.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ENV = "/home/shil6647/attack-llm-judge/.env"
for line in open(ENV):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

from huggingface_hub import HfApi, hf_hub_download


MODEL_DIR = Path("/data/shil6647/attack-llm-judge/final_models/qwen3-14b-grpo-fold1-clarity-100")
EVAL_SUMMARY = Path("/data/shil6647/attack-llm-judge/grpo_run/pilot_qwen3_14b_fold1_clarity_100/eval_summary.json")
BASE_MODEL = "Qwen/Qwen3-14B"
REPO_ID = "daxmavy/qwen3-14b-grpo-fold1-clarity-100"


def build_readme(summary_obj):
    summary = summary_obj.get("summary", {})
    delta_lines = []
    for jn, s in summary.items():
        role = s.get("role", "in_panel")
        delta_lines.append(
            f"| {jn} | {role} | {s['pre_mean']:.2f} | {s['post_mean']:.2f} | {s['delta']:+.2f} |"
        )

    return f"""# Qwen3-14B GRPO — fold1 clarity (100 steps)

QLoRA adapter for `{BASE_MODEL}`, trained with GRPO against a 2-judge clarity
ensemble (Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct), held-out judge
Gemma-2-9b-it. Part of the attack-llm-judge pipeline replication on a modern
rewriter (2026-04-21 overnight mission).

- **Base model**: `{BASE_MODEL}` (bf16 via QLoRA 4-bit)
- **Adapter**: LoRA r=16, α=32, targets Q/K/V/O + MLP projections
- **Training judges**: Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct
- **Held-out judge**: google/gemma-2-9b-it
- **Reward**: `(mean of 2 training judges) − 100·(word_count / target − 1)²`
- **GRPO**: G=4, bsz=8 prompts/step (grad_accum=4), lr=5e-6, β=0.01, T=1.0, **100 steps** (mission scope adjustment vs the 400-step original plan)
- **Training data**: 238 train paragraphs from the controversial_40 3-fold split, fold 1

## Pre vs post on eval set (fold-1 eval, 714 paragraphs)

| Judge | Role | Pre mean | Post mean | Δ |
|---|---|---|---|---|
{chr(10).join(delta_lines)}

## Notes

- `frac_outside_tol = 0.75` on post-eval — 75% of rewrites violated the ±10% length
  tolerance. Length-penalty α=100 did not hold length under a high-variance
  14B rewriter; a length-matched baseline or bounded-generation variant is
  needed before using these rewrites for cross-method comparison.
- Post-eval generation used HF `model.generate` (batch=8, max_new=260) for the
  held-out judge scoring pass — ~60 min for 50 eval rows on one A100. vLLM
  integration for the post-eval rewriter was not available at run time.

See the main project repo (`attack-llm-judge`) `EXPERIMENT_NOTES.md` and
`MISSION.md` for full context.
"""


def main():
    assert MODEL_DIR.exists(), f"missing {MODEL_DIR}"
    assert EVAL_SUMMARY.exists(), f"missing {EVAL_SUMMARY}"
    assert "HF_TOKEN" in os.environ, "HF_TOKEN missing from .env"

    summary_obj = json.loads(EVAL_SUMMARY.read_text())
    readme = build_readme(summary_obj)
    (MODEL_DIR / "README.md").write_text(readme)
    print(f"[{time.strftime('%H:%M:%S')}] README.md written ({len(readme)} bytes)", flush=True)

    api = HfApi(token=os.environ["HF_TOKEN"])
    print(f"[{time.strftime('%H:%M:%S')}] creating repo {REPO_ID} (private)", flush=True)
    api.create_repo(repo_id=REPO_ID, private=True, exist_ok=True, repo_type="model")

    print(f"[{time.strftime('%H:%M:%S')}] uploading folder {MODEL_DIR}", flush=True)
    t0 = time.time()
    api.upload_folder(folder_path=str(MODEL_DIR), repo_id=REPO_ID, repo_type="model",
                       commit_message="GRPO fold1 clarity 100-step QLoRA adapter + metadata")
    print(f"[{time.strftime('%H:%M:%S')}] folder upload done in {(time.time()-t0):.1f}s", flush=True)

    api.upload_file(path_or_fileobj=str(EVAL_SUMMARY), path_in_repo="eval_summary.json",
                     repo_id=REPO_ID, repo_type="model",
                     commit_message="pre/post eval on 714 fold-1 paragraphs")

    # Round-trip: download adapter_config.json from HF and diff
    print(f"[{time.strftime('%H:%M:%S')}] round-trip verify: downloading adapter_config.json", flush=True)
    local_path = hf_hub_download(repo_id=REPO_ID, filename="adapter_config.json", repo_type="model",
                                   token=os.environ["HF_TOKEN"])
    remote_cfg = json.loads(Path(local_path).read_text())
    local_cfg = json.loads((MODEL_DIR / "adapter_config.json").read_text())
    if remote_cfg == local_cfg:
        print(f"[{time.strftime('%H:%M:%S')}] ROUND-TRIP OK: adapter_config.json matches", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ROUND-TRIP MISMATCH!", flush=True)
        sys.exit(2)

    print(f"[{time.strftime('%H:%M:%S')}] pushed → https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
