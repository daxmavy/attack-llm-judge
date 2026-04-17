# Training scripts

GRPO training pipeline for the rewriter. Scripts live here; actual run artefacts live at `/workspace/grpo_run/` on the pod.

## Layout

- `scripts/run_min.py` — the original (successful) 25-step minimum run. HF-transformers judges + HF rewriter (TRL default `generate`). Reward = mean of 2 proxy-judge scores.
- `scripts/run_min_vllm.py` — same algorithm but the two judges run via vLLM (3–5× faster scoring). Rewriter still via HF `generate`. **This is the proven reward-climbing path.**
- `scripts/run_min_repro.py` / `run_min_repro_vllm.py` — reproducibility copies with manifest capture and HF-push disabled.
- `scripts/run_pilot_len_pen.py` — pilot launcher with CLI flags for length penalty (`--alpha`, `--tol`), GRPO knobs (`--beta`, `--lr`, `--temperature`, `--num-generations`, `--scale-rewards`, `--loss-type`), step count, and save path.
- `scripts/length_penalty.py` — tolerance-band additive length penalty module. `reward = judge_mean − α·max(0, |len_ratio−1|−tol)`.
- `scripts/run_manifest.py` — per-run manifest: GRPOConfig dump, package versions, git SHA, script SHA-256, GPU info.
- `scripts/finish_eval.py` — post-hoc eval pass: generate pre/post rewrites, score all 3 judges (handles Gemma's no-system-role chat template).

## The 3 judges (see `../MODELS.md`)

- Training proxies (mean-of-two reward): `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`
- Held-out: `google/gemma-2-9b-it`

## Path choice: vLLM judges + HF rewriter

TRL's `use_vllm=True` rewriter path has a numerical drift issue with vllm 0.18.1 + torch 2.10 (importance-sampling ratios collapse within ~5 steps, killing the training signal). The proven alternative uses vLLM only for the 2 judges (reward scoring) and leaves the rewriter to the HF `model.generate` path. 3× fewer vLLM engines, no weight-sync mechanics, proven to produce reward climb.

See `../EXPERIMENT_NOTES.md` §B-10 for the full diagnosis.

## How to launch a pilot

```bash
cd /workspace/grpo_run  # scripts also live here for tmux-friendly invocation
PYTHONUSERBASE=/workspace/pylocal \
PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages \
HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 -u run_pilot_len_pen.py \
  --max-steps 25 --alpha 25 --tol 0.10 --name-suffix pilot1
```

## Known constraints

- VRAM budget on an A100-80GB: rewriter-training ~20 GB + ref policy ~3 GB + 2 vLLM judges ~37 GB = ~60 GB. TRL's `use_vllm=True` (additional vLLM engine for rewriter) pushes over 75 GB and risks OOM during the optimizer step.
- vllm pinned to 0.18.1 to stay inside TRL 1.2.0's "supported vLLM" band.
- Pod disk is 20 GB on `/`; models cache to `/workspace/hf_cache` (830 TB persistent volume).
