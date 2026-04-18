#!/bin/bash
cd /workspace/grpo_run
COMMON_ENV="PYTHONUSERBASE=/workspace/pylocal PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline"
COMMON_ARGS="--max-steps 50 --alpha 100 --penalty-shape quadratic --dataset-json /workspace/grpo_run/controversial_25_dataset.json --train-judges qwen7b,llama8b"

for LR in 5e-6 1e-5 2e-5; do
  TAG="lr${LR}"
  echo "=== [$(date +%T)] launching LR=${LR} pilot ==="
  LOG="progress_lr_${TAG}.log"
  > $LOG
  env $COMMON_ENV python3 -u run_pilot_len_pen.py $COMMON_ARGS --lr $LR --name-suffix "pilot_${TAG}" --save-final "/workspace/grpo_run/final_pilot_${TAG}" 2>&1 | tee $LOG
  echo "=== [$(date +%T)] LR=${LR} done ==="
  sleep 5
done
echo "=== [$(date +%T)] LR sweep complete ==="
