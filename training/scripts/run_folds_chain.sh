#!/bin/bash
# Chain fold 2 → fold 3 sequentially after fold 1 finishes.
# Waits for fold 1 DONE marker, then runs each.
set -e
cd /workspace/grpo_run

COMMON_ENV="PYTHONUSERBASE=/workspace/pylocal PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True"
COMMON_ARGS="--max-steps 200 --alpha 100 --penalty-shape quadratic --n-propositions 33 --n-train 130 --n-eval 35"

echo "=== [$(date +%T)] waiting for fold 1 DONE ==="
until grep -qE "DONE total|ERROR|Traceback" progress_fold1.log 2>/dev/null; do sleep 30; done
echo "=== [$(date +%T)] fold 1 finished ==="

echo "=== [$(date +%T)] launching fold 2: train qwen7b+gemma9b, held-out llama8b ==="
> progress_fold2.log
env $COMMON_ENV python3 -u run_pilot_len_pen.py $COMMON_ARGS --name-suffix fold2_heldout_llama --train-judges qwen7b,gemma9b --save-final /workspace/grpo_run/final_fold2 2>&1 | tee progress_fold2.log
echo "=== [$(date +%T)] fold 2 done ==="

echo "=== [$(date +%T)] launching fold 3: train llama8b+gemma9b, held-out qwen7b ==="
> progress_fold3.log
env $COMMON_ENV python3 -u run_pilot_len_pen.py $COMMON_ARGS --name-suffix fold3_heldout_qwen --train-judges llama8b,gemma9b --save-final /workspace/grpo_run/final_fold3 2>&1 | tee progress_fold3.log
echo "=== [$(date +%T)] fold 3 done ==="

echo "=== [$(date +%T)] fold 1 held-out eval (gemma9b): fold 1 used old script, missing held-out ==="
> progress_fold1_heldout.log
env $COMMON_ENV python3 -u heldout_only_eval.py --fold-dir /workspace/grpo_run/pilot_fold1_heldout_gemma --heldout gemma9b 2>&1 | tee progress_fold1_heldout.log
echo "=== [$(date +%T)] fold 1 held-out eval done ==="
