#!/bin/bash
# After fold 2 finishes, run fold 3 then fold 1 held-out eval.
# More robust to python errors than the original chain.
cd /workspace/grpo_run

COMMON_ENV="PYTHONUSERBASE=/workspace/pylocal PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True"
COMMON_ARGS="--max-steps 200 --alpha 100 --penalty-shape quadratic --n-propositions 33 --n-train 130 --n-eval 35"

echo "=== [$(date +%T)] waiting for fold 2 DONE ==="
until grep -qE "DONE total|RuntimeError|ValueError|OutOfMemoryError" progress_fold2.log 2>/dev/null; do sleep 30; done
echo "=== [$(date +%T)] fold 2 finished (or errored) ==="
sleep 10  # let GPU settle

echo "=== [$(date +%T)] launching fold 3: train llama8b+gemma9b, held-out qwen7b ==="
> progress_fold3.log
env $COMMON_ENV python3 -u run_pilot_len_pen.py $COMMON_ARGS --name-suffix fold3_heldout_qwen --train-judges llama8b,gemma9b --save-final /workspace/grpo_run/final_fold3 2>&1 | tee progress_fold3.log
echo "=== [$(date +%T)] fold 3 done ==="
sleep 10

echo "=== [$(date +%T)] fold 1 held-out eval (gemma9b) ==="
> progress_fold1_heldout.log
env $COMMON_ENV python3 -u heldout_only_eval.py --fold-dir /workspace/grpo_run/pilot_fold1_heldout_gemma --heldout gemma9b 2>&1 | tee progress_fold1_heldout.log
echo "=== [$(date +%T)] all done ==="
