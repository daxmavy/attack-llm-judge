#!/bin/bash
# 3-fold GRPO: 400 steps each, T7 bundle but embed-sim β=200 (~30% weight target).
# Each fold: train on the same dataset, judges rotate.
set -e
cd /workspace/grpo_run

COMMON_ENV="PYTHONUSERBASE=/workspace/pylocal PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline TMPDIR=/workspace/tmp"

CRITERION=${CRITERION:-clarity}
COMMON_ARGS=(
  --max-steps 400
  --alpha 100
  --dataset-json /workspace/grpo_run/controversial_40_3fold.json
  --lr 5e-6
  --penalty-shape asymm_cubic --penalty-gamma 1000 --penalty-over-tol 0.15
  --embed-sim --embed-beta 200 --embed-threshold 0.85
  --criterion $CRITERION
)

# Folds: rotating held-out judge across the modern panel
declare -A IN_PANEL HELD_OUT
IN_PANEL[1]="qwen95b,llama8b";  HELD_OUT[1]="gemma9b"
IN_PANEL[2]="qwen95b,gemma9b";  HELD_OUT[2]="llama8b"
IN_PANEL[3]="llama8b,gemma9b";  HELD_OUT[3]="qwen95b"

# FOLDS env var lets you resume: FOLDS="2 3" skips fold 1
FOLDS=${FOLDS:-"1 2 3"}
for FOLD in $FOLDS; do
  TRAIN_JUDGES="${IN_PANEL[$FOLD]}"
  HELD="${HELD_OUT[$FOLD]}"
  NAME="grpo_400step_fold${FOLD}_${CRITERION}"
  LOG="progress_${NAME}.log"
  echo "=== [$(date +%T)] launching fold ${FOLD}: train=${TRAIN_JUDGES}, held-out=${HELD} ==="
  > $LOG
  env $COMMON_ENV python3 -u run_pilot_len_pen.py "${COMMON_ARGS[@]}" \
    --train-judges $TRAIN_JUDGES \
    --heldout-judge $HELD \
    --name-suffix $NAME \
    --save-final /workspace/grpo_run/final_${NAME} \
    2>&1 | tee $LOG
  echo "=== [$(date +%T)] fold ${FOLD} done ==="
  # push to HF as soon as this fold finishes (don't wait for other folds)
  REPO="daxmavy/grpo-400step-fold${FOLD}-${CRITERION}"
  echo "=== [$(date +%T)] pushing ${NAME} to ${REPO} ==="
  export $(grep -v '^#' /home/max/attack-llm-judge/.env | xargs) >/dev/null 2>&1
  env $COMMON_ENV python3 -c "
from huggingface_hub import HfApi
import os
tok = os.environ['HF_TOKEN']
api = HfApi(token=tok)
api.create_repo('$REPO', exist_ok=True, private=True, token=tok)
api.upload_folder(folder_path='/workspace/grpo_run/final_${NAME}', repo_id='$REPO',
                  commit_message='GRPO ${NAME}: fold ${FOLD}, criterion ${CRITERION}', token=tok)
print('pushed:', 'https://huggingface.co/$REPO')
" >> $LOG 2>&1 &
  sleep 10
done
echo "=== [$(date +%T)] 3-fold GRPO complete ==="
