#!/bin/bash
# GRPO 3-fold training for an arbitrary rewriter base.
# Usage: bash run_grpo_3fold_with_rewriter.sh <rewriter-hf-id> <rewriter-short-name> <criterion>
#   e.g. bash run_grpo_3fold_with_rewriter.sh LiquidAI/LFM2.5-1.2B-Instruct lfm25-12b clarity
set -e
cd /workspace/grpo_run

REWRITER="${1:?rewriter HF id required}"
SHORT="${2:?short name required}"
CRITERION="${3:-clarity}"

COMMON_ENV="PYTHONUSERBASE=/workspace/pylocal PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline TMPDIR=/workspace/tmp"

COMMON_ARGS=(
  --max-steps 400
  --alpha 100
  --dataset-json /workspace/grpo_run/controversial_40_3fold.json
  --lr 5e-6
  --penalty-shape asymm_cubic --penalty-gamma 1000 --penalty-over-tol 0.15
  --embed-sim --embed-beta 200 --embed-threshold 0.85
  --criterion "$CRITERION"
  --base-model "$REWRITER"
)

declare -A IN_PANEL HELD_OUT
IN_PANEL[1]="qwen95b,llama8b";  HELD_OUT[1]="gemma9b"
IN_PANEL[2]="qwen95b,gemma9b";  HELD_OUT[2]="llama8b"
IN_PANEL[3]="llama8b,gemma9b";  HELD_OUT[3]="qwen95b"

FOLDS=${FOLDS:-"1 2 3"}
for FOLD in $FOLDS; do
  TRAIN_JUDGES="${IN_PANEL[$FOLD]}"
  HELD="${HELD_OUT[$FOLD]}"
  NAME="grpo_${SHORT}_fold${FOLD}_${CRITERION}"
  LOG="progress_${NAME}.log"
  FINAL_DIR="/workspace/grpo_run/final_${NAME}"
  echo "=== [$(date +%T)] GRPO ${NAME}: base=${REWRITER} train=${TRAIN_JUDGES} held=${HELD} ==="
  > "$LOG"
  env $COMMON_ENV python3 -u run_pilot_len_pen.py "${COMMON_ARGS[@]}" \
    --train-judges $TRAIN_JUDGES \
    --heldout-judge $HELD \
    --name-suffix $NAME \
    --save-final "$FINAL_DIR" \
    2>&1 | tee "$LOG"
  echo "=== [$(date +%T)] ${NAME} training done ==="

  # Push to HF
  REPO="daxmavy/grpo-${SHORT}-fold${FOLD}-${CRITERION}"
  echo "=== [$(date +%T)] pushing ${NAME} to ${REPO} ==="
  export $(grep -v '^#' /home/max/attack-llm-judge/.env | xargs) >/dev/null 2>&1
  env $COMMON_ENV python3 -c "
from huggingface_hub import HfApi
import os
tok = os.environ['HF_TOKEN']
api = HfApi(token=tok)
api.create_repo('$REPO', exist_ok=True, private=True, token=tok)
api.upload_folder(folder_path='$FINAL_DIR', repo_id='$REPO',
                  commit_message='GRPO ${NAME}: fold ${FOLD}, criterion ${CRITERION}, base ${REWRITER}', token=tok)
print('pushed:', 'https://huggingface.co/$REPO')
" >> "$LOG" 2>&1

  # Backfill DB (rewrites + held-out scores) via dedicated script
  env $COMMON_ENV python3 /home/max/backfill_grpo_rewriter.py \
    --short "$SHORT" --fold "$FOLD" --criterion "$CRITERION" \
    --rewriter "$REWRITER" --held-out "$HELD" >> "$LOG" 2>&1

  echo "=== [$(date +%T)] ${NAME} pushed + backfilled ==="
  sleep 5
done
echo "=== [$(date +%T)] all 3 folds done for ${SHORT} × ${CRITERION} ==="
