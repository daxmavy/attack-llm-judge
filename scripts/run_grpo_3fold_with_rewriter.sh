#!/bin/bash
# GRPO 3-fold training for an arbitrary rewriter base.
# Usage: bash run_grpo_3fold_with_rewriter.sh <rewriter-hf-id> <rewriter-short-name> <criterion>
#   e.g. bash run_grpo_3fold_with_rewriter.sh LiquidAI/LFM2.5-1.2B-Instruct lfm25-12b clarity
set -e
cd /data/shil6647/attack-llm-judge/grpo_run

REWRITER="${1:?rewriter HF id required}"
SHORT="${2:?short name required}"
CRITERION="${3:-clarity}"

COMMON_ENV="HF_HOME=/data/shil6647/attack-llm-judge/hf_cache VLLM_CACHE_ROOT=/data/shil6647/attack-llm-judge/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline TMPDIR=/data/shil6647/attack-llm-judge/tmp"

COMMON_ARGS=(
  --max-steps 400
  --alpha 100
  --dataset-json /data/shil6647/attack-llm-judge/grpo_run/controversial_40_3fold.json
  --lr 5e-6
  --penalty-shape asymm_cubic --penalty-gamma 1000 --penalty-over-tol 0.15
  --embed-sim --embed-beta 200 --embed-threshold 0.85
  --criterion "$CRITERION"
  --base-model "$REWRITER"
)

# Fold rotation is resolved from config/models.py (single source of truth;
# see the 2026-04-22 EXPERIMENT_NOTES.md entry on the model-stub refactor).
# We emit the bash array declarations to a tempfile and source it so that a
# require_config() failure inside the python block aborts the whole script
# (set -e + eval can't see the non-zero exit when command substitution wraps
# it).
declare -A IN_PANEL HELD_OUT
_FOLDS_SH=$(mktemp)
trap "rm -f $_FOLDS_SH" EXIT
env $COMMON_ENV python3 - >"$_FOLDS_SH" <<'PY'
import sys
sys.path.insert(0, "/home/shil6647/attack-llm-judge")
from config.models import FOLDS, require_config
require_config()
for fold, spec in FOLDS.items():
    print(f'IN_PANEL[{fold}]="{",".join(spec["in_panel"])}"')
    print(f'HELD_OUT[{fold}]="{spec["held_out"]}"')
PY
source "$_FOLDS_SH"

FOLDS=${FOLDS:-"1 2 3"}
for FOLD in $FOLDS; do
  TRAIN_JUDGES="${IN_PANEL[$FOLD]}"
  HELD="${HELD_OUT[$FOLD]}"
  NAME="grpo_${SHORT}_fold${FOLD}_${CRITERION}"
  LOG="progress_${NAME}.log"
  FINAL_DIR="/data/shil6647/attack-llm-judge/grpo_run/final_${NAME}"
  echo "=== [$(date +%T)] GRPO ${NAME}: base=${REWRITER} train=${TRAIN_JUDGES} held=${HELD} ==="
  > "$LOG"
  env $COMMON_ENV python3 -u /home/shil6647/attack-llm-judge/training/scripts/run_pilot_len_pen.py "${COMMON_ARGS[@]}" \
    --train-judges $TRAIN_JUDGES \
    --heldout-judge $HELD \
    --name-suffix $NAME \
    --save-final "$FINAL_DIR" \
    2>&1 | tee "$LOG"
  echo "=== [$(date +%T)] ${NAME} training done ==="

  # Push to HF
  REPO="daxmavy/grpo-${SHORT}-fold${FOLD}-${CRITERION}"
  echo "=== [$(date +%T)] pushing ${NAME} to ${REPO} ==="
  export $(grep -v '^#' /home/shil6647/attack-llm-judge/.env | xargs) >/dev/null 2>&1
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
  env $COMMON_ENV python3 /home/shil6647/attack-llm-judge/scripts/backfill_grpo_rewriter.py \
    --short "$SHORT" --fold "$FOLD" --criterion "$CRITERION" \
    --rewriter "$REWRITER" --held-out "$HELD" >> "$LOG" 2>&1

  echo "=== [$(date +%T)] ${NAME} pushed + backfilled ==="
  sleep 5
done
echo "=== [$(date +%T)] all 3 folds done for ${SHORT} × ${CRITERION} ==="
