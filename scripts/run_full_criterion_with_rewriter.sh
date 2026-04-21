#!/bin/bash
# Run the full 3-fold attack pipeline (feedback-free + BoN + ICIR + GRPO) for one rewriter × criterion.
# Usage: bash run_full_criterion_with_rewriter.sh <rewriter-hf-id> <short> <criterion>
set -e
cd /data/shil6647/attack-llm-judge/grpo_run
REWRITER="${1:?rewriter required}"
SHORT="${2:?short required}"
CRITERION="${3:?criterion required}"

COMMON_ENV="HF_HOME=/data/shil6647/attack-llm-judge/hf_cache VLLM_CACHE_ROOT=/data/shil6647/attack-llm-judge/vllm_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline TMPDIR=/data/shil6647/attack-llm-judge/tmp"

SCRIPTS=/home/shil6647/attack-llm-judge/scripts

echo "=== [$(date +%T)] [$SHORT × $CRITERION] feedback_free ==="
env $COMMON_ENV python3 -u "$SCRIPTS/run_mission_attacks.py" feedback_free --criterion "$CRITERION" --rewriter "$REWRITER"

echo "=== [$(date +%T)] [$SHORT × $CRITERION] bon_generate K=16 ==="
env $COMMON_ENV python3 -u "$SCRIPTS/run_mission_attacks.py" bon_generate --criterion "$CRITERION" --k 16 --rewriter "$REWRITER"

for FOLD in 1 2 3; do
  echo "=== [$(date +%T)] [$SHORT × $CRITERION] bon_score fold $FOLD ==="
  env $COMMON_ENV python3 -u "$SCRIPTS/run_mission_attacks.py" bon_score $FOLD --criterion "$CRITERION" --rewriter "$REWRITER"
done

for FOLD in 1 2 3; do
  echo "=== [$(date +%T)] [$SHORT × $CRITERION] icir fold $FOLD ==="
  env $COMMON_ENV python3 -u "$SCRIPTS/run_icir.py" --folds $FOLD --criteria "$CRITERION" --rewriter "$REWRITER"
done

echo "=== [$(date +%T)] [$SHORT × $CRITERION] launching 3-fold GRPO ==="
bash "$SCRIPTS/run_grpo_3fold_with_rewriter.sh" "$REWRITER" "$SHORT" "$CRITERION"

echo "=== [$(date +%T)] [$SHORT × $CRITERION] FULL PIPELINE DONE ==="
