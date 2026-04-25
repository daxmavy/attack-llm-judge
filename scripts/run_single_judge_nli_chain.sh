#!/usr/bin/env bash
# Single-judge NLI GRPO sweep (18 runs).
# Waits for cmdr7b OOS scoring (PID 2041659) to exit before starting GPU work.
# fold N → in-panel = single judge ; rest are held-out (not used in reward).
set -u

if kill -0 2041659 2>/dev/null; then
  echo "=== [$(date +%H:%M:%S)] waiting for cmdr7b scoring (PID 2041659) to exit ==="
  while kill -0 2041659 2>/dev/null; do sleep 30; done
  echo "=== [$(date +%H:%M:%S)] cmdr7b done; starting single-judge sweep ==="
fi

FOLD_TO_JUDGES() {
  # echoes "IN_PANEL HELDOUT" — third mission judge stays fully unseen by training
  # for cleaner OOS-style measurement post-hoc.
  case "$1" in
    1) echo "qwen95b llama8b" ;;
    2) echo "llama8b gemma9b" ;;
    3) echo "gemma9b qwen95b" ;;
  esac
}

COMBOS=(
  "qwen25-15b   Qwen/Qwen2.5-1.5B-Instruct     clarity"
  "qwen25-15b   Qwen/Qwen2.5-1.5B-Instruct     informativeness"
  "lfm25-12b    LiquidAI/LFM2.5-1.2B-Instruct  clarity"
  "lfm25-12b    LiquidAI/LFM2.5-1.2B-Instruct  informativeness"
  "gemma3-1b    google/gemma-3-1b-it           clarity"
  "gemma3-1b    google/gemma-3-1b-it           informativeness"
)

for combo in "${COMBOS[@]}"; do
  read -r SHORT HFID CRIT <<<"$combo"
  for FOLD in 1 2 3; do
    read -r JUDGE HELDOUT <<<"$(FOLD_TO_JUDGES $FOLD)"
    NAME="grpo_nli_single_${SHORT}_fold${FOLD}_${CRIT}"
    SAVE="/workspace/grpo_run/final_${NAME}"
    HF_REPO="daxmavy/grpo-${SHORT}-fold${FOLD}-${CRIT}-nli-single"

    # Skip if already done
    DONE_ROWS=$(python3 -c "
import sqlite3
c = sqlite3.connect('/home/max/attack-llm-judge/data/paragraphs.db')
n = c.execute(\"SELECT COUNT(*) FROM attack_rewrites WHERE method='grpo_nli_single' AND rewriter_model=? AND fold=? AND criterion=?\", ('$HFID', $FOLD, '$CRIT')).fetchone()[0]
print(n)
")
    if [ "$DONE_ROWS" = "714" ]; then
      echo "=== [$(date +%H:%M:%S)] SKIP $NAME (already 714 rows in DB) ==="
      continue
    fi

    echo "=== [$(date +%H:%M:%S)] starting $NAME (in-panel=$JUDGE  heldout=$HELDOUT) ==="
    python3 -u /workspace/grpo_run/run_pilot_len_pen.py \
      --max-steps 400 --alpha 100 \
      --dataset-json /workspace/grpo_run/controversial_40_3fold.json \
      --lr 5e-6 --penalty-shape asymm_cubic --penalty-gamma 1000 --penalty-over-tol 0.15 \
      --nli-fidelity --criterion "$CRIT" \
      --base-model "$HFID" \
      --train-judges "$JUDGE" --heldout-judge "$HELDOUT" \
      --name-suffix "$NAME" \
      --save-final "$SAVE"
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "=== [$(date +%H:%M:%S)] $NAME training FAILED rc=$RC — aborting chain ==="
      exit $RC
    fi

    echo "=== [$(date +%H:%M:%S)] $NAME training done; pushing + backfilling ==="
    PYTHONPATH=/workspace/pylocal/lib/python3.11/site-packages python3 -c "
from huggingface_hub import HfApi, create_repo
import os
token = os.environ['HF_TOKEN']
api = HfApi(token=token)
create_repo('$HF_REPO', token=token, exist_ok=True, private=False)
api.upload_folder(folder_path='$SAVE', repo_id='$HF_REPO', token=token,
                  commit_message='GRPO 400-step $SHORT fold $FOLD $CRIT --nli-fidelity + single in-panel=$JUDGE held-out=$HELDOUT')
print('HF push done')
"
    python3 /home/max/attack-llm-judge/scripts/backfill_grpo_rewriter.py \
      --short "$SHORT" --fold $FOLD --criterion "$CRIT" \
      --rewriter "$HFID" --held-out "$HELDOUT" \
      --method-tag grpo_nli_single --name-prefix grpo_nli_single
    rm -rf "$SAVE"
    echo "=== [$(date +%H:%M:%S)] $NAME DONE ==="
  done
done
echo "=== [$(date +%H:%M:%S)] ALL SINGLE-JUDGE GRPO DONE ==="
