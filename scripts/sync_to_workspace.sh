#!/usr/bin/env bash
# Sync canonical repo scripts into /workspace/grpo_run/ (the live runtime dir).
#
# The pipeline runs FROM /workspace/grpo_run/: shell drivers invoke `python3
# /workspace/grpo_run/run_pilot_len_pen.py ...`, and Python scripts do
# `sys.path.insert(0, "/workspace/grpo_run")` to import JudgeVLLM etc.
#
# The canonical copy lives in this repo. After pulling updates, run this
# script to refresh /workspace/grpo_run/ so runtime matches the repo.
#
# On a fresh replication host, rebind /workspace/grpo_run/ to wherever
# your scratch dir is and edit the paths below.
set -euo pipefail

REPO="${REPO:-/home/max/attack-llm-judge}"
DEST="${DEST:-/workspace/grpo_run}"
mkdir -p "$DEST"

# Python modules + entrypoints that are imported / executed at runtime
for f in length_penalty.py run_manifest.py run_pilot_len_pen.py; do
  cp "$REPO/training/scripts/$f" "$DEST/$f"
done
for f in run_icir.py run_mission_attacks.py stop_signal.py \
         run_full_criterion_with_rewriter.sh run_grpo_3fold_with_rewriter.sh; do
  cp "$REPO/scripts/$f" "$DEST/$f"
done

echo "synced $(ls "$DEST"/*.py "$DEST"/*.sh 2>/dev/null | wc -l) files into $DEST"
