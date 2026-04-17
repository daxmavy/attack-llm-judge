#!/bin/bash
# Autonomous Claude Code loop. Each iteration spawns a fresh headless session
# that reads plan.md + git state and makes progress. On non-zero exit (likely
# rate limit), sleeps briefly and retries. On success, sleeps ~5h to align
# with Max session windows.
#
# Run with: caffeinate -i "$0" > .claude-loop-logs/driver.log 2>&1 &
# Stop by: touch .claude-loop-logs/STOP  (takes effect before next iteration)

set -u

PROJECT_DIR="$HOME/oxford/attack-llm-judge"
LOG_DIR="$PROJECT_DIR/.claude-loop-logs"
STOP_FILE="$LOG_DIR/STOP"
mkdir -p "$LOG_DIR"

PROMPT='You are resuming autonomous work on this project. Steps:
1. Read plan.md for the roadmap and current intent.
2. Run `git log --oneline -20` and `git status` to see the current state.
3. Work on the next unfinished item in plan.md.
4. Commit at natural checkpoints with clear messages.
5. When you complete an item in plan.md, mark it done in the file.

Stop conditions (exit after writing to .claude-loop-logs/blockers.md AND creating .claude-loop-logs/STOP):
- plan.md appears complete
- you hit an ambiguity or blocker that needs human input
- tests are failing in a way you cannot diagnose

Do not invent work outside plan.md. Do not modify continue.sh or files under .claude-loop-logs/ (except blockers.md and STOP).'

MAX_ITERATIONS=${MAX_ITERATIONS:-40}
SUCCESS_SLEEP_SECONDS=${SUCCESS_SLEEP_SECONDS:-18000}  # 5h
RETRY_SLEEP_SECONDS=${RETRY_SLEEP_SECONDS:-900}        # 15m

cd "$PROJECT_DIR" || { echo "cannot cd to $PROJECT_DIR"; exit 1; }

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  if [[ -f "$STOP_FILE" ]]; then
    echo "[$(date)] STOP file present, exiting"
    exit 0
  fi

  TS=$(date '+%Y-%m-%d_%H-%M-%S')
  LOG="$LOG_DIR/run_${TS}.log"
  echo "[$(date)] iteration $i -> $LOG"

  if claude -p --dangerously-skip-permissions "$PROMPT" > "$LOG" 2>&1; then
    echo "[$(date)] iteration $i ok, sleeping ${SUCCESS_SLEEP_SECONDS}s"
    sleep "$SUCCESS_SLEEP_SECONDS"
  else
    rc=$?
    echo "[$(date)] iteration $i failed rc=$rc, retrying in ${RETRY_SLEEP_SECONDS}s"
    sleep "$RETRY_SLEEP_SECONDS"
  fi
done

echo "[$(date)] hit MAX_ITERATIONS=$MAX_ITERATIONS, stopping"
