#!/bin/bash
LOG=/Users/daxmavy/Desktop/attack-llm-judge/.runpod_poll/poll.log
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes"
TARGET="z0zrhzy0tdquav-64410f17@ssh.runpod.io"
KEY="$HOME/.ssh/id_ed25519"
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  OUT=$(ssh $SSH_OPTS -i "$KEY" "$TARGET" 'echo POD_ALIVE; uptime; df -h /workspace 2>/dev/null | tail -1' 2>&1)
  RC=$?
  if echo "$OUT" | grep -q "POD_ALIVE"; then
    echo "[$TS] UP rc=$RC :: $(echo "$OUT" | tr '\n' ' ' | cut -c1-300)" >> "$LOG"
    echo "[$TS] *** POD IS UP ***" >> "$LOG"
  else
    SHORT=$(echo "$OUT" | tr '\n' ' ' | cut -c1-200)
    echo "[$TS] DOWN rc=$RC :: $SHORT" >> "$LOG"
  fi
  sleep 300
done
