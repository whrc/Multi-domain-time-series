#!/bin/bash
# Resilient runner for 01_preprocess.py: relaunches it until it exits successfully.
#
# Why this exists: a long-running background python process can be killed unpredictably —
# observed causes so far: this project's Claude Code tool sessions (anywhere from ~1 to ~25
# minutes, unrelated to code bugs, memory, or sandbox settings), and running on battery
# power instead of AC (macOS applies background-process throttling that caffeinate's -s
# flag doesn't cover, since -s is explicitly AC-only). Safe to relaunch indefinitely because
# 01_preprocess.py caches each grid's derived pass-1 summary to disk
# (outputs/arctic_domain/preprocessed/.grid_summary_cache/), so every restart resumes
# instead of re-fetching from GCS. If you're running this on infrastructure that doesn't
# exhibit random kills (e.g. a real terminal on AC power, tmux/SSH on the VM), you likely
# don't need this wrapper at all -- just run 01_preprocess.py directly.
#
# Usage (any 01_preprocess.py flag is forwarded as-is):
#   domains/arctic_domain/run_preprocess_resilient.sh --train-size 500000
#   domains/arctic_domain/run_preprocess_resilient.sh --train-size 2000000
#   domains/arctic_domain/run_preprocess_resilient.sh                       # full uncapped run
#
# Run in the background yourself if you don't want to hold a terminal open, e.g.:
#   nohup domains/arctic_domain/run_preprocess_resilient.sh --train-size 500000 &
#
# Progress: outputs/arctic_domain/supervisor.log (attempt history) and
# outputs/arctic_domain/preprocess_run.log (the wrapped script's own output).

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1  # repo root (this script lives in domains/arctic_domain/)

PP_LOG="outputs/arctic_domain/preprocess_run.log"
SUP_LOG="outputs/arctic_domain/supervisor.log"
MAX_ATTEMPTS=300

mkdir -p outputs/arctic_domain
echo "$(date '+%Y-%m-%d %H:%M:%S') supervisor started: 01_preprocess.py $*" >> "$SUP_LOG"

# If a manually-launched run is already in progress, wait for it instead of racing a
# duplicate process against it.
while pgrep -f "01_preprocess.py" > /dev/null; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') existing run detected, waiting for it to end..." >> "$SUP_LOG"
  sleep 15
done

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  cached=$(ls outputs/arctic_domain/preprocessed/.grid_summary_cache/*.pkl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt starting (cached_grids=$cached)" >> "$SUP_LOG"

  # Blocking call (no trailing &): waits here until this exits or is killed, then loops
  # to relaunch. caffeinate holds sleep-prevention assertions only for this child's life.
  caffeinate -i -s .venv/bin/python domains/arctic_domain/01_preprocess.py "$@" >> "$PP_LOG" 2>&1
  rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') COMPLETED SUCCESSFULLY after $attempt attempt(s)" >> "$SUP_LOG"
    exit 0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt ended with exit code $rc — retrying" >> "$SUP_LOG"
  sleep 2
done

echo "$(date '+%Y-%m-%d %H:%M:%S') GAVE UP after $MAX_ATTEMPTS attempts without success" >> "$SUP_LOG"
exit 1
