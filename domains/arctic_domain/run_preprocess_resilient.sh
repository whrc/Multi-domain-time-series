#!/bin/bash
# Relaunch 01_preprocess.py until it succeeds (for environments that randomly SIGKILL long
# background jobs). Restarts resume from 01_preprocess.py's per-grid pass-1/pass-2 caches.
# Not needed in a stable terminal/tmux; just run 01_preprocess.py directly.
#
# Usage (flags are forwarded to 01_preprocess.py):
#   nohup domains/arctic_domain/run_preprocess_resilient.sh --train-size 500000 &
#
# Logs: outputs/arctic_domain/supervisor.log (attempts), outputs/arctic_domain/preprocess_run.log (script output).

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1  # repo root (this script lives in domains/arctic_domain/)

PP_LOG="outputs/arctic_domain/preprocess_run.log"
SUP_LOG="outputs/arctic_domain/supervisor.log"
MAX_ATTEMPTS=300

mkdir -p outputs/arctic_domain
echo "$(date '+%Y-%m-%d %H:%M:%S') supervisor started: 01_preprocess.py $*" >> "$SUP_LOG"

# Wait for any already-running 01_preprocess.py instead of racing it.
while pgrep -f "01_preprocess.py" > /dev/null; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') existing run detected, waiting for it to end..." >> "$SUP_LOG"
  sleep 15
done

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  cached=$(ls outputs/arctic_domain/preprocessed/.grid_pass1_summary_cache/*.pkl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt starting (cached_grids=$cached)" >> "$SUP_LOG"

  # caffeinate (macOS only) prevents sleep while the child runs.
  if command -v caffeinate > /dev/null 2>&1; then
    caffeinate -i -s .venv/bin/python domains/arctic_domain/01_preprocess.py "$@" >> "$PP_LOG" 2>&1
  else
    .venv/bin/python domains/arctic_domain/01_preprocess.py "$@" >> "$PP_LOG" 2>&1
  fi
  rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') COMPLETED SUCCESSFULLY after $attempt attempt(s)" >> "$SUP_LOG"
    exit 0
  fi
  # rc 2 = CONFIG_MISMATCH_EXIT_CODE in 01_preprocess.py: retrying can't help, so stop.
  if [ "$rc" -eq 2 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt ended with exit code 2 (config/precondition error — will not clear up on retry, see $PP_LOG for details) — stopping" >> "$SUP_LOG"
    exit 2
  fi
  # rc 137/143 = SIGKILL/SIGTERM (expected external kill); other codes are retried but flagged.
  if [ "$rc" -eq 137 ] || [ "$rc" -eq 143 ]; then
    note="killed by signal (matches the known external-kill pattern this script exists for)"
  else
    note="exited with a non-signal code — does NOT match the known kill pattern, may be a real bug"
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt ended with exit code $rc ($note) — retrying" >> "$SUP_LOG"
  sleep 2
done

echo "$(date '+%Y-%m-%d %H:%M:%S') GAVE UP after $MAX_ATTEMPTS attempts without success" >> "$SUP_LOG"
exit 1
