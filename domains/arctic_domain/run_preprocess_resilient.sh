#!/bin/bash
# Resilient runner for 01_preprocess.py: relaunches it until it exits successfully.
#
# Why this exists: a long-running background python process can be killed unpredictably in
# this project's Claude Code tool sessions (exit code 137/SIGKILL, roughly every 15-25
# minutes) for a cause that's still unidentified as of 2026-07-03 -- ruled out so far: code
# bugs, memory pressure, disk space, and running on battery vs AC power (confirmed crashing
# continuously for 6+ hours on AC power alone, so that's not it either). Safe to relaunch
# indefinitely regardless of the cause because 01_preprocess.py caches each grid's derived
# pass-1 summary and pass-2 selection to disk (outputs/arctic_domain/preprocessed/
# .grid_pass1_summary_cache/ and .grid_pass2_records_cache/), so every restart resumes instead of
# re-fetching from GCS. If you're running this on infrastructure that doesn't exhibit random
# kills (e.g. a real terminal, tmux/SSH on the VM), you likely don't need this wrapper at all
# -- just run 01_preprocess.py directly.
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
  cached=$(ls outputs/arctic_domain/preprocessed/.grid_pass1_summary_cache/*.pkl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt starting (cached_grids=$cached)" >> "$SUP_LOG"

  # Blocking call (no trailing &): waits here until this exits or is killed, then loops
  # to relaunch. caffeinate holds sleep-prevention assertions only for this child's life —
  # macOS-only, so on Linux (e.g. the GCE VMs, which don't sleep) run the command directly.
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
  # rc 2 (CONFIG_MISMATCH_EXIT_CODE in 01_preprocess.py) means a precondition/config error —
  # e.g. an existing val.pkl/test.pkl sidecar that doesn't match this run's config, or a
  # --grids override too small to populate every split. Unlike a transient GCS fetch failure,
  # rerunning the exact same command can never succeed without a human changing something
  # first, so retrying would just burn up to MAX_ATTEMPTS doing nothing — stop immediately.
  if [ "$rc" -eq 2 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') attempt $attempt ended with exit code 2 (config/precondition error — will not clear up on retry, see $PP_LOG for details) — stopping" >> "$SUP_LOG"
    exit 2
  fi
  # rc 137/143 (SIGKILL/SIGTERM, i.e. 128+signal) match this script's known external-kill
  # cause and are expected to clear up on retry. Any other code (a real Python traceback,
  # a config/credential error, ...) still gets retried the same way, but is logged distinctly
  # so a human scanning this log later can tell "known issue" apart from "possible real bug,
  # worth investigating" instead of every failure looking identical.
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
