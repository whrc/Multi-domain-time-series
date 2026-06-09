# Code Review Protocol

<!-- Human-maintained. Claude reads only. -->

## Rule

All Claude-generated code must be fully tested before execution.
No code is committed until review passes.

## Process

1. Invoke the `superpowers:requesting-code-review` skill before committing
   any generated code — including "small" changes.

2. For high-stakes changes (new pipeline scripts, generate_report.py
   modifications, config schema changes), also run an adversarial review:
   spawn a fresh agent with no prior context and ask it to independently
   critique the implementation against the plan and CLAUDE.md rules.

3. Address all findings before committing. Push fixes as additional commits —
   do not rewrite history.

## What counts as "tested"

- The code runs without exception on the intended input
- Edge cases that can realistically occur are handled (missing files,
  empty DataFrames, MLflow experiment not yet created)
- No silent failures — errors raise specific exceptions with clear messages
