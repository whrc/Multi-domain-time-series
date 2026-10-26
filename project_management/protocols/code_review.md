# Code Review Protocol

<!-- Human-maintained. Claude reads only. -->

No Claude-generated code is executed until fully tested, or committed until review passes.

1. Run the `superpowers:requesting-code-review` skill before every commit, however small.
2. High-stakes changes (new pipeline scripts, `generate_report.py`, config schema): also have a fresh, context-free agent adversarially critique the code against the plan and `CLAUDE.md`.
3. Fix all findings as new commits; never rewrite history.

**"Tested" means:** runs without exception on the intended input; realistic edge cases handled (missing files, empty DataFrames, missing MLflow experiment); errors raise specific exceptions with clear messages.
