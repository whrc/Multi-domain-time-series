# Session Start Protocol

<!-- Human-maintained. Claude reads only. -->

## Purpose

Defines what context to load at the start of a conversation. Not every tier
runs every session — loading is triggered by the user's first message.

---

Always use plan-mode first.

## Tiers

| Tier | Trigger condition | What to read |
| --- | --- | --- |
| T1 (always) | Every conversation | `project_management/proj_mgmt.md` |
| T2 | Task involves any file in `domains/<domain>/`, or user mentions an experiment, metric, or finding | `project_management/current_project_status.md` |
| T3 | Task involves running code, or user mentions MLflow / run_id / exp_id | `project_management/environment_spec.md` |
| T4 | User says "review code" or "check code quality" | `project_management/protocols/code_review.md` |

Triggers are evaluated against the **user's first message**, not subjective judgment.
Before starting work, state which tiers were loaded and why — one line.

---

## What NOT to do at session start

- Do not run the pipeline without human confirmation
- Do not assume experiment_log is up to date — MLflow is SSOT, not memory
