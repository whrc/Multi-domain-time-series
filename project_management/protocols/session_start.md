# Session Start Protocol

<!-- Human-maintained. Claude reads only. -->

Always start in plan mode. Load context by tier, judged on the user's **first message**, and state in one line which tiers you loaded and why.

| Tier | Trigger | Read |
| --- | --- | --- |
| T1 | Always | `project_management/proj_mgmt.md` |
| T2 | Touches `domains/<domain>/`, or mentions an experiment, metric or finding | `project_management/current_project_status.md` |
| T3 | Running code, or mentions MLflow / run_id / exp_id | `project_management/environment_spec.md` |
| T4 | "review code" / "check code quality" | `project_management/protocols/code_review.md` |

- Don't run the pipeline without human confirmation.
- Don't assume `key_findings_log.md` is current — MLflow is the SSOT.
