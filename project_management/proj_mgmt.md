# Project Management — Index

<!-- Read first every conversation, then follow protocols/session_start.md.
Navigation only: the one allowed edit is adding a row when a new management file is created. -->

| Need | File |
| --- | --- |
| Current state, diary, open questions | `current_project_status.md` |
| Past experiment findings | `key_findings_log.md` |
| Environment, VMs, compute placement | `environment_spec.md` |
| Session start | `protocols/session_start.md` |
| What to log per experiment | `protocols/log_experiments.md` |
| Code review before commit | `protocols/code_review.md` |
| Git branching, PRs, cleanup | `protocols/git_workflow.md` |
| Multi-agent work | `protocols/parallel_agents.md` |
| HTML report | `generate_report.py` |

## Single Source of Truth

| Data | Authoritative | Read-only mirrors |
| --- | --- | --- |
| Experiment params + metrics | MLflow (`mlruns/`) | `key_findings_log.md` (interpretation) |
| Domain stage | `current_project_status.md` Section 1 | `CLAUDE.md` (pointer) |
| Hyperparams + paths | `config/<domain>.yaml` | pipeline scripts, `generate_report.py` |
| Interpretation of results | `key_findings_log.md` | HTML report |
| Env / hardware | `environment_spec.md` | protocols |

If files disagree, the authoritative source wins; update the mirror.

## Ownership

| File | Who updates |
| --- | --- |
| `proj_mgmt.md`, `protocols/*.md` | Human only (Claude may add nav rows here) |
| `current_project_status.md` | Claude (diary + Section 1) |
| `key_findings_log.md` | Claude drafts "What happened"; human writes "Interpretation" |
| `environment_spec.md` | Both (human: VM/GCS; Claude: MLflow URI) |
| `generate_report.py` | Claude (code review before committing) |
