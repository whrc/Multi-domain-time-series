# Project Management — Master Index

<!-- Claude Instructions ─────────────────────────────────────────────────────
READ THIS FILE FIRST in every new conversation (T1, always).
Then follow session_start.md to determine which additional files to load.
Do NOT write content into this file — it is a pure navigation index.
The only allowed edit: adding a new row to the Navigation table when a new
management file is created.
──────────────────────────────────────────────────────────────────────────── -->

## Navigation

| Need | File |
| --- | --- |
| Where is the project right now? | `current_project_status.md` |
| What did past experiments find? | `key_findings_log.md` |
| Reproduce the environment | `environment_spec.md` |
| Start a new coding session | `protocols/session_start.md` |
| Log results after evaluation | `protocols/experiment_workflow.md` |
| Review generated code before committing | `protocols/code_review.md` |
| Git branching, PR, merge, cleanup | `protocols/git_workflow.md` |
| Run tasks with multiple agents | `protocols/parallel_agents.md` |
| Add a new domain | `protocols/new_domain_checklist.md` |
| Generate or refresh the HTML report | `generate_report.py` |

---

## Single Source of Truth (SSOT)

| Data | Authoritative source | Mirrors / consumers (read-only) |
| --- | --- | --- |
| Experiment params + metrics | MLflow (`mlruns/`) | `key_findings_log.md` (interpretation only) |
| Domain stage | `current_project_status.md` DOMAINS table | `CLAUDE.md` Current Stage (pointer, not definition) |
| Hyperparams + paths | `config/<domain>.yaml` | pipeline scripts, `generate_report.py` |
| Human interpretation of results | `key_findings_log.md` | HTML report |
| Env / hardware facts | `environment_spec.md` | all protocols |

**Conflict resolution:** if two files disagree, the authoritative source wins. The mirror is stale — update it to match.

---

## File Ownership

| File | Who may update |
| --- | --- |
| `proj_mgmt.md` | Human only (add nav rows when new files created) |
| `current_project_status.md` | Claude updates diary + Domains table |
| `key_findings_log.md` | Claude drafts "What happened"; human fills "Interpretation" |
| `environment_spec.md` | Both (human fills hardware; Claude may update MLflow URI) |
| `protocols/*.md` | Human only |
| `generate_report.py` | Claude (run code review before committing any change) |

---

## New Domain

For project-management steps when adding a new domain, see
`protocols/new_domain_checklist.md`.
