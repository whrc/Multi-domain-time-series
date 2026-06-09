# Git Workflow Protocol

<!-- Human-maintained. Claude reads only. -->

- Only send PR when human requests

- **Claude does not merge.** The human reviews and merges.

- Claude performs cleanup only after explicit human confirmation that the PR
was merged.

- To clean up, check local and remote branch status, delete merged (include squash merge), and rebase current working branch if necessary.

- Do not use stash

- When waiting a PR, use another branch off the PR-branch if you want to keep developing on top of it.