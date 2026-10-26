# Git Workflow Protocol

<!-- Human-maintained. Claude reads only. -->

- Open a PR only when the human asks.
- Claude never merges; the human reviews and merges.
- Clean up only after the human confirms the merge: check local/remote branches, delete merged ones (including squash-merged), rebase the working branch if needed.
- Never use `git stash`.
- To keep developing while a PR waits, branch off the PR branch.
