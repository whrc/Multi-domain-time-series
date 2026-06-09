# Git Workflow Protocol

<!-- Human-maintained. Claude reads only. -->

## When to branch

Create a branch for any change to executable code (`.py`, `.yaml`, `.ipynb`)
beyond a single trivial line.

Exception: direct-to-main is allowed only for single-word typo fixes in
documentation files.

## Branch naming

```
<type>/<short-kebab-description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `experiment`

Examples: `feat/amazon-preprocess`, `fix/arctic-scaler-leak`,
`experiment/transformer-lr-sweep`

## Commit conventions (Conventional Commits)

```
<type>(<scope>): <imperative summary>   # max 72 chars
```

Scope = domain name or module: `arctic`, `amazon`, `config`, `models`,
`proj-mgmt`

## Before opening a PR

- If main has advanced since the branch was created: `git rebase main`
- If the branch is already pushed to remote: `git merge main` (preserves
  shared history — do not rebase a pushed branch)
- Run `superpowers:requesting-code-review` and resolve all findings

## PR

Claude uses `gh pr create` with a HEREDOC body:
- Title: Conventional Commit style, ≤72 chars
- Body: What changed / Why / How to verify

**Claude does not merge.** The human reviews and merges.

## Handling review feedback

- Read all review comments before touching any code
- Address one comment per commit (not one mega-fix commit)
- If a comment is unclear or technically questionable, ask for clarification —
  do not implement blindly (see `superpowers:receiving-code-review`)
- Push additional commits to the same branch — do not rewrite history
- Use `--force-with-lease` only if explicitly requested by the human

## Merge strategy

- Feature branches: **squash merge** (clean main history)
- Hotfixes: **merge commit** (preserves timeline)
- Never force-push to main

## Cleanup (after human confirms merge)

```bash
git checkout main
git pull
git branch -d <branch>
git push origin --delete <branch>
```

Claude performs cleanup only after explicit human confirmation that the PR
was merged.

## Blocked states

If a hook fails, a conflict blocks merge, or CI fails — investigate root
cause. Never use `--no-verify` or `--force` without explicit human
instruction.
