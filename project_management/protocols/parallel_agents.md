# Parallel Agents Protocol

<!-- Human-maintained. Claude reads only. -->

## When to use multiple agents

- 2+ independent tasks with no shared state (different files, different
  domains, different pipeline stages)
- A task that benefits from a fresh perspective: adversarial review,
  independent validation of an implementation
- Exploration across multiple code areas simultaneously

## When NOT to use multiple agents

- Task B requires output from Task A — run sequentially, wait for each result
- Tasks share a file being modified — concurrent writes cause conflicts
- Single-file changes where spawning overhead exceeds the benefit
- Already inside a spawned agent — do not nest agents