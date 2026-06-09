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

## How to spawn agents in parallel

Issue multiple `Agent` tool calls in a **single response message**.
Each agent starts cold with no conversation history — every prompt must be
fully self-contained.

## Available agent types

| Type | Use for |
| --- | --- |
| `Explore` | Read-only search: locating files, grepping symbols, understanding patterns |
| `Plan` | Architecture and design before implementation |
| `claude` | Writing or editing code |
| `superpowers:code-reviewer` | Code review against plan and CLAUDE.md standards |

## Writing prompts for subagents

Each prompt must include:
1. What to accomplish and why
2. Relevant file paths and code snippets needed to reason about the task
3. Constraints (e.g., "do not modify existing functions")
4. What NOT to do
5. Expected output format

Agents cannot ask follow-up questions. Give enough context for judgment calls.

## Coordination pattern for this project

For a new domain (e.g., Amazon pipeline):

1. **Explore phase (parallel):** spawn two `Explore` agents simultaneously —
   one researches the data schema and GCS bucket layout, the other reads the
   existing Arctic pipeline as a reference pattern
2. **Design phase:** synthesize results, then spawn one `Plan` agent to design
   the domain pipeline
3. **Implementation phase:** after plan approval, spawn parallel `claude`
   agents for independent files (e.g., `01_preprocess.py` and
   `config/amazon_domain.yaml` if they do not overlap in content)
4. **Review phase:** after each logical chunk, invoke
   `superpowers:code-reviewer` before moving to the next chunk

## Adversarial review pattern

To get an independent critique of an implementation:

```
Agent(
  subagent_type="claude",
  prompt="You are reviewing the following implementation for correctness,
          SSOT compliance, and CLAUDE.md adherence. You have NOT seen the
          prior conversation. Critique only — do not fix. [paste code + plan]"
)
```

Spawn this agent with no shared context from the main conversation.
