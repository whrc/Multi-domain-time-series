# Paper — Working Directory

This folder is where we write the manuscript. Plain Markdown, version-controlled
alongside the code, so every technical claim can be traced back to an actual
config, run, or output file instead of drifting from memory.

## Files

| File | Purpose |
| --- | --- |
| `paper_description.md` | This file. Provides an overview of the paper's structure and workflow. |
| `manuscript.md` | The actual draft prose, growing section by section as we agree on content. |

## Workflow

- Human makes a skeleton outline of the paper with section headings and keywords/bullets for content in each section.
- Human and AI collaborate to fill in the outline.
- The language should be clear, concise, and accessible to a broad audience, while still being technically accurate. Don't use overly complex jargon or obscure terminology unless necessary, and provide explanations for any technical terms that are used. Use natural human-like phrasing and sentence structure, avoiding overly formal or robotic language and excessive use of passive voice and complex sentence structures and notations including colons, and emdashes.
- Where citations is required just leave as (cite) for now, Human will fill in the actual citation later.

## Rendering a preview

Not required to write — only useful if you want to see it as a formatted
PDF/Word doc rather than raw Markdown. Requires Pandoc (`brew install pandoc`):

```bash
pandoc manuscript.md -o preview.pdf
```

(Once real citations replace the `(cite)` placeholders and a `references.bib` exists, add `--citeproc --bibliography=references.bib` back to this command.)

`preview.pdf` is a local convenience file, not committed to git (see
`.gitignore`).
