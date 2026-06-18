"""Generate a self-contained HTML project report.

Reads:
  - MLflow tracking server (gracefully skipped if mlflow is not installed)
  - project_management/current_project_status.md
  - project_management/key_findings_log.md
  - outputs/<domain>/evaluation/metrics_boxplot_*.png (embedded as base64)

Output:
  project_management/report.html  (single file, no external dependencies)

Usage:
  python project_management/generate_report.py [--entries N]
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PM_DIR = REPO_ROOT / "project_management"
REPORT_PATH = PM_DIR / "report.html"

DOMAINS = ["arctic_domain", "amazon_domain", "rangeland_domain", "multi_domain"]

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def read_text_block(path: Path, start_header: str, end_header: Optional[str] = None) -> str:
    """Return the text between start_header and end_header (exclusive).

    If end_header is None, reads to end of file.
    Returns empty string if start_header is not found.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

    start = text.find(f"\n{start_header}")
    if start == -1:
        start = text.find(start_header)
        if start == -1:
            return ""
    start = text.find("\n", start) + 1  # skip past the header line itself

    if end_header:
        end = text.find(f"\n{end_header}", start)
        if end == -1:
            end = len(text)
    else:
        end = len(text)

    return text[start:end].strip()


def parse_md_table(path: Path, after_header: str) -> list[dict]:
    """Parse the first Markdown pipe-table that appears after after_header.

    Returns a list of dicts keyed by column headers.
    Returns [] if the file or table is not found.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    start = text.find(after_header)
    if start == -1:
        return []
    text = text[start:]

    rows = []
    headers: list[str] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if not cells:
            continue
        if not headers:
            headers = cells
            in_table = True
            continue
        # separator row (---|---|...)
        if all(re.match(r"^-+$", c.replace(" ", "")) for c in cells):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def parse_md_entries(path: Path, n: int = 5) -> list[dict]:
    """Return the last n entries from key_findings_log.md.

    Each entry starts with a '## ' heading. Returns [] if none found.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    raw_entries = re.split(r"\n(?=## )", text)
    entries = []
    for chunk in raw_entries:
        if not chunk.strip().startswith("## "):
            continue
        first_line = chunk.splitlines()[0].lstrip("# ").strip()
        body = "\n".join(chunk.splitlines()[1:]).strip()
        entries.append({"title": first_line, "body": body})

    return entries[-n:] if entries else []


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def png_to_base64(path: Path) -> Optional[str]:
    """Return a data URI for a PNG, or None if the file does not exist."""
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def make_metrics_figure(domain: str, metrics_csv: Path) -> Optional[str]:
    """Generate a matplotlib NSE boxplot from metrics.csv and return as base64 PNG.

    Returns None if matplotlib is unavailable or the CSV is empty/missing.
    """
    if not metrics_csv.exists():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib or pandas not available; skipping figure generation")
        return None

    try:
        df = pd.read_csv(metrics_csv)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("Could not read %s: %s", metrics_csv, exc)
        return None

    if df.empty or "NSE" not in df.columns or "target" not in df.columns:
        return None

    targets = df["target"].unique().tolist()
    data = [df[df["target"] == t]["NSE"].dropna().tolist() for t in targets]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=targets, patch_artist=True)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title(f"{domain} — NSE distribution across test units")
    ax.set_ylabel("NSE")
    ax.set_ylim(-1, 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    data_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    return data_uri


# ---------------------------------------------------------------------------
# MLflow helper
# ---------------------------------------------------------------------------


def load_mlflow_runs(domains: list[str]) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Query MLflow for the last 20 runs across all given experiments.

    Returns an empty DataFrame if mlflow is not installed, the tracking
    server is unreachable, or no experiments exist yet.
    """
    try:
        import mlflow
        import pandas as pd
    except ImportError:
        return _empty_runs_df()

    try:
        uri = str(REPO_ROOT / "mlruns")
        mlflow.set_tracking_uri(uri)
        existing = [
            e.name
            for e in mlflow.search_experiments()
            if e.name in domains
        ]
        if not existing:
            return _empty_runs_df()
        df = mlflow.search_runs(
            experiment_names=existing,
            order_by=["start_time DESC"],
            max_results=20,
        )
        return df if not df.empty else _empty_runs_df()
    except Exception as exc:
        logger.warning("MLflow query failed: %s", exc)
        return _empty_runs_df()


def _empty_runs_df() -> "pd.DataFrame":  # type: ignore[name-defined]
    try:
        import pandas as pd
        return pd.DataFrame()
    except ImportError:
        return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f5f5; color: #222; }
header { background: #1a3a5c; color: #fff; padding: 1rem 2rem; }
header h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
header p { margin: 0; font-size: 0.85rem; opacity: 0.8; }
main { max-width: 1100px; margin: 1.5rem auto; padding: 0 1rem; }
section { background: #fff; border-radius: 6px; padding: 1.25rem 1.5rem;
          margin-bottom: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
h2 { margin-top: 0; font-size: 1.1rem; border-bottom: 2px solid #e0e0e0;
     padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem; margin: 1rem 0 0.4rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th { background: #f0f4f8; text-align: left; padding: 6px 10px; }
td { padding: 5px 10px; border-bottom: 1px solid #eee; }
.tag-complete  { background: #d4edda; color: #155724; border-radius: 3px; padding: 2px 7px; }
.tag-active    { background: #fff3cd; color: #856404; border-radius: 3px; padding: 2px 7px; }
.tag-notstart  { background: #e2e3e5; color: #383d41; border-radius: 3px; padding: 2px 7px; }
.nse-good  { background: #d4edda; }
.nse-ok    { background: #fff3cd; }
.nse-bad   { background: #f8d7da; }
.needs-review { background: #fff3cd; border-left: 4px solid #ffc107;
                padding: 0.5rem 0.75rem; margin: 0.5rem 0; font-size: 0.85rem; }
pre { background: #f8f8f8; padding: 0.75rem; border-radius: 4px;
      font-size: 0.82rem; overflow-x: auto; }
.placeholder { color: #888; font-style: italic; }
footer { text-align: center; font-size: 0.75rem; color: #999; padding: 1.5rem; }
img { max-width: 100%; height: auto; display: block; margin: 0.75rem 0; }
"""


def _stage_tag(stage: str) -> str:
    if stage == "Complete":
        return f'<span class="tag-complete">{stage}</span>'
    if stage == "Not Started":
        return f'<span class="tag-notstart">{stage}</span>'
    return f'<span class="tag-active">{stage}</span>'


def _nse_cell(val: str) -> str:
    try:
        v = float(val)
        cls = "nse-good" if v >= 0.7 else ("nse-ok" if v >= 0.4 else "nse-bad")
        return f'<td class="{cls}">{val}</td>'
    except (ValueError, TypeError):
        return f"<td>{val}</td>"


def _runs_table_html(df: "pd.DataFrame") -> str:  # type: ignore[name-defined]
    if df is None or (hasattr(df, "empty") and df.empty):
        return '<p class="placeholder">No MLflow runs recorded yet. '  \
               "Run the pipeline and complete Stage 2 MLflow integration.</p>"

    keep_cols = [c for c in df.columns if not c.startswith("params.paths")
                 and not c.startswith("params.gcs")]
    display_cols = (
        ["tags.exp_id", "tags.mlflow.runName", "start_time", "status"]
        + [c for c in keep_cols if c.startswith("metrics.")]
    )
    display_cols = [c for c in display_cols if c in df.columns][:12]

    rows = ""
    for _, row in df.head(10).iterrows():
        cells = ""
        for col in display_cols:
            val = str(row.get(col, ""))
            if "NSE" in col:
                cells += _nse_cell(val)
            else:
                cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c.replace('tags.','').replace('metrics.','')}</th>"
                      for c in display_cols)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _findings_html(entries: list[dict]) -> str:
    if not entries:
        return '<p class="placeholder">No key findings logged yet.</p>'
    parts = []
    for e in reversed(entries):
        body_html = e["body"].replace(
            "NEEDS HUMAN REVIEW",
            '<span class="needs-review">⚠ NEEDS HUMAN REVIEW</span>',
        )
        parts.append(f"<h3>{e['title']}</h3><pre>{body_html}</pre>")
    return "\n".join(parts)


def _domain_rows_html(rows: list[dict]) -> str:
    if not rows:
        return '<p class="placeholder">Domain table not found in current_project_status.md.</p>'
    cells = ""
    for r in rows:
        stage = r.get("stage", "")
        active = r.get("active", "")
        active_tag = '<span class="tag-active">Yes</span>' if active == "Yes" else "No"
        cells += (
            f"<tr><td>{r.get('domain','')}</td>"
            f"<td>{_stage_tag(stage)}</td>"
            f"<td>{active_tag}</td>"
            f"<td>{r.get('notes','')}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Domain</th><th>Stage</th>"
        "<th>Active</th><th>Notes</th></tr></thead>"
        f"<tbody>{cells}</tbody></table>"
    )


def _open_questions_html(rows: list[dict]) -> str:
    if not rows:
        return '<p class="placeholder">No open questions.</p>'
    cells = "".join(
        f"<tr><td>{r.get('#','')}</td><td>{r.get('Question','')}</td>"
        f"<td>{r.get('Raised','')}</td><td>{r.get('Status','')}</td></tr>"
        for r in rows
    )
    return (
        "<table><thead><tr><th>#</th><th>Question</th>"
        "<th>Raised</th><th>Status</th></tr></thead>"
        f"<tbody>{cells}</tbody></table>"
    )


def _metrics_section_html(domain: str) -> str:
    eval_dir = REPO_ROOT / "outputs" / domain / "evaluation"
    parts = []

    # Embed any saved boxplot figures (domain-agnostic; Arctic emits per-SSP files,
    # Amazon/Rangeland emit a single one).
    for png_path in sorted(eval_dir.glob("*boxplot*.png")):
        uri = png_to_base64(png_path)
        if uri:
            parts.append(f"<h3>{png_path.stem}</h3><img src='{uri}' alt='{png_path.stem}'>")

    # Fall back to generating an NSE boxplot from metrics.csv.
    if not parts:
        fig_uri = make_metrics_figure(domain, eval_dir / "metrics.csv")
        if fig_uri:
            parts.append(f"<h3>NSE distribution</h3><img src='{fig_uri}' alt='NSE boxplot'>")

    return "\n".join(parts) if parts else (
        '<p class="placeholder">No evaluation figures found. Run 04_evaluate.py first.</p>'
    )


def render_html(ctx: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metrics_sections = ""
    for domain in DOMAINS:
        section_html = _metrics_section_html(domain)
        metrics_sections += f"<h3>{domain}</h3>{section_html}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Domain Time Series — Project Report</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>Multi-Domain Time Series Prediction</h1>
  <p>Generated {ts} &nbsp;|&nbsp; Active: {ctx['active_domain']} &nbsp;|&nbsp; Stage: {ctx['active_stage']}</p>
</header>
<main>

<section>
  <h2>Project Status</h2>
  <h3>Current</h3>
  <pre>{ctx['current_block']}</pre>
  <h3>Next</h3>
  <pre>{ctx['next_block']}</pre>
</section>

<section>
  <h2>Domain Progress</h2>
  {ctx['domain_table']}
</section>

<section>
  <h2>Experiment Timeline (MLflow)</h2>
  {ctx['runs_table']}
</section>

<section>
  <h2>Evaluation Metrics</h2>
  {metrics_sections}
</section>

<section>
  <h2>Key Findings</h2>
  {ctx['findings']}
</section>

<section>
  <h2>Open Questions</h2>
  {ctx['open_questions']}
</section>

</main>
<footer>
  Regenerate: <code>python project_management/generate_report.py</code>
  &nbsp;|&nbsp;
  MLflow UI: <code>mlflow ui --backend-store-uri mlruns/</code>
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(n_entries: int = 5) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    status_path = PM_DIR / "current_project_status.md"
    findings_path = PM_DIR / "key_findings_log.md"

    # -- domain progress table
    domain_rows = parse_md_table(status_path, "## Section 1")
    active_domain = next((r["domain"] for r in domain_rows if r.get("active") == "Yes"), "—")
    active_stage = next((r["stage"] for r in domain_rows if r.get("active") == "Yes"), "—")

    # -- diary blocks
    current_block = read_text_block(status_path, "### CURRENT", "### NEXT") or "—"
    next_block = read_text_block(status_path, "### NEXT", "### PAST") or "—"

    # -- open questions
    oq_rows = parse_md_table(status_path, "## Section 3")

    # -- MLflow runs
    runs_df = load_mlflow_runs(DOMAINS)

    # -- key findings
    findings_entries = parse_md_entries(findings_path, n=n_entries)

    ctx = {
        "active_domain": active_domain,
        "active_stage": active_stage,
        "current_block": current_block,
        "next_block": next_block,
        "domain_table": _domain_rows_html(domain_rows),
        "runs_table": _runs_table_html(runs_df),
        "findings": _findings_html(findings_entries),
        "open_questions": _open_questions_html(oq_rows),
    }

    html = render_html(ctx)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=5,
                        help="Number of key findings entries to include (default: 5)")
    args = parser.parse_args()
    main(n_entries=args.entries)
