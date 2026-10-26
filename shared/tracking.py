"""MLflow helpers for 02/03/04 (schema: project_management/protocols/log_experiments.md).

Local file store, gated by `mlflow.enabled`; 02 opens the run, 03/04 reopen it via the
`best_model.run_id` sidecar.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# MLflow 3.x deprecates the file store; the project uses one at mlruns/.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

_CODES = {"arctic_domain": "AR", "amazon_domain": "AM", "rangeland_domain": "RG"}
_EXCLUDE_TOP = ("paths", "gcs", "mlflow")
_MAX_VALUE_LEN = 250
_SUMMARY_METRICS = ("NSE", "KGE", "RMSE", "PBIAS")


def domain_code(domain: str) -> str:
    return _CODES.get(domain, domain[:2].upper())


def setup(cfg: dict) -> bool:
    """Point MLflow at this domain's store/experiment; return False if tracking is disabled."""
    mcfg = cfg.get("mlflow")
    if not mcfg or not mcfg.get("enabled", False):
        return False
    uri = Path(mcfg["tracking_uri"]).resolve().as_uri()  # file:// URI (Windows-safe)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(mcfg["experiment"])
    logger.info("MLflow tracking at %s (experiment=%s)", uri, mcfg["experiment"])
    return True


def flatten_params(cfg: dict) -> dict:
    """Flatten config to dotted params (excluding paths/gcs/mlflow), stringifying and truncating long values."""
    out: dict[str, object] = {}

    def _recurse(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            if prefix == "" and k in _EXCLUDE_TOP:
                continue
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                _recurse(v, key + ".")
            else:
                s = v if isinstance(v, (int, float, bool)) or v is None else str(v)
                if isinstance(s, str) and len(s) > _MAX_VALUE_LEN:
                    s = s[:_MAX_VALUE_LEN]
                out[key] = s

    _recurse(cfg)
    return out


@contextmanager
def training_run(cfg: dict, domain: str, enabled: bool):
    """Open the training run and log params/tags; yields the run, or None if disabled."""
    if not enabled:
        yield None
        return
    with mlflow.start_run() as run:
        mlflow.log_params(flatten_params(cfg))
        mlflow.set_tag("arch", cfg["model"]["architecture"])
        mlflow.set_tag("mode", cfg.get("mode", ""))
        mlflow.set_tag("exp_id", f"{domain_code(domain)}-{run.info.run_id[:8]}")
        yield run


@contextmanager
def resume_run(run_id: str | None):
    """Reopen the training run for 03/04; yields whether a run is active."""
    if not run_id:
        yield False
        return
    with mlflow.start_run(run_id=run_id):
        yield True


def read_run_id(sidecar: Path) -> str | None:
    """Read the run_id sidecar written by 02, or None (with a warning)."""
    if sidecar.exists():
        return sidecar.read_text().strip()
    logger.warning("MLflow enabled but no run_id sidecar at %s — this stage will not be logged", sidecar)
    return None


def log_history(history: dict, target_names: list[str], eval_every: int) -> None:
    """Log per-epoch train/val loss and per-target val loss with step indices."""
    for i, tr in enumerate(history["train_loss"], start=1):
        mlflow.log_metric("train_loss", tr, step=i)
    for j, vl in enumerate(history["val_loss"]):
        step = (j + 1) * eval_every
        mlflow.log_metric("val_loss", vl, step=step)
        for name in target_names:
            v = history["per_target_val"][name][j]
            if v == v:  # skip NaN
                mlflow.log_metric(f"val_loss_{name}", v, step=step)


def log_artifacts(paths: list[Path]) -> None:
    for p in paths:
        p = Path(p)
        if p.exists():
            mlflow.log_artifact(str(p))


def log_prediction_complete() -> None:
    mlflow.log_metric("prediction_complete", 1)
    mlflow.set_tag("predict_timestamp", datetime.now(timezone.utc).isoformat())


def log_median_metrics(metrics_df, target_names: list[str]) -> None:
    """Log median-across-units summary metrics per target (e.g. GPP_NSE_med)."""
    for name in target_names:
        sub = metrics_df[metrics_df["target"] == name]
        for metric in _SUMMARY_METRICS:
            if metric not in sub:
                continue
            val = float(np.nanmedian(sub[metric].to_numpy(dtype=float))) if len(sub) else float("nan")
            if val == val:  # skip NaN
                mlflow.log_metric(f"{name}_{metric}_med", val)
