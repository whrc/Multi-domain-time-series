"""
Multi-domain — Step 4: evaluation.

See domains/multi_domain/multi_description.md § "Step 4 — Evaluation".

Recomputes predictions from checkpoints for each domain × stage, then computes per-unit
metrics (RMSE, NSE, KGE, PBIAS) and writes CSVs and diagnostic plots. Both stage1 and
stage2 checkpoints are evaluated; cross-model comparison with individual baselines is
handled separately in a root-level intercomparison script.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from domains.multi_domain.model import MultiDomainModel  # noqa: E402
from shared.evaluate import per_unit_metrics, predict_and_inverse, stack_by_target  # noqa: E402
from shared.metrics import compute_metrics  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_pred_vs_true  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["arctic", "amazon", "rangeland"]
DOMAIN_NTARGETS = {"arctic": 4, "amazon": 3, "rangeland": 10}
DOMAIN_TARGET_NAMES = {
    "arctic":    ["ALD", "GPP", "RECO", "VEGC"],
    "amazon":    ["discharge", "active_fire_count", "burned_area"],
    "rangeland": ["GPP", "RECO", "Rm", "Rg", "AGB", "BGB", "AGL", "BGL", "POC", "HOC"],
}
DOMAIN_ID_FIELDS = {
    "arctic":    ["grid", "y", "x", "ssp"],
    "amazon":    ["station_id"],
    "rangeland": ["site", "pft"],
}
ARCTIC_YEARLY = {"ALD", "VEGC"}


def load_model(cfg: dict, domain_specs: dict, ckpt_path: Path,
               device: torch.device) -> MultiDomainModel:
    model = MultiDomainModel(cfg, domain_specs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


def arctic_metrics(seg_meta: list[dict], pred_list: list[np.ndarray],
                   obs_list: list[np.ndarray], arctic_cfg: dict) -> pd.DataFrame:
    """Per-pixel per-target per-SSP per-period metrics, with January-only filter for yearly targets."""
    idx_map    = {k: pd.date_range(v["start"], v["end"], freq="MS")
                  for k, v in arctic_cfg["time"]["scenarios"].items()}
    proj_start = arctic_cfg["time"]["projected_start_year"]
    target_names = DOMAIN_TARGET_NAMES["arctic"]
    rows = []
    for meta, pred, obs in zip(seg_meta, pred_list, obs_list):
        pred_r = np.round(pred, 3)
        time   = idx_map["ssp1" if "ssp1" in meta["ssp"] else "ssp5"]
        periods = (("historical", time.year < proj_start), ("projected", time.year >= proj_start))
        for i, name in enumerate(target_names):
            pos = (time.month == 1) if name in ARCTIC_YEARLY else np.ones(len(time), dtype=bool)
            for period, in_period in periods:
                sel = pos & in_period
                if not sel.any():
                    continue
                rows.append({
                    "grid": meta["grid"], "y": meta["y"], "x": meta["x"],
                    "lat": meta["lat"], "lon": meta["lon"], "ssp": meta["ssp"],
                    "target": name, "period": period,
                    **compute_metrics(pred_r[sel, i], obs[sel, i]),
                })
    return pd.DataFrame(rows).round(3)


def evaluate_domain(domain: str, stage: str, cfg: dict, domain_specs: dict,
                    device: torch.device, models_dir: Path, eval_dir: Path) -> None:
    ckpt_file = "stage1_best.pt" if stage == "stage1" else f"stage2_{domain}_best.pt"
    ckpt_path = models_dir / ckpt_file
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found: %s — skipping %s/%s", ckpt_path, domain, stage)
        return

    with (Path(cfg["paths"][domain]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"][domain]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)

    model = load_model(cfg, domain_specs, ckpt_path, device)
    domain_model = lambda x, _d=domain: model(x, domain=_d)
    n_targets    = DOMAIN_NTARGETS[domain]
    seq_len      = cfg["model"]["seq_len"]
    target_names = DOMAIN_TARGET_NAMES[domain]

    seg_meta, pred_list, obs_list = predict_and_inverse(
        domain_model, test_records, n_targets, seq_len, device, scaler
    )
    logger.info("%s/%s: predicted %d segments", domain, stage, len(seg_meta))

    # Compute metrics
    if domain == "arctic":
        arctic_cfg   = load_config("arctic_domain")
        metrics_df   = arctic_metrics(seg_meta, pred_list, obs_list, arctic_cfg)
        group_col    = "ssp"
    elif domain == "amazon":
        metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, DOMAIN_ID_FIELDS[domain])
        group_col  = None
    else:  # rangeland
        metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, DOMAIN_ID_FIELDS[domain])
        group_col  = "pft"

    out_dir = eval_dir / (stage if stage == "stage1" else f"stage2_{domain}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{domain}_{stage}_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Saved metrics: %s (%d rows)", csv_path, len(metrics_df))

    # Scatter and boxplot
    pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
    plot_pred_vs_true(pred_d, obs_d, save_path=out_dir / f"{domain}_scatter.png")
    plot_metric_boxplot(metrics_df, group_col=group_col,
                        title=f"{domain} {stage}", save_path=out_dir / f"{domain}_boxplot.png")
    logger.info("Saved diagnostic plots to %s", out_dir)


def main() -> None:
    cfg        = load_config("multi_domain")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir   = Path(cfg["paths"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Infer nF_arctic from arctic test.pkl
    arctic_test_path = Path(cfg["paths"]["arctic"]["preprocessed_dir"]) / "test.pkl"
    with arctic_test_path.open("rb") as f:
        nF_arctic = pickle.load(f)[0]["data"].shape[1] - 4
    domain_specs = {
        "arctic":    {"nFeatures": nF_arctic, "nTargets": 4},
        "amazon":    {"nFeatures": 14,        "nTargets": 3},
        "rangeland": {"nFeatures": 22,        "nTargets": 10},
    }
    logger.info("Device=%s | arctic nFeatures=%d", device, nF_arctic)

    for domain in DOMAINS:
        for stage in ["stage1", "stage2"]:
            evaluate_domain(domain, stage, cfg, domain_specs, device, models_dir, eval_dir)


if __name__ == "__main__":
    main()
