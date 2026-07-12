"""
Multi-domain — Step 4: evaluation.

See domains/multi_domain/multi_description.md § "Step 4 — Evaluation".

Recomputes predictions from checkpoints for each domain × stage, then computes per-unit
metrics (RMSE, NSE, KGE, PBIAS) and writes CSVs and diagnostic plots. Both the pretrained and
finetuned checkpoints are evaluated, for whichever target-set variant --flux-only selects;
cross-model comparison with individual baselines is handled separately in a future root-level
intercomparison script (not yet implemented).

Arctic additionally exports a 50-pixel deterministic prediction sample (full time series),
using the exact same seed and site-selection as the individual Arctic pipeline's own
04_evaluate.py, so the two runs are directly comparable at identical sites — dense per-pixel
predictions are NOT saved for all test pixels (see arctic_description.md's caution about this
reaching hundreds of GB).
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from domains.arctic_domain._naming import sample_test_pixels, save_prediction_sample  # noqa: E402
from domains.multi_domain.flux_only import (  # noqa: E402
    DOMAIN_ID_FIELDS,
    DOMAIN_NTARGETS,
    DOMAINS,
    apply_flux_only,
    checkpoint_path,
    stage_output_dir,
    variant_ntargets,
    variant_target_names,
)
from domains.multi_domain.model import DomainRoutedModel, MultiDomainModel  # noqa: E402
from shared.evaluate import per_unit_metrics, predict_and_inverse, stack_by_target  # noqa: E402
from shared.metrics import compute_metrics  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_pred_vs_true  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCTIC_YEARLY = {"ALD", "VEGC"}
ARCTIC_SAMPLE_PIXELS = 50


def load_model(cfg: dict, domain_specs: dict, ckpt_path: Path,
               device: torch.device) -> MultiDomainModel:
    model = MultiDomainModel(cfg, domain_specs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


def arctic_idx_map(arctic_cfg: dict) -> dict[str, pd.DatetimeIndex]:
    return {k: pd.date_range(v["start"], v["end"], freq="MS") for k, v in arctic_cfg["time"]["scenarios"].items()}


def arctic_metrics(seg_meta: list[dict], pred_list: list[np.ndarray],
                   obs_list: list[np.ndarray], idx_map: dict, proj_start: int,
                   target_names: list[str]) -> pd.DataFrame:
    """Per-pixel per-target per-SSP per-period metrics, with January-only filter for yearly
    targets (flux-only has no yearly targets — ALD/VEGC are dropped — so every position is used)."""
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


def evaluate_domain(domain: str, stage: str, flux_only: bool, cfg: dict, domain_specs: dict,
                    device: torch.device, models_dir: Path, eval_dir: Path) -> None:
    variant_label = "fluxonly" if flux_only else "full"
    ckpt_path = checkpoint_path(models_dir, stage, domain, flux_only)
    out_dir   = stage_output_dir(eval_dir, stage, domain, flux_only)
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found: %s — skipping %s/%s/%s", ckpt_path, domain, stage, variant_label)
        return

    with (Path(cfg["paths"][domain]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"][domain]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    if flux_only:
        test_records, scaler = apply_flux_only(domain, test_records, scaler)

    model = load_model(cfg, domain_specs, ckpt_path, device)
    domain_model = DomainRoutedModel(model, domain)
    ntargets     = variant_ntargets(flux_only)
    n_targets    = ntargets[domain]
    seq_len      = cfg["model"]["seq_len"]
    target_names = variant_target_names(flux_only)[domain]

    seg_meta, pred_list, obs_list = predict_and_inverse(
        domain_model, test_records, n_targets, seq_len, device, scaler
    )
    logger.info("%s/%s/%s: predicted %d segments", domain, stage, variant_label, len(seg_meta))

    # Compute metrics
    if domain == "arctic":
        arctic_cfg   = load_config("arctic_domain")
        idx_map      = arctic_idx_map(arctic_cfg)
        proj_start   = arctic_cfg["time"]["projected_start_year"]
        metrics_df   = arctic_metrics(seg_meta, pred_list, obs_list, idx_map, proj_start, target_names)
        group_col    = "ssp"
    elif domain == "amazon":
        metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, DOMAIN_ID_FIELDS[domain])
        group_col  = None
    else:  # rangeland
        metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, DOMAIN_ID_FIELDS[domain])
        group_col  = "pft"

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{domain}_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Saved metrics: %s (%d rows)", csv_path, len(metrics_df))

    # Scatter and boxplot
    pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
    plot_pred_vs_true(pred_d, obs_d, save_path=out_dir / f"{domain}_scatter.png")
    plot_metric_boxplot(metrics_df, group_col=group_col,
                        title=f"{domain} {stage} ({variant_label})", save_path=out_dir / f"{domain}_boxplot.png")
    logger.info("Saved diagnostic plots to %s", out_dir)

    # Arctic: deterministic 50-pixel prediction sample, same seed/site-selection as the
    # individual Arctic pipeline's 04_evaluate.py, so the two runs are directly comparable at
    # identical sites without saving every test pixel's dense prediction (see module docstring).
    if domain == "arctic":
        arctic_seed = arctic_cfg["preprocessing"]["random_seed"]
        pred_list_rounded = [np.round(p, 3) for p in pred_list]
        sampled_pixels = sample_test_pixels(seg_meta, seed=arctic_seed, n_pixels=ARCTIC_SAMPLE_PIXELS)
        save_prediction_sample(
            seg_meta, pred_list_rounded, obs_list, target_names, idx_map, sampled_pixels,
            save_path=out_dir / "prediction_sample.parquet",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-only", action="store_true",
                        help="Evaluate the flux-only checkpoint/target-set variant — see "
                             "02_train.py --flux-only.")
    args = parser.parse_args()
    flux_only = args.flux_only

    cfg        = load_config("multi_domain")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir   = Path(cfg["paths"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Infer nF_arctic from arctic's raw (full-target) test.pkl — flux-only only reduces target
    # columns, not feature columns, so this is the same regardless of --flux-only.
    arctic_test_path = Path(cfg["paths"]["arctic"]["preprocessed_dir"]) / "test.pkl"
    with arctic_test_path.open("rb") as f:
        nF_arctic = pickle.load(f)[0]["data"].shape[1] - DOMAIN_NTARGETS["arctic"]

    ntargets = variant_ntargets(flux_only)
    domain_specs = {
        "arctic":    {"nFeatures": nF_arctic, "nTargets": ntargets["arctic"]},
        "amazon":    {"nFeatures": 14,        "nTargets": ntargets["amazon"]},
        "rangeland": {"nFeatures": 22,        "nTargets": ntargets["rangeland"]},
    }
    logger.info("Device=%s | arctic nFeatures=%d | flux_only=%s", device, nF_arctic, flux_only)

    for domain in DOMAINS:
        for stage in ["pretrained", "finetuned"]:
            evaluate_domain(domain, stage, flux_only, cfg, domain_specs, device, models_dir, eval_dir)


if __name__ == "__main__":
    main()
