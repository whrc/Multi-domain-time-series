# domains/arctic_domain/04_evaluate.py

import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config

DOMAIN = os.environ.get("ARCTIC_DOMAIN", "arctic_domain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_TARGETS = 4  # ALD, GPP, RECO, VEGC (always last NUM_TARGETS columns)
SSP1 = "ssp1_2_6_mri_esm2_0"
SSP5 = "ssp5_8_5_mri_esm2_0"


# ── Actuals ───────────────────────────────────────────────────────────────────

def load_test_actuals(cfg: dict, num_features: int) -> dict:
    """
    Load test split and inverse-transform target columns.
    Returns dict keyed by (grid, ssp, y_idx, x_idx) -> (T, NUM_TARGETS) ndarray.
    """
    base = Path(cfg["paths"]["preprocessed_dir"])
    with open(base / "test.pkl", "rb") as f:
        test_records = pickle.load(f)
    with open(cfg["paths"]["scaler"], "rb") as f:
        scaler = pickle.load(f)

    mean = scaler["mean"][num_features:]  # (NUM_TARGETS,)
    std  = scaler["std"][num_features:]   # (NUM_TARGETS,)

    lookup = {}
    for rec in test_records:
        actual_norm = rec["data"][:, num_features:]   # (T, NUM_TARGETS)
        actual      = actual_norm * std + mean         # (T, NUM_TARGETS)
        lookup[(rec["grid"], rec["ssp"], rec["y"], rec["x"])] = actual

    log.info("Loaded actuals for %d test pixels", len(lookup))
    return lookup


# ── Predictions ───────────────────────────────────────────────────────────────

def load_predictions_for_grid(grid: str, ssp: str, cfg: dict) -> dict:
    """
    Load all prediction NetCDF files for (grid, ssp).
    Returns dict:
      (variable_name, split_key)  -> xr.DataArray(time, y, x)
      'lat'                       -> np.ndarray(y, x) or None
      'lon'                       -> np.ndarray(y, x) or None
    split_key is 'tr' (historical) or 'sc' (projected).
    """
    pred_dir = Path(cfg["paths"]["predictions"]) / grid / ssp
    if not pred_dir.exists():
        raise FileNotFoundError(f"No prediction files at {pred_dir}. Run 03_predict.py first.")

    result: dict = {}
    lat_arr = lon_arr = None

    for tgt in cfg["targets"]:
        name       = tgt["name"]
        resolution = tgt["resolution"]
        for nc_path in sorted(pred_dir.glob(f"{name}_{resolution}_pred_*.nc")):
            split_key = nc_path.stem.split("_pred_")[-1]
            with xr.open_dataset(nc_path) as ds:
                result[(name, split_key)] = ds[name].load()
                if lat_arr is None and "lat" in ds:
                    lat_arr = ds["lat"].values
                    lon_arr = ds["lon"].values

    result["lat"] = lat_arr
    result["lon"] = lon_arr
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """
    RMSE, NSE, KGE, PBIAS for paired 1-D arrays. NaN-safe.
    Returns all-NaN if fewer than 2 valid time steps.
    """
    nan_result = {"RMSE": np.nan, "NSE": np.nan, "KGE": np.nan, "PBIAS": np.nan}
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return nan_result

    o, p = obs[mask], pred[mask]
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))

    obs_mean = float(np.mean(o))
    ss_res = float(np.sum((o - p) ** 2))
    ss_tot = float(np.sum((o - obs_mean) ** 2))
    nse    = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    r      = float(np.corrcoef(o, p)[0, 1])
    sig_o  = float(np.std(o))
    alpha  = float(np.std(p) / sig_o) if sig_o > 0 else np.nan
    beta   = float(np.mean(p) / obs_mean) if obs_mean != 0 else np.nan
    if any(np.isnan(v) for v in (r, alpha, beta)):
        kge = np.nan
    else:
        kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    obs_sum = float(np.sum(o))
    pbias   = float(100.0 * np.sum(o - p) / obs_sum) if obs_sum != 0 else np.nan

    return {"RMSE": rmse, "NSE": nse, "KGE": kge, "PBIAS": pbias}


# ── Per-grid evaluation ───────────────────────────────────────────────────────

def evaluate_grid_ssp(
    grid: str, ssp: str, actuals_lookup: dict, cfg: dict
) -> list[dict]:
    """
    Compute per-pixel metrics for all periods and variables in (grid, ssp).
    Reads predictions from disk; uses actuals_lookup for ground truth.
    """
    pred_data = load_predictions_for_grid(grid, ssp, cfg)
    lat_grid  = pred_data["lat"]   # (ny, nx) or None
    lon_grid  = pred_data["lon"]   # (ny, nx) or None

    if ssp == SSP1:
        full_monthly = pd.date_range("1901-01-01", "2100-12-31", freq="MS")  # 2400
        hist_mask    = full_monthly <= pd.Timestamp("2024-12-01")
        proj_mask    = ~hist_mask
        period_defs  = [
            ("historical", "tr", hist_mask, full_monthly[hist_mask]),
            ("projected",  "sc", proj_mask, full_monthly[proj_mask]),
        ]
    else:  # SSP5
        full_monthly = pd.date_range("2025-01-01", "2100-12-31", freq="MS")  # 912
        period_defs  = [
            ("projected", "sc", np.ones(len(full_monthly), dtype=bool), full_monthly),
        ]

    rows: list[dict] = []
    for (g, s, yi, xi), actual_ts in actuals_lookup.items():
        if g != grid or s != ssp:
            continue

        lat_val = float(lat_grid[yi, xi]) if lat_grid is not None else np.nan
        lon_val = float(lon_grid[yi, xi]) if lon_grid is not None else np.nan

        for period_label, split_key, period_mask, monthly_idx in period_defs:
            actual_period = actual_ts[period_mask]  # (T_period, NUM_TARGETS)

            for col_idx, tgt in enumerate(cfg["targets"]):
                name       = tgt["name"]
                resolution = tgt["resolution"]
                key        = (name, split_key)

                if key not in pred_data:
                    log.warning("Missing prediction: grid=%s ssp=%s %s %s", grid, ssp, name, split_key)
                    continue

                pred_series = pred_data[key].isel(y=yi, x=xi).values  # (T_pred,)
                obs_monthly = actual_period[:, col_idx]

                if resolution == "yearly":
                    obs_series = (
                        pd.Series(obs_monthly, index=monthly_idx)
                        .resample("YS").mean()
                        .values
                    )
                else:
                    obs_series = obs_monthly

                metrics = compute_metrics(obs_series, pred_series)
                rows.append({
                    "grid": grid, "y": yi, "x": xi,
                    "lat": lat_val, "lon": lon_val,
                    "ssp": ssp, "period": period_label,
                    "variable": name,
                    **metrics,
                })

    log.info("  grid=%s ssp=%s: %d rows", grid, ssp, len(rows))
    return rows


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation(actuals_lookup: dict, cfg: dict) -> pd.DataFrame:
    """
    Discover all (grid, ssp) pairs from predictions dir, evaluate each, save metrics.csv.
    """
    pred_root = Path(cfg["paths"]["predictions"])
    all_rows: list[dict] = []

    for grid_dir in sorted(pred_root.iterdir()):
        if not grid_dir.is_dir():
            continue
        for ssp_dir in sorted(grid_dir.iterdir()):
            if not ssp_dir.is_dir():
                continue
            grid, ssp = grid_dir.name, ssp_dir.name
            log.info("Evaluating grid=%s ssp=%s", grid, ssp)
            all_rows.extend(evaluate_grid_ssp(grid, ssp, actuals_lookup, cfg))

    df = pd.DataFrame(all_rows, columns=[
        "grid", "y", "x", "lat", "lon", "ssp", "period", "variable",
        "RMSE", "NSE", "KGE", "PBIAS",
    ])
    out_dir = Path(cfg["paths"]["evaluation"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    log.info("Saved %s  (%d rows)", metrics_path, len(df))
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

_SSP_SHORT    = {SSP1: "ssp1", SSP5: "ssp5"}
_PERIOD_COLOR = {"historical": "#1f77b4", "projected": "#ff7f0e"}


def plot_boxplots(df: pd.DataFrame, cfg: dict) -> None:
    """2x2 boxplot figure per SSP: one subplot per metric, grouped by variable."""
    metrics   = ["RMSE", "NSE", "KGE", "PBIAS"]
    variables = [t["name"] for t in cfg["targets"]]
    out_dir   = Path(cfg["paths"]["evaluation"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for ssp in sorted(df["ssp"].unique()):
        ssp_df  = df[df["ssp"] == ssp]
        periods = sorted(ssp_df["period"].unique())
        n_p     = len(periods)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Metrics — {_SSP_SHORT.get(ssp, ssp)}")

        for ax, metric in zip(axes.flatten(), metrics):
            positions, box_data, box_colors = [], [], []
            tick_pos, tick_labels = [], []

            for vi, var in enumerate(variables):
                center = vi * (n_p + 1)
                tick_pos.append(center + (n_p - 1) / 2)
                tick_labels.append(var)
                for pi, period in enumerate(periods):
                    vals = ssp_df[
                        (ssp_df["variable"] == var) & (ssp_df["period"] == period)
                    ][metric].dropna().values
                    positions.append(center + pi)
                    box_data.append(vals)
                    box_colors.append(_PERIOD_COLOR.get(period, "#2ca02c"))

            bp = ax.boxplot(
                box_data, positions=positions, widths=0.6,
                patch_artist=True, whis=(5, 95), showfliers=False,
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)

            ax.set_title(metric)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)
            ax.set_ylabel(metric)

        legend_handles = [
            Patch(facecolor=_PERIOD_COLOR.get(p, "#2ca02c"), label=p) for p in periods
        ]
        fig.legend(handles=legend_handles, loc="upper right")

        out_path = out_dir / f"metrics_boxplot_{_SSP_SHORT.get(ssp, ssp)}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", out_path)


def plot_spatial_nse(df: pd.DataFrame, cfg: dict) -> None:
    """Scatter map of NSE per (ssp, period, variable)."""
    out_dir   = Path(cfg["paths"]["evaluation"]) / "spatial_metrics_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    variables = [t["name"] for t in cfg["targets"]]

    for ssp in sorted(df["ssp"].unique()):
        short  = _SSP_SHORT.get(ssp, ssp)
        ssp_df = df[df["ssp"] == ssp]
        for period in sorted(ssp_df["period"].unique()):
            subset = ssp_df[ssp_df["period"] == period]
            for var in variables:
                var_df = subset[subset["variable"] == var].dropna(subset=["lat", "lon", "NSE"])
                if var_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                sc = ax.scatter(
                    var_df["lon"], var_df["lat"], c=var_df["NSE"],
                    cmap="RdYlGn", vmin=-1, vmax=1, s=10, alpha=0.8,
                )
                plt.colorbar(sc, ax=ax, label="NSE")
                ax.set_title(f"NSE — {var} | {short} {period}")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                out_path = out_dir / f"NSE_{short}_{period}_{var}.png"
                fig.savefig(out_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                log.info("Saved %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config(DOMAIN)
    with open(cfg["paths"]["scaler"], "rb") as f:
        scaler = pickle.load(f)
    num_features = len(scaler["mean"]) - NUM_TARGETS

    actuals_lookup = load_test_actuals(cfg, num_features)
    df = run_evaluation(actuals_lookup, cfg)
    plot_boxplots(df, cfg)
    plot_spatial_nse(df, cfg)
    log.info("Evaluation complete. %d rows written to metrics.csv.", len(df))


if __name__ == "__main__":
    main()
