"""
Figure 8 — observed vs. individual vs. fine-tuned time series at one representative test site
per domain (8a Arctic, both SSPs; 8b Amazon; 8c Rangeland), using SEED=1 predictions.
Site rule (pick_site): no target regressed beyond TOLERANCE, then fewest test segments, then
mean individual NSE closest to REPRESENTATIVE_PERCENTILE. Saves PNG + SVG.
"""

import pickle
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from config.config import load_config  # noqa: E402
from shared.plots import PALETTE  # noqa: E402
from make_figure1 import SVG_DIR, _load_module  # noqa: E402
from _common import (  # noqa: E402
    AMAZON_TEST, ARCTIC_FLUXONLY_TEST, MD_FINETUNED_SEEDAVG, RANGELAND_DAY_TO_MONTH,
    RANGELAND_FLUXONLY_TEST, SSP_LABELS, _load_seedavg, _save, _style,
)

SEED = 1
ARCTIC_WINDOW_YEARS = 15  # years shown from proj_start (SSP5-8.5 has no historical period)
TOLERANCE = 0.02  # max NSE drop (fine-tuned vs. individual) still counted as not regressed
REPRESENTATIVE_PERCENTILE = 0.75  # a solidly good site, not the best one

ARCTIC_TARGETS = ["GPP", "RECO"]
AMAZON_TARGETS = ["discharge", "active_fire_count", "burned_area"]
RANGELAND_TARGETS = ["GPP", "RECO", "Rm", "Rg"]

# Arctic units from the source NetCDF; Rangeland is displayed x RANGELAND_DAY_TO_MONTH.
ARCTIC_UNITS = {"GPP": "g m$^{-2}$ month$^{-1}$", "RECO": "g m$^{-2}$ month$^{-1}$"}
AMAZON_UNITS = {"discharge": "m$^3$ s$^{-1}$", "active_fire_count": "count", "burned_area": "km$^2$"}
RANGELAND_UNITS = {"GPP": "g m$^{-2}$ month$^{-1}$", "RECO": "g m$^{-2}$ month$^{-1}$",
                   "Rm": "g m$^{-2}$ month$^{-1}$", "Rg": "g m$^{-2}$ month$^{-1}$"}

AMAZON_LABELS = {"discharge": "Discharge", "active_fire_count": "Active fire count",
                 "burned_area": "Burned area"}

OBS_COLOR = "black"
IND_COLOR = PALETTE[0]    # matches Figure 6
FT_COLOR = PALETTE[2]

ARCTIC_IND_SAMPLE = REPO_ROOT / f"outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed{SEED}/prediction_sample.parquet"
ARCTIC_FT_SAMPLE = REPO_ROOT / f"outputs/multi_domain/evaluation/finetuned_fluxonly_seed{SEED}/arctic/prediction_sample.parquet"
AMAZON_IND_PRED = REPO_ROOT / f"outputs/amazon_domain/predictions/amazon_test_predictions_seed{SEED}.parquet"
AMAZON_FT_PRED = REPO_ROOT / f"outputs/multi_domain/predictions/finetuned_fluxonly_seed{SEED}/amazon/amazon_predictions.parquet"
RANGELAND_IND_PRED = REPO_ROOT / f"outputs/rangeland_domain/predictions/predictions_fluxonly_seed{SEED}.parquet"
RANGELAND_FT_PRED = REPO_ROOT / f"outputs/multi_domain/predictions/finetuned_fluxonly_seed{SEED}/rangeland/rangeland_predictions.parquet"


def pick_site(individual: pd.DataFrame, multi: pd.DataFrame, id_cols: list[str], targets: list[str],
             segment_counts: dict | None = None):
    """Site-selection rule from the module docstring; returns the chosen id tuple."""
    merged = individual.merge(multi, on=id_cols + ["target"], suffixes=("_ind", "_ft"))
    wide_ind = merged.pivot(index=id_cols, columns="target", values="NSE_ind")[targets].dropna()
    wide_ft = merged.pivot(index=id_cols, columns="target", values="NSE_ft")[targets].dropna()
    common = wide_ind.index.intersection(wide_ft.index)
    wide_ind, wide_ft = wide_ind.loc[common], wide_ft.loc[common]

    mean_ind = wide_ind.mean(axis=1)
    not_regressed = ((wide_ft - wide_ind) >= -TOLERANCE).all(axis=1)
    candidates = mean_ind[not_regressed]
    if candidates.empty:
        candidates = mean_ind

    target_val = candidates.quantile(REPRESENTATIVE_PERCENTILE)
    closeness = (candidates - target_val).abs()
    if segment_counts is not None:
        best = min(candidates.index, key=lambda k: (segment_counts.get(k, 999), closeness[k]))
    else:
        best = closeness.idxmin()
    return best if isinstance(best, tuple) else (best,)


def _target_metrics(path: Path, targets: list[str], target_col: str = "target") -> pd.DataFrame:
    df = _load_seedavg(path)
    return df[df[target_col].isin(targets)]


def _segment_counts(test_pkl_path: Path, id_field: str) -> dict:
    """{site/station id -> number of disjoint test segments}. Not used for Arctic (always 1)."""
    with test_pkl_path.open("rb") as f:
        records = pickle.load(f)
    return {r[id_field]: len(r["segments"]) for r in records}


def arctic_site() -> tuple[str, int, int]:
    """One pixel sampled under both SSPs, scored on projected-period "{ssp}_{target}" pseudo-targets."""
    ind = _target_metrics(ARCTIC_FLUXONLY_TEST, ARCTIC_TARGETS)
    ind = ind[~ind["obs_degenerate"] & (ind["period"] == "projected")].copy()
    ft = _target_metrics(MD_FINETUNED_SEEDAVG / "arctic" / "arctic_metrics_seedavg.csv", ARCTIC_TARGETS)
    ft = ft[ft["period"] == "projected"].copy()
    ind["target"] = ind["ssp"].map(SSP_LABELS) + "_" + ind["target"]
    ft["target"] = ft["ssp"].map(SSP_LABELS) + "_" + ft["target"]
    pseudo_targets = sorted(ind["target"].unique())

    # Only prediction_sample pixels have saved raw time series.
    sample = pd.read_parquet(ARCTIC_IND_SAMPLE, columns=["grid", "y", "x", "ssp"]).drop_duplicates()
    both_ssp = sample.groupby(["grid", "y", "x"])["ssp"].nunique()
    valid_pixels = set(both_ssp[both_ssp == 2].index)
    id_cols = ["grid", "y", "x"]
    ind = ind[ind[id_cols].apply(tuple, axis=1).isin(valid_pixels)]
    ft = ft[ft[id_cols].apply(tuple, axis=1).isin(valid_pixels)]

    return pick_site(ind, ft, id_cols, pseudo_targets)


def amazon_site() -> tuple[int]:
    ind = _target_metrics(AMAZON_TEST, AMAZON_TARGETS)
    ft = _target_metrics(MD_FINETUNED_SEEDAVG / "amazon" / "amazon_metrics_seedavg.csv", AMAZON_TARGETS)
    cfg = load_config("amazon_domain")
    segment_counts = _segment_counts(Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl", "station_id")
    segment_counts = {int(k): v for k, v in segment_counts.items()}  # match metrics' int64 station_id
    return pick_site(ind, ft, ["station_id"], AMAZON_TARGETS, segment_counts)


def rangeland_site() -> tuple[str]:
    ind = _load_seedavg(RANGELAND_FLUXONLY_TEST)
    ind = ind.copy()
    ind["target"] = ind["target"].str.replace("_predicted", "", regex=False)
    ind = ind[ind["target"].isin(RANGELAND_TARGETS)]
    ft = _target_metrics(MD_FINETUNED_SEEDAVG / "rangeland" / "rangeland_metrics_seedavg.csv", RANGELAND_TARGETS)
    cfg = load_config("rangeland_domain")
    segment_counts = _segment_counts(Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl", "site")
    return pick_site(ind, ft, ["site"], RANGELAND_TARGETS, segment_counts)


def load_arctic_series(site: tuple[str, int, int]) -> tuple:
    """SSP1-2.6 then SSP5-8.5 GPP/RECO over the same projected-period window."""
    grid, y, x = site
    proj_start = load_config("arctic_domain")["time"]["projected_start_year"]
    window_start = pd.Timestamp(f"{proj_start}-01-01")
    window_end = pd.Timestamp(f"{proj_start + ARCTIC_WINDOW_YEARS - 1}-12-31")
    ind_all = pd.read_parquet(ARCTIC_IND_SAMPLE)
    ft_all = pd.read_parquet(ARCTIC_FT_SAMPLE)

    time, lat, lon = None, None, None
    obs_d, ind_d, ft_d = {}, {}, {}
    for ssp, ssp_label in SSP_LABELS.items():
        sel = lambda df: df[(df.grid == grid) & (df.y == y) & (df.x == x) & (df.ssp == ssp)
                           & (df.time >= window_start) & (df.time <= window_end)].sort_values("time")
        ind, ft = sel(ind_all), sel(ft_all)
        if time is None:
            time = ind["time"].to_numpy()
            lat, lon = float(ind["lat"].iloc[0]), float(ind["lon"].iloc[0])
        for t in ARCTIC_TARGETS:
            key = f"{ssp_label}_{t}"
            obs_d[key] = ind[f"obs_{t.lower()}"].to_numpy()
            ind_d[key] = ind[f"pred_{t.lower()}"].to_numpy()
            ft_d[key] = ft[f"pred_{t.lower()}"].to_numpy()
    return time, obs_d, ind_d, ft_d, lat, lon


def load_amazon_series(site: tuple[int]) -> tuple[pd.Series, dict, dict]:
    (station_id,) = site
    ev04 = _load_module("amazon_domain", "04_evaluate.py", "_am04_fig8")
    cfg = load_config("amazon_domain")
    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    obs_long = ev04.ground_truth_long(test_records, scaler, cfg["targets"])
    obs_long["station_id"] = obs_long["station_id"].astype(str)

    ind_pred = pd.read_parquet(AMAZON_IND_PRED)
    ft_pred = pd.read_parquet(AMAZON_FT_PRED)
    for df in (ind_pred, ft_pred):
        df["station_id"] = df["station_id"].astype(str)

    station_id = str(station_id)
    obs = obs_long[obs_long["station_id"] == station_id]
    ind = ind_pred[ind_pred["station_id"] == station_id].sort_values(["year", "month"])
    ft = ft_pred[ft_pred["station_id"] == station_id].sort_values(["year", "month"])
    time = pd.to_datetime(ind[["year", "month"]].assign(day=1))

    obs_d, ind_d, ft_d = {}, {}, {}
    for t in AMAZON_TARGETS:
        o = obs[obs["target"] == t].sort_values(["year", "month"])
        obs_d[t] = o["obs"].to_numpy()
        ind_d[t] = ind[f"{t}_pred"].to_numpy()
        ft_d[t] = ft[f"{t}_pred"].to_numpy()
    return time.to_numpy(), obs_d, ind_d, ft_d


def load_rangeland_series(site: tuple[str]) -> tuple[pd.Series, dict, dict]:
    (site_id,) = site
    ev04 = _load_module("rangeland_domain", "04_evaluate.py", "_rl04_fig8")
    cfg = load_config("rangeland_domain")
    flux_names = cfg["targets"]["fluxes"]
    with (Path(cfg["paths"]["preprocessed_dir"]) / "test.pkl").open("rb") as f:
        test_records = pickle.load(f)
    with Path(cfg["paths"]["scaler"]).open("rb") as f:
        scaler = pickle.load(f)
    # Flux-only slicing as in 04_evaluate.py --flux-only: features, then fluxes (before pools).
    keep = ev04.NUM_FEATURES + len(flux_names)
    test_records = [{**r, "segments": [s[:, :keep] for s in r["segments"]]} for r in test_records]
    scaler = {"mean": scaler["mean"][:keep], "std": scaler["std"][:keep]}
    obs_long = ev04.ground_truth_long(test_records, scaler, flux_names, len(flux_names))
    obs_long["target"] = obs_long["target"].str.replace("_predicted", "", regex=False)
    obs = obs_long[obs_long["site"] == site_id].sort_values("date")

    ind_pred = pd.read_parquet(RANGELAND_IND_PRED)
    ft_pred = pd.read_parquet(RANGELAND_FT_PRED)
    ind = ind_pred[ind_pred["site"] == site_id].sort_values("date")
    ft = ft_pred[ft_pred["site"] == site_id].sort_values("date")
    time = ind["date"].to_numpy()

    obs_d, ind_d, ft_d = {}, {}, {}
    for t in RANGELAND_TARGETS:
        o = obs[obs["target"] == t].sort_values("date")
        obs_d[t] = o["obs"].to_numpy() * RANGELAND_DAY_TO_MONTH
        ind_d[t] = ind[f"{t}_predicted"].to_numpy() * RANGELAND_DAY_TO_MONTH
        ft_d[t] = ft[f"{t}_predicted"].to_numpy() * RANGELAND_DAY_TO_MONTH
    return time, obs_d, ind_d, ft_d


def _reindex_monthly(time, series_dict: dict) -> tuple:
    """Reindex onto a full monthly axis so gaps between test segments show as line breaks."""
    idx = pd.DatetimeIndex(time)
    full_time = pd.date_range(idx.min(), idx.max(), freq="MS")
    out = {k: pd.Series(v, index=idx).reindex(full_time).to_numpy() for k, v in series_dict.items()}
    return full_time.to_numpy(), out


def plot_three_line_timeseries(time, obs_d: dict, ind_d: dict, ft_d: dict, title: str,
                               filename: str, units: dict, labels: dict | None = None) -> None:
    targets = list(obs_d.keys())
    labels = labels or {}
    fig, axes = plt.subplots(len(targets), 1, figsize=(7.5, min(1.8 * len(targets), 8.5)),
                             squeeze=False, sharex=True)
    for i, t in enumerate(targets):
        ax = axes[i, 0]
        full_time, series = _reindex_monthly(time, {"obs": obs_d[t], "ind": ind_d[t], "ft": ft_d[t]})
        ax.plot(full_time, series["obs"], color=OBS_COLOR, linewidth=1.0, label="Observed")
        ax.plot(full_time, series["ind"], color=IND_COLOR, linewidth=1.0, label="Individual")
        ax.plot(full_time, series["ft"], color=FT_COLOR, linewidth=1.0, label="Fine-tuned")
        ax.set_ylabel(f"{labels.get(t, t)}\n({units[t]})")
    axes[-1, 0].set_xlabel("time")
    # Title a fixed 0.08 in from the top, since figure heights differ (3 vs. 4 rows).
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=9, fontweight="bold", y=1 - 0.08 / fig_h, va="top")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=True, fancybox=False,
              bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    # tight_layout over-pads the top; set it explicitly: 0.30 in = title line + small gap.
    fig.subplots_adjust(top=1 - 0.30 / fig_h)
    SVG_DIR.mkdir(exist_ok=True)
    svg_path = SVG_DIR / Path(filename).with_suffix(".svg").name
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {svg_path}")
    _save(fig, filename)


def make_arctic() -> None:
    grid, y, x = arctic_site()
    print(f"Arctic site: grid={grid} y={y} x={x}")
    time, obs_d, ind_d, ft_d, lat, lon = load_arctic_series((grid, y, x))
    units = {k: ARCTIC_UNITS[k.split("_", 1)[1]] for k in obs_d}
    labels = {k: k.replace("_", " ") for k in obs_d}
    lon_label = f"{lon:.1f}°E" if lon >= 0 else f"{-lon:.1f}°W"
    title = f"Arctic pixel: {lat:.1f}°N, {lon_label}"
    plot_three_line_timeseries(time, obs_d, ind_d, ft_d, title,
                               "fig8a_arctic_timeseries.png", units, labels)


def make_amazon() -> None:
    (station_id,) = amazon_site()
    print(f"Amazon site: station_id={station_id}")
    time, obs_d, ind_d, ft_d = load_amazon_series((station_id,))
    plot_three_line_timeseries(time, obs_d, ind_d, ft_d, f"Amazon station: {station_id}",
                               "fig8b_amazon_timeseries.png", AMAZON_UNITS, AMAZON_LABELS)


def make_rangeland() -> None:
    (site,) = rangeland_site()
    print(f"Rangeland site: {site}")
    time, obs_d, ind_d, ft_d = load_rangeland_series((site,))
    plot_three_line_timeseries(time, obs_d, ind_d, ft_d, f"Rangeland site: {site}",
                               "fig8c_rangeland_timeseries.png", RANGELAND_UNITS)


def main() -> None:
    _style()
    make_arctic()
    make_amazon()
    make_rangeland()


if __name__ == "__main__":
    main()
