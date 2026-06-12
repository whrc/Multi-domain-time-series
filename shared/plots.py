"""
Shared plotting utilities — used across all domains.

Standards applied throughout all functions:
- Color palette: Okabe-Ito 8-color color-blind-safe palette (see PALETTE below)
- Figure size: never exceeds 10 × 10 inches; default to the smallest sensible size per plot type
- All functions return a matplotlib Figure; the caller is responsible for saving (fig.savefig)
  or displaying (plt.show). Pass save_path to have the function save and close automatically.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Okabe-Ito 8-color color-blind-safe palette
PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path | None = None,
) -> Figure:
    """Epoch-by-epoch training and validation loss line plot."""
    raise NotImplementedError


def plot_pred_vs_true(
    pred_dict: dict[str, np.ndarray],
    obs_dict: dict[str, np.ndarray],
    save_path: Path | None = None,
) -> Figure:
    """4-panel scatter of predicted vs true, one panel per target variable.

    Annotates each panel with RMSE and NSE.
    pred_dict / obs_dict: keys are target variable names (ALD, GPP, RECO, VEGC).
    """
    raise NotImplementedError


def plot_metric_boxplot(
    metrics_df: pd.DataFrame,
    ssp: str,
    save_path: Path | None = None,
) -> Figure:
    """Boxplot for one SSP scenario.

    metrics_df columns: variable, period (historical/projected), RMSE, NSE, KGE, PBIAS.
    Produces 4 subplots (one per metric); each subplot shows historical vs projected
    distributions across test pixels for all target variables.
    """
    raise NotImplementedError


def plot_cdf(
    series_dict: dict[str, np.ndarray],
    xlabel: str,
    save_path: Path | None = None,
) -> Figure:
    """Overlapping empirical CDF lines, one line per key in series_dict."""
    raise NotImplementedError


def plot_spatial_map(
    data_2d: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> Figure:
    """2D heatmap of per-pixel metric values (e.g. NSE map over a grid)."""
    raise NotImplementedError
