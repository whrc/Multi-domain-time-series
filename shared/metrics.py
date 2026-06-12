"""
Shared metric functions — used across all domains.

Each function accepts 1-D (or flattened) arrays of predictions and observations.
NaN values in either array are excluded before computation.
"""

import numpy as np


def _clean(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop positions where either pred or obs is NaN."""
    mask = ~(np.isnan(pred) | np.isnan(obs))
    return pred[mask], obs[mask]


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Squared Error: √mean((pred − obs)²)."""
    raise NotImplementedError


def nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency: 1 − Σ(pred−obs)² / Σ(obs−mean(obs))²."""
    raise NotImplementedError


def kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency: 1 − √((r−1)² + (α−1)² + (β−1)²).

    r = Pearson correlation between pred and obs.
    α = std(pred) / std(obs)  (variability ratio).
    β = mean(pred) / mean(obs)  (bias ratio).
    """
    raise NotImplementedError


def pbias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Percent Bias: 100 × Σ(pred−obs) / Σobs  (%)."""
    raise NotImplementedError


def compute_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """Compute all four metrics; return {"RMSE": ..., "NSE": ..., "KGE": ..., "PBIAS": ...}."""
    raise NotImplementedError
