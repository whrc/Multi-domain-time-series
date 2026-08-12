"""
Shared metric functions — used across all domains.

Each function accepts 1-D (or flattened) arrays of predictions and observations.
NaN values in either array are excluded before computation. When the cleaned
sample is too small (< 2 points) or degenerate (constant or zero-sum
observations) a metric is mathematically undefined; these functions return
``nan`` rather than raising, so callers can aggregate across units with
``np.nanmedian`` / ``np.nanmean`` without special-casing.
"""

import numpy as np


def _clean(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays and drop positions where either pred or obs is NaN."""
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(obs))
    return pred[mask], obs[mask]


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Squared Error: √mean((pred − obs)²). NaN if < 2 valid points."""
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency: 1 − Σ(pred−obs)² / Σ(obs−mean(obs))².

    NaN if < 2 valid points or obs is constant (zero variance → undefined).
    """
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    denom = float(np.sum((o - o.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((p - o) ** 2) / denom)


def kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency: 1 − √((r−1)² + (α−1)² + (β−1)²).

    r = Pearson correlation between pred and obs.
    α = std(pred) / std(obs)  (variability ratio).
    β = mean(pred) / mean(obs)  (bias ratio).

    NaN if < 2 valid points, if std(obs) or std(pred) is 0 (r and α undefined),
    or if mean(obs) is 0 (β undefined).
    """
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    std_o = float(o.std())
    std_p = float(p.std())
    mean_o = float(o.mean())
    if std_o == 0.0 or std_p == 0.0 or mean_o == 0.0:
        return float("nan")
    r = float(np.corrcoef(p, o)[0, 1])
    if np.isnan(r):
        return float("nan")
    alpha = std_p / std_o
    beta = float(p.mean()) / mean_o
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def kge_components(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """The three terms `kge()` combines into a single scalar, exposed individually for
    decomposition analysis: r (Pearson correlation), alpha (variability ratio,
    std(pred)/std(obs)), beta (bias ratio, mean(pred)/mean(obs)).

    Same NaN/degenerate-input rules as kge() (returns NaN for each component in those cases),
    and by construction 1 - sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2) == kge(pred, obs).
    """
    p, o = _clean(pred, obs)
    if p.size < 2:
        return {"r": float("nan"), "alpha": float("nan"), "beta": float("nan")}
    std_o = float(o.std())
    std_p = float(p.std())
    mean_o = float(o.mean())
    if std_o == 0.0 or std_p == 0.0 or mean_o == 0.0:
        return {"r": float("nan"), "alpha": float("nan"), "beta": float("nan")}
    r = float(np.corrcoef(p, o)[0, 1])
    if np.isnan(r):
        return {"r": float("nan"), "alpha": float("nan"), "beta": float("nan")}
    return {"r": r, "alpha": std_p / std_o, "beta": float(p.mean()) / mean_o}


def pbias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Percent Bias: 100 × Σ(pred−obs) / Σobs  (%). Positive = overprediction.

    NaN if < 2 valid points or Σobs is 0 (undefined).
    """
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    denom = float(np.sum(o))
    if denom == 0.0:
        return float("nan")
    return float(100.0 * np.sum(p - o) / denom)


def compute_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """Compute all metrics; return {"RMSE", "NSE", "KGE", "PBIAS", "r", "alpha", "beta"}.

    r/alpha/beta are KGE's own components (see kge_components) -- included here, not just in
    kge_components directly, so every existing caller of compute_metrics (per_unit_metrics,
    metrics_df_by_period, and each domain's own 04_evaluate.py) gets them for free, enabling
    post-hoc metric decomposition without new pooling logic anywhere.
    """
    return {
        "RMSE": rmse(pred, obs),
        "NSE": nse(pred, obs),
        "KGE": kge(pred, obs),
        "PBIAS": pbias(pred, obs),
        **kge_components(pred, obs),
    }
