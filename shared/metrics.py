"""Metrics on 1-D pred/obs arrays. NaN pairs are dropped; undefined cases (< 2 points,
degenerate obs) return NaN rather than raising, so callers can use np.nanmedian/nanmean.
"""

import numpy as np


def _clean(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = ~(np.isnan(pred) | np.isnan(obs))
    return pred[mask], obs[mask]


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)))


def nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. NaN if obs is constant."""
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    denom = float(np.sum((o - o.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((p - o) ** 2) / denom)


def kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency. NaN if std(obs), std(pred) or mean(obs) is 0."""
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
    """KGE terms: r (Pearson), alpha = std(pred)/std(obs), beta = mean(pred)/mean(obs).
    Same NaN rules as kge()."""
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
    """Percent bias (%); positive = overprediction. NaN if sum(obs) is 0."""
    p, o = _clean(pred, obs)
    if p.size < 2:
        return float("nan")
    denom = float(np.sum(o))
    if denom == 0.0:
        return float("nan")
    return float(100.0 * np.sum(p - o) / denom)


def compute_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """Return {"RMSE", "NSE", "KGE", "PBIAS", "r", "alpha", "beta"}."""
    return {
        "RMSE": rmse(pred, obs),
        "NSE": nse(pred, obs),
        "KGE": kge(pred, obs),
        "PBIAS": pbias(pred, obs),
        **kge_components(pred, obs),
    }
