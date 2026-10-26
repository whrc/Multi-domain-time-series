"""Checks for shared/metrics.py kge_components(). Run directly (no pytest)."""

import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.metrics import kge, kge_components  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_components_reconstruct_kge_scalar() -> None:
    """1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2) from the components must equal kge()."""
    rng = np.random.default_rng(0)
    obs = rng.normal(loc=10.0, scale=3.0, size=200)
    cases = {
        "perfect": obs.copy(),
        "noisy": obs + rng.normal(scale=1.0, size=obs.size),
        "biased": obs + 5.0,
        "scaled": obs * 1.5,
        "biased_and_scaled": obs * 0.7 + 2.0,
    }
    for name, pred in cases.items():
        expected = kge(pred, obs)
        c = kge_components(pred, obs)
        reconstructed = 1.0 - np.sqrt((c["r"] - 1) ** 2 + (c["alpha"] - 1) ** 2 + (c["beta"] - 1) ** 2)
        assert np.isclose(reconstructed, expected, atol=1e-9), (
            f"{name}: reconstructed KGE {reconstructed} != kge() {expected} (components={c})"
        )
        logger.info("PASS (%s): components=%s -> reconstructed=%.6f == kge()=%.6f",
                    name, {k: round(v, 4) for k, v in c.items()}, reconstructed, expected)


def test_degenerate_inputs_return_nan() -> None:
    """Degenerate inputs return all-NaN components rather than raising."""
    obs = np.array([1.0, 1.0, 1.0])  # zero variance
    pred = np.array([1.0, 2.0, 3.0])
    c = kge_components(pred, obs)
    assert all(np.isnan(v) for v in c.values()), f"expected all-NaN for zero-variance obs, got {c}"
    logger.info("PASS: zero-variance obs -> all components NaN")

    too_short = kge_components(np.array([1.0]), np.array([1.0]))
    assert all(np.isnan(v) for v in too_short.values()), f"expected all-NaN for <2 points, got {too_short}"
    logger.info("PASS: <2 valid points -> all components NaN")


if __name__ == "__main__":
    test_components_reconstruct_kge_scalar()
    test_degenerate_inputs_return_nan()
    logger.info("ALL CHECKS PASSED")
