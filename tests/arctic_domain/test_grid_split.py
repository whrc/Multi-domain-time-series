"""Pure-function regression checks for assign_grid_splits() in 01_preprocess.py.

No pytest dependency (none exists in this project yet — see requirements.txt) and no GCS
calls — assign_grid_splits() only needs a dict of grid -> (lat, lon) centroids, so these run
in milliseconds. Each check is a plain function; run this file directly to execute all of them.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "_pp01", REPO_ROOT / "domains" / "arctic_domain" / "01_preprocess.py"
)
_pp01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp01)
assign_grid_splits = _pp01.assign_grid_splits


def _synthetic_centroids(n_grids: int = 240, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Grids spread across a realistic Arctic latitude range (55N-83N), random longitudes."""
    rng = np.random.default_rng(seed)
    lats = rng.uniform(55, 83, size=n_grids)
    lons = rng.uniform(-180, 180, size=n_grids)
    return {f"H{i}_V{i}": (float(lats[i]), float(lons[i])) for i in range(n_grids)}


def test_grid_exclusive_membership():
    centroids = _synthetic_centroids()
    pp = {"random_seed": 42, "train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20, "split_lat_bins": 6}
    result = assign_grid_splits(centroids, pp)
    assert set(result) == set(centroids), "every grid must get exactly one label, no more no less"
    assert all(v in ("train", "val", "test") for v in result.values())
    print("PASS: grid-exclusive membership")


def test_global_ratios_close_to_target():
    centroids = _synthetic_centroids()
    pp = {"random_seed": 42, "train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20, "split_lat_bins": 6}
    result = assign_grid_splits(centroids, pp)
    n = len(result)
    counts = {s: sum(v == s for v in result.values()) for s in ("train", "val", "test")}
    for split_name, target_frac in (("train", 0.60), ("val", 0.20), ("test", 0.20)):
        realized = counts[split_name] / n
        assert abs(realized - target_frac) < 0.03, (
            f"{split_name}: realized {realized:.3f} vs target {target_frac} — too far off "
            f"(counts={counts}, n={n})"
        )
    print(f"PASS: global ratios close to 60/20/20 (realized={ {k: round(v/n, 3) for k, v in counts.items()} })")


def test_no_stratum_systematically_empties_a_split():
    centroids = _synthetic_centroids(n_grids=240)
    pp = {"random_seed": 42, "train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20, "split_lat_bins": 6}
    result = assign_grid_splits(centroids, pp)
    # With 240 grids over 6 strata (~40 grids/stratum) and 20% val/test fractions, every
    # stratum should contribute at least one grid to val and to test.
    lat_edges = np.quantile(
        [centroids[g][0] for g in centroids], np.linspace(0, 1, 7)[1:-1]
    )
    stratum_of = {g: int(np.digitize(centroids[g][0], lat_edges)) for g in centroids}
    for stratum in set(stratum_of.values()):
        members = [g for g, s in stratum_of.items() if s == stratum]
        labels = {result[g] for g in members}
        assert "val" in labels and "test" in labels, (
            f"stratum {stratum} ({len(members)} grids) has no val or no test grid: {labels}"
        )
    print("PASS: every stratum contributes to val and test")


def test_deterministic():
    centroids = _synthetic_centroids()
    pp = {"random_seed": 42, "train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20, "split_lat_bins": 6}
    result_a = assign_grid_splits(centroids, pp)
    result_b = assign_grid_splits(centroids, pp)
    assert result_a == result_b, "same input twice must give identical output"
    print("PASS: deterministic (same seed -> identical split)")


def test_different_seed_changes_split():
    centroids = _synthetic_centroids()
    pp_a = {"random_seed": 42, "train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20, "split_lat_bins": 6}
    pp_b = {**pp_a, "random_seed": 43}
    result_a = assign_grid_splits(centroids, pp_a)
    result_b = assign_grid_splits(centroids, pp_b)
    diffs = sum(1 for g in result_a if result_a[g] != result_b[g])
    assert diffs > 0, "a different seed should produce a different split, not the identical one"
    print(f"PASS: different seed changes the split ({diffs}/{len(result_a)} grids differ)")


def test_small_grid_count_still_works():
    # Mirrors the real risk this design flags: a tiny --grids debug run.
    centroids = {"H1_V10": (65.0, -150.0), "H1_V7": (68.0, -145.0), "H9_V9": (72.0, 90.0),
                 "H14_V6": (60.0, 30.0), "H19_V10": (78.0, 60.0), "H23_V13": (56.0, 120.0)}
    pp = {"random_seed": 42, "train_frac": 0.60, "val_frac": 0.20, "test_frac": 0.20, "split_lat_bins": 6}
    result = assign_grid_splits(centroids, pp)
    assert set(result) == set(centroids)
    counts = {s: sum(v == s for v in result.values()) for s in ("train", "val", "test")}
    print(f"PASS: small grid count (6 grids) runs without error, counts={counts}")
    if counts["train"] == 0:
        print("  WARNING: 0 grids assigned to train with this tiny set — main() raises loudly in this case")


if __name__ == "__main__":
    test_grid_exclusive_membership()
    test_global_ratios_close_to_target()
    test_no_stratum_systematically_empties_a_split()
    test_deterministic()
    test_different_seed_changes_split()
    test_small_grid_count_still_works()
    print("\nALL CHECKS PASSED")
