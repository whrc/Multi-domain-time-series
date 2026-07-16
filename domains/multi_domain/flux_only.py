"""
Shared flux-only target-set constants/helpers and output-path conventions for the
multi-domain pipeline.

See domains/multi_domain/multi_description.md § "Flux-Only Variant" and "Outputs". Used by
01_preprocess.py, 02_train.py, 03_predict.py, and 04_evaluate.py so this logic lives in
exactly one place instead of being re-derived per script.
"""

from pathlib import Path

import numpy as np

from config.config import load_config
from domains.arctic_domain._naming import (
    load_stride_seq_len,
    select_flux_scaler_stats,
    select_flux_target_columns,
)

RANGELAND_NFEATURES = 22
RANGELAND_NFLUX = 4  # GPP, RECO, Rm, Rg — already the first 4 trailing target columns, no reorder needed

DOMAINS = ["arctic", "amazon", "rangeland"]
DOMAIN_NTARGETS = {"arctic": 4, "amazon": 3, "rangeland": 10}
DOMAIN_NTARGETS_FLUXONLY = {"arctic": 2, "amazon": 3, "rangeland": RANGELAND_NFLUX}
DOMAIN_TARGET_NAMES = {
    "arctic":    ["ALD", "GPP", "RECO", "VEGC"],
    "amazon":    ["discharge", "active_fire_count", "burned_area"],
    "rangeland": ["GPP", "RECO", "Rm", "Rg", "AGB", "BGB", "AGL", "BGL", "POC", "HOC"],
}
DOMAIN_TARGET_NAMES_FLUXONLY = {
    "arctic":    ["GPP", "RECO"],
    "amazon":    ["discharge", "active_fire_count", "burned_area"],
    "rangeland": ["GPP", "RECO", "Rm", "Rg"],
}
DOMAIN_ID_FIELDS = {
    "arctic":    ["grid", "y", "x", "ssp"],
    "amazon":    ["station_id"],
    "rangeland": ["site", "pft"],
}


def variant_ntargets(flux_only: bool) -> dict[str, int]:
    return DOMAIN_NTARGETS_FLUXONLY if flux_only else DOMAIN_NTARGETS


def variant_target_names(flux_only: bool) -> dict[str, list[str]]:
    return DOMAIN_TARGET_NAMES_FLUXONLY if flux_only else DOMAIN_TARGET_NAMES


def arctic_full_target_order() -> list[str]:
    """Read from config/arctic_domain.yaml (not hardcoded) — matches how
    domains/arctic_domain/02_train.py itself derives `full_target_names`, so this can't drift
    from the individual Arctic pipeline's own target order."""
    return [t["name"] for t in load_config("arctic_domain")["targets"]]


def apply_flux_only(domain: str, records: list[dict], scaler: dict) -> tuple[list[dict], dict]:
    """Reduce records/scaler to the flux-only target subset for arctic/rangeland; amazon is
    returned unchanged (no flux/pool distinction)."""
    if domain == "arctic":
        full_order = arctic_full_target_order()
        records = select_flux_target_columns(records, full_order)
        scaler = select_flux_scaler_stats(scaler, full_order)
    elif domain == "rangeland":
        keep = RANGELAND_NFEATURES + RANGELAND_NFLUX
        records = [{**r, "segments": [seg[:, :keep] for seg in r["segments"]]} for r in records]
        scaler = {"mean": scaler["mean"][:keep], "std": scaler["std"][:keep]}
    elif domain != "amazon":
        raise ValueError(f"apply_flux_only: unknown domain {domain!r}")
    return records, scaler


def inverse_amazon_log1p(
    arr_list: list[np.ndarray], seg_meta: list[dict], target_names: list[str],
) -> list[np.ndarray]:
    """Undo amazon's log1p (+ discharge's drainage-area normalization) on top of the z-score
    inversion predict_and_inverse already applied.

    Amazon's targets are log1p-transformed in domains/amazon_domain/01_preprocess.py before the
    scaler fit, and discharge is additionally divided by drainage_area first — predict_and_inverse
    only undoes the z-score, so callers must apply this afterward. Mirrors
    domains/amazon_domain/04_evaluate.py's ground_truth_long() / 03_predict.py's discharge rescale.
    """
    discharge_idx = target_names.index("discharge")
    out = []
    for arr, meta in zip(arr_list, seg_meta):
        arr = np.expm1(arr)
        arr[:, discharge_idx] *= meta["drainage_area"]
        out.append(arr)
    return out


def arctic_train_pkl_path(cfg: dict) -> Path:
    label = cfg["paths"]["arctic"]["train_label"]
    return Path(cfg["paths"]["arctic"]["preprocessed_dir"]) / f"train_{label}.pkl"


def arctic_stride_seq_len(pkl_path: Path) -> tuple[int, int]:
    """Arctic pkls are size/stride-labeled and carry their own (stride, seq_len) in a sidecar
    (see arctic_description.md) — a global cfg.preprocessing.stride does NOT apply to Arctic.
    Using the wrong stride here silently changes the window count by orders of magnitude (e.g.
    500K windows at the pkl's real stride=400 vs. ~400x more at stride=1), which is exactly the
    kind of resource-scaling mistake CLAUDE.md's Hard Rules warn against."""
    return load_stride_seq_len(pkl_path)


def stage_folder_name(stage: str, flux_only: bool, seed: int | None = None) -> str:
    name = f"{stage}_fluxonly" if flux_only else stage
    return f"{name}_seed{seed}" if seed is not None else name


def checkpoint_path(
    models_dir: Path, stage: str, domain: str | None, flux_only: bool, seed: int | None = None,
) -> Path:
    """stage='pretrained' -> one shared checkpoint (domain ignored); stage='finetuned' -> one
    checkpoint per domain."""
    folder = models_dir / stage_folder_name(stage, flux_only, seed)
    return folder / "best.pt" if stage == "pretrained" else folder / f"{domain}_best.pt"


def stage_output_dir(root: Path, stage: str, domain: str, flux_only: bool, seed: int | None = None) -> Path:
    """Shared layout for predictions_dir/evaluation_dir — always nested by domain (even at the
    pretrained stage, since the one shared checkpoint is still evaluated separately per domain)."""
    return root / stage_folder_name(stage, flux_only, seed) / domain


def pretrain_shared_dir(root: Path, flux_only: bool, seed: int | None = None) -> Path:
    """Non-domain-specific pretrain-stage artifacts (e.g. the LR finder plot, which is
    Arctic-routed but conceptually a pretrain-level, not per-domain, artifact)."""
    return root / stage_folder_name("pretrained", flux_only, seed)
