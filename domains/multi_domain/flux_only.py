"""Multi-domain constants, flux-only target-set helpers, and output-path conventions."""

from pathlib import Path

import numpy as np

from config.config import load_config
from domains.arctic_domain._naming import (
    load_stride_seq_len,
    select_flux_scaler_stats,
    select_flux_target_columns,
)

AMAZON_NFEATURES = 14
RANGELAND_NFEATURES = 22
RANGELAND_NFLUX = 4  # GPP, RECO, Rm, Rg — already the first 4 target columns

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
    """Arctic target order from config/arctic_domain.yaml."""
    return [t["name"] for t in load_config("arctic_domain")["targets"]]


def apply_flux_only(domain: str, records: list[dict], scaler: dict) -> tuple[list[dict], dict]:
    """Reduce records/scaler to flux-only targets for arctic/rangeland; amazon is unchanged."""
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
    """Undo Amazon's log1p and discharge/drainage_area scaling (apply after z-score inversion)."""
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
    """Arctic (stride, seq_len) from the pkl's sidecar; cfg.preprocessing.stride does not apply."""
    return load_stride_seq_len(pkl_path)


def domains_suffix(domains: list[str] | None) -> str:
    """Path suffix for ablation runs on a proper subset of DOMAINS; empty for all domains."""
    if domains is None or sorted(domains) == sorted(DOMAINS):
        return ""
    return "_dom-" + "-".join(sorted(domains))


def stage_folder_name(
    stage: str, flux_only: bool, seed: int | None = None, domains: list[str] | None = None,
) -> str:
    name = f"{stage}_fluxonly" if flux_only else stage
    name += domains_suffix(domains)
    return f"{name}_seed{seed}" if seed is not None else name


def checkpoint_path(
    models_dir: Path, stage: str, domain: str | None, flux_only: bool, seed: int | None = None,
    domains: list[str] | None = None,
) -> Path:
    """'pretrained' -> one shared checkpoint (domain ignored); 'finetuned' -> one per domain."""
    folder = models_dir / stage_folder_name(stage, flux_only, seed, domains)
    return folder / "best.pt" if stage == "pretrained" else folder / f"{domain}_best.pt"


def stage_output_dir(
    root: Path, stage: str, domain: str, flux_only: bool, seed: int | None = None,
    domains: list[str] | None = None,
) -> Path:
    """Predictions/evaluation dir, always nested by domain (including the pretrained stage)."""
    return root / stage_folder_name(stage, flux_only, seed, domains) / domain


def pretrain_shared_dir(
    root: Path, flux_only: bool, seed: int | None = None, domains: list[str] | None = None,
) -> Path:
    """Dir for non-domain-specific pretrain artifacts (e.g. the LR finder plot)."""
    return root / stage_folder_name("pretrained", flux_only, seed, domains)
