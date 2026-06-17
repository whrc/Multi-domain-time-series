from pathlib import Path
import yaml

DOMAINS = ["arctic_domain", "amazon_domain", "rangeland_domain", "multi_domain"]

_CONFIG_DIR = Path(__file__).parent


def load_config(domain: str) -> dict:
    """Load the YAML config for the given domain and resolve the active mode profile."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of {DOMAINS}.")
    path = _CONFIG_DIR / f"{domain}.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return _resolve_mode(cfg)


_MODES = ("dev", "production")
_PROFILE_SECTIONS = ("preprocessing", "model", "training")


def _resolve_mode(cfg: dict) -> dict:
    """Merge the active mode profile into each section and remove both profile sub-dicts.

    Fails loudly on a missing/invalid `mode`, a missing section, or a section that
    lacks the active mode profile — so misnamed keys surface here, not as a confusing
    KeyError deep inside training.
    """
    mode = cfg.get("mode")
    if mode not in _MODES:
        raise ValueError(f"config 'mode' must be one of {_MODES}, got {mode!r}.")
    other = "production" if mode == "dev" else "dev"
    for section in _PROFILE_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise KeyError(f"config missing required section '{section}'.")
        sec = cfg[section]
        if mode not in sec:
            raise KeyError(f"section '{section}' has no '{mode}' profile (expected a '{mode}:' block).")
        profile = sec.pop(mode)
        sec.pop(other, None)
        sec.update(profile)
    return cfg
