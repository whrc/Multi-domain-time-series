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


def _resolve_mode(cfg: dict) -> dict:
    """Merge the active mode profile into each section and remove both profile sub-dicts."""
    mode = cfg.get("mode", "dev")
    other = "production" if mode == "dev" else "dev"
    for section in ("preprocessing", "model", "training"):
        sec = cfg.get(section)
        if sec is None:
            continue
        profile = sec.pop(mode, {})
        sec.pop(other, None)
        sec.update(profile)
    return cfg
