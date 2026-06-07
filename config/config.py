from pathlib import Path
import yaml

DOMAINS = ["arctic_domain", "amazon_domain", "multi_domain"]

_CONFIG_DIR = Path(__file__).parent


def load_config(domain: str) -> dict:
    """Load and return the YAML config for the given domain."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of {DOMAINS}.")
    path = _CONFIG_DIR / f"{domain}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)
