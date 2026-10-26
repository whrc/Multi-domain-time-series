"""Multi-domain model: per-domain projection → shared causal transformer → per-domain MLP head."""

import sys
from pathlib import Path

import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.transformer import TransformerModel  # noqa: E402


class MultiDomainModel(nn.Module):
    """domain_specs: {"arctic": {"nFeatures": int, "nTargets": int}, ...}."""

    def __init__(self, cfg: dict, domain_specs: dict[str, dict]) -> None:
        super().__init__()
        m = cfg["model"]
        common_dim      = m["common_dim"]
        head_hidden_dim = m["head_hidden_dim"]
        head_num_layers = m["head_num_layers"]

        self.projections = nn.ModuleDict({
            d: nn.Linear(spec["nFeatures"], common_dim)
            for d, spec in domain_specs.items()
        })

        # Shared causal encoder: common_dim → common_dim (other dims read from cfg["model"]).
        self.transformer = TransformerModel(
            num_features=common_dim,
            num_targets=common_dim,
            cfg=cfg,
        )

        heads: dict[str, nn.Module] = {}
        for d, spec in domain_specs.items():
            layers: list[nn.Module] = []
            in_dim = common_dim
            for _ in range(head_num_layers):
                layers += [nn.Linear(in_dim, head_hidden_dim), nn.GELU()]
                in_dim = head_hidden_dim
            layers.append(nn.Linear(in_dim, spec["nTargets"]))
            heads[d] = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: Tensor, domain: str) -> Tensor:
        """x: (B, T, nFeatures_d) → (B, T, nTargets_d)."""
        h   = self.projections[domain](x)   # (B, T, common_dim)
        enc = self.transformer(h)            # (B, T, common_dim)
        return self.heads[domain](enc)       # (B, T, nTargets_d)


class DomainRoutedModel(nn.Module):
    """Binds a domain so MultiDomainModel fits the plain `forward(x)` API of shared/inference.py.

    An nn.Module (not a lambda) so `.eval()`/`.to()` propagate to `base`.
    """

    def __init__(self, base: "MultiDomainModel", domain: str) -> None:
        super().__init__()
        self.base = base
        self.domain = domain

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x, domain=self.domain)
