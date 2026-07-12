"""
Multi-domain model: per-domain projection → shared causal transformer → per-domain MLP head.

See domains/multi_domain/multi_description.md § "Architecture".
"""

import sys
from pathlib import Path

import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.transformer import TransformerModel  # noqa: E402


class MultiDomainModel(nn.Module):
    """Per-domain input projection → shared transformer → per-domain MLP output head.

    Constructor args:
        cfg          — loaded multi_domain config (dev/production already merged)
        domain_specs — {"arctic": {"nFeatures": int, "nTargets": int}, ...}
    """

    def __init__(self, cfg: dict, domain_specs: dict[str, dict]) -> None:
        super().__init__()
        m = cfg["model"]
        common_dim      = m["common_dim"]
        head_hidden_dim = m["head_hidden_dim"]
        head_num_layers = m["head_num_layers"]

        # Per-domain linear projection: nFeatures_d → common_dim
        self.projections = nn.ModuleDict({
            d: nn.Linear(spec["nFeatures"], common_dim)
            for d, spec in domain_specs.items()
        })

        # Shared causal encoder: common_dim → common_dim.
        # TransformerModel reads hidden_dim, num_heads, feedforward_dim, num_layers, dropout
        # from cfg["model"]. With hidden_dim == common_dim the internal input_proj and output
        # head are identity-dimension linears that are still learned end-to-end.
        self.transformer = TransformerModel(
            num_features=common_dim,
            num_targets=common_dim,
            cfg=cfg,
        )

        # Per-domain MLP heads: common_dim → nTargets_d
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
        """
        x: (batch, seq_len, nFeatures_domain)
        returns: (batch, seq_len, nTargets_domain)
        """
        h   = self.projections[domain](x)   # (B, T, common_dim)
        enc = self.transformer(h)            # (B, T, common_dim)
        return self.heads[domain](enc)       # (B, T, nTargets_d)


class DomainRoutedModel(nn.Module):
    """Adapts MultiDomainModel's `forward(x, domain=...)` to the plain `forward(x)` signature
    `shared/inference.py::predict_last_position` and `shared/evaluate.py::predict_and_inverse`
    expect. A bare `lambda x: model(x, domain=d)` looks equivalent but isn't usable here:
    `predict_last_position` calls `model.eval()` on whatever it's given, and a plain function
    has no such method. Wrapping in an nn.Module with `base` as a real submodule makes
    `.eval()`/`.train()`/`.to()` all propagate correctly to the underlying MultiDomainModel."""

    def __init__(self, base: "MultiDomainModel", domain: str) -> None:
        super().__init__()
        self.base = base
        self.domain = domain

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x, domain=self.domain)
