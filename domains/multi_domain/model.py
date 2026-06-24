# MultiDomainModel: per-domain linear projections → shared CausalTransformer → per-domain MLP heads.
# forward(x, domain) selects the correct input projection and MLP head for the specified domain.
# All domains share the transformer weights; only projections and heads are domain-specific.
