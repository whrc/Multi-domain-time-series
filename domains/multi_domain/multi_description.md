# Multi-domain deep learning model — architecture design

*Preliminary Working reference Only, Not Source of Truth, Iterate freely.*

---

## 1. Problem

Build one unified model that makes causal same-step predictions (monthly step, inputs
t=1..T → prediction at T) across three land-surface domains with different inputs and
different targets:

| Domain | Units | nFeatures | nTargets | Targets | Data scale | Source |
|--------|-------|-----------|----------|---------|------------|--------|
| Arctic | grid pixels | nStatic + 5 | 4 | ALD, GPP, RECO, VEGC | Very large (263 grids × pixels × 2,400 months) | TEM process model |
| Rangeland | 59 sites | 22 | 10 | GPP, RECO, Rm, Rg, AGB, BGB, AGL, BGL, POC, HOC | Small | RangeSTAR process model |
| Amazon | 98 stations | 14 | 3 | discharge, active_fire_count, burned_area | Small | Observations |

**Status quo:** a separate transformer per domain — this is the individual baseline,
already implemented.

**Goal:** (a) a single deployable model and (b) test whether the data-rich Arctic domain
can regularize and improve predictions in data-scarce Amazon and Rangeland domains through
shared architecture and representation learning. The scientific question is whether a
unified model trained on all three domains outperforms per-domain models trained on each
domain's data alone — particularly for the data-poor domains.

**Data scale:** Arctic ≫ Amazon ≈ Rangeland.

---

## 2. Architecture

### Core design

```
[per-domain linear projection → dim D]
         ↓
[Shared Causal Transformer]   ← shared/transformer.py backbone, unchanged
         ↓
[per-domain MLP head (1–2 hidden layers, GELU)]
         ↓
[per-domain output linear → nTargets]
```

**Per-domain input projection:** a single linear layer per domain maps each domain's
feature vectors from their native dimension to a common embedding dimension D (config
hyperparameter). Each domain has its own projection weights; no weight sharing at this
layer. This is the only place where input-space heterogeneity is resolved.

**Shared causal transformer:** the existing `shared/transformer.py` backbone — a causal
encoder with sinusoidal positional encoding — used unchanged. All domains pass through
the same transformer weights. No domain token is added: the transformer infers domain
identity from the distinct activation patterns produced by the per-domain projections.
Domain tokens are unnecessary here because the three domains project from completely
different feature spaces, producing systematically distinct distributions in the shared
embedding space.

**Per-domain MLP head:** a small MLP (1–2 hidden layers, GELU activation) followed by a
final linear layer to nTargets. Each domain has fully separate head weights. The head
is the only place where domain-specific target semantics are learned.

**Per-domain scaler:** each domain retains its own scaler (fit on that domain's training
split, same as the individual model), applied before the input projection. No unified
scaler.

**Sequence lengths:** each domain uses its own `seq_len` from its config. Batches contain
windows from one domain only (see sampling below), so no padding or variable-length
handling is required. (OR maybe for simplicity all domain could use the same seq_len i guess 12 months is a reasonable for all domains)

---

## 3. Training strategy

### Stage 1 — Pre-training (joint multi-domain)

All three domains train simultaneously through the full architecture. Batches rotate
uniformly across domains in round-robin order (Arctic → Amazon → Rangeland → repeat)
regardless of dataset size. This ensures equal gradient contribution from all three
domains and prevents Arctic (which has orders of magnitude more samples) from dominating
training.

- Each domain's windows are sampled from its own WindowedDataset (per existing pipeline)
- Loss per batch: masked MSE, same formulation as individual domain training
- Validation: mean loss across all three domain validation sets; early stopping on this
- Checkpoint: best shared transformer + all three heads saved together as one file
- Optimizer/scheduler: Adam + cosine LR scheduler, same as individual domain training

### Stage 2 — Fine-tuning (per-domain head)

Load the best Stage 1 checkpoint. Freeze all shared transformer weights and all input
projection weights. Fine-tune each domain's MLP head independently, one domain at a
time, using that domain's **full training split** (no sampling constraint — all available
training windows from that domain). Each head is initialized from the Stage 1 weights
(not random). Fine-tuning runs until validation loss saturates (early stopping per domain).

---

## 4. Experimental comparison

Three model variants, evaluated on each domain's held-out test set:

| Model | Transformer | Head | Training data |
|-------|-------------|------|---------------|
| **Individual** (baseline) | Separate per domain | Separate per domain | Domain's own data only |
| **Unified joint** | Shared (Stage 1) | Per-domain, jointly trained | All three domains (round-robin) |
| **Unified fine-tuned** | Shared, frozen (Stage 1) | Per-domain, fine-tuned separately | Full training split per domain |

**Key scientific questions:**
- Do Unified models outperform Individual for Amazon and Rangeland (data-scarce)?
- Does this come at any cost to Arctic performance?
- Does fine-tuning the heads (Unified fine-tuned) add anything over joint training (Unified joint)?
- Is the benefit stronger for Rangeland (ecologically closer to Arctic) than Amazon?

---

## 5. Data splits

To make the transfer benefit visible, data-scarce domains use constrained training sets.
These splits apply identically to all three model variants so comparisons within each
domain are apples-to-apples. Split unit is spatial (pixel / station / site), same as
individual domain pipelines.

| Domain | Train | Val | Test | Training windows (approx) |
|--------|-------|-----|------|--------------------------|
| Arctic | 70% | 10% | 20% | ~5M (5 production grids); ~6.5M (2 EDA grids) |
| Amazon | 40% | 10% | 50% | ~9,400 (~39 stations) |
| Rangeland | 40% | 10% | 50% | ~1,500 (~24 sites) |

**Arctic grid scope for multi-domain:** the full circumpolar bucket contains **263 grids**
(~1M+ windows/grid → ~263M+ windows total). Using all grids is completely unnecessary
and would be enormous GPU waste. A configurable `arctic_grids` list in `multi_domain.yaml`
with **5 representative grids** gives ~5M training windows — 530× more than Amazon and
3,300× more than Rangeland — ample transfer signal while keeping training tractable.
Grids should be chosen for geographic diversity (different H/V tiles spanning varied
regions of the circumpolar Arctic).

**Rationale for Amazon/Rangeland 40/10/50:** enough training data for the individual
model to learn meaningful signal, but insufficient geographic coverage for good spatial
generalization to held-out locations. The unified model should recover this gap by
leveraging shared representations from Arctic. The stark contrast in training window
counts is by design: round-robin batching equalizes gradient contributions despite this
imbalance.

**Rationale for Arctic 70/10/20:** Arctic is the data-rich anchor. More training data
yields stronger shared transformer representations and stronger transfer signal.

> **Note for later:** The 40/10/50 split for Amazon and Rangeland must also be applied
> when those domains are revisited for the multi-domain experiment — both in their
> `*_description.md` files and their config files. This change is out of scope for the
> current update (only `multi_description.md` is being modified now).

---

## 6. Config

`config/multi_domain.yaml` — key hyperparameters:
- `model.common_dim` — shared embedding dimension D
- `model.head_hidden_dim`, `model.head_num_layers` — MLP head architecture
- `training.pretrain_epochs`, `training.finetune_epochs`
- `training.early_stopping_patience` (applies to both stages)
- `preprocessing.arctic_grids` — explicit list of Arctic grid folders to use for
  multi-domain training (dev: 2 EDA grids; production: 5 representative grids).
  Using all 263 grids is unnecessary — 5 grids provide ~5M windows, already 530×
  more than Amazon and 3,300× more than Rangeland, while keeping training tractable.
- Paths pointing to each domain's preprocessed pkl files and scalers, plus output paths
  for the unified model checkpoint and per-domain fine-tuned heads

---

## 7. Next steps (not part of this brainstorm)

The coding pipeline (01_preprocess.py, 02_train.py, 03_predict.py, 04_evaluate.py) for
the multi-domain model will be specified and implemented as a separate step, following
the same structure as the individual domain pipelines.
