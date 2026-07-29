# Multi-Domain: Unified Transformer Across Arctic, Amazon, and Rangeland

## Overview

Train a single `MultiDomainModel` — per-domain input projections → shared causal transformer → per-domain MLP heads — to make causal same-step monthly predictions across all three domains simultaneously. The scientific goal is to test whether the data-rich Arctic domain regularizes and improves predictions in the data-scarce Amazon and Rangeland domains via shared transformer representations.

Training is two-stage: **pretrain** — joint training with mixed-step domain batching; **finetune** — frozen shared weights, each domain head fine-tuned independently. Three model variants are compared: **Individual** (per-domain baseline, already implemented), **Unified-joint** (pretrain stage), and **Unified-fine-tuned** (finetune stage). Full cross-model comparison (Individual vs. Unified-joint vs. Unified-fine-tuned) is deferred to a future root-level `compare_models.py` script — **not yet implemented**; this pipeline only trains, predicts, and evaluates the pretrain/finetune stages themselves.

Two **target-set variants** are trained end-to-end, each through the full pretrain → finetune → predict → evaluate pipeline: the **full-target** variant (all of each domain's native targets) and a **flux-only** variant (Arctic and Rangeland reduced to their fast-responding flux targets, dropping the slow accumulated-pool targets; Amazon unaffected — see "Flux-Only Variant" below). This mirrors the flux-only mode already adopted in the individual Arctic and Rangeland pipelines.

Evaluation is by **spatial generalization** — held-out pixels/stations/sites never seen in training, scored across the full time range.

**Config:** `config/multi_domain.yaml`
**Model:** `domains/multi_domain/model.py`

**Pipeline steps:**
| Step | File | Role |
|------|------|------|
| Pre-flight check | `01_preprocess.py` | Verify individual domain pkl files exist; log sizes |
| Training | `02_train.py` | pretrain + finetune, `--flux-only` selects the target-set variant |
| Prediction | `03_predict.py` | Inference per domain × checkpoint × target-set variant |
| Evaluation | `04_evaluate.py` | pretrain vs. finetune metrics + plots, per target-set variant |

**Prerequisites:** Before running any multi-domain step, complete the individual domain pipelines in full. All three individual domain configs use 60/20/20 train/val/test splits (already set). Arctic uses all grids in production (auto-discovered from GCS bucket, grid-level latitude-stratified split — see `arctic_description.md`) — no grid list needed. Multi-domain reuses their preprocessed pkl files and scalers directly — no separate preprocessing. **Arctic's train split is not a plain `train.pkl`** — its individual pipeline size/stride-labels train variants (see "Domain Data Reference" below); multi-domain is pinned to the settled production config, `train_500K_s400.pkl`.

**Shared modules used:** `shared/transformer.py`, `shared/dataset.py` (`WindowedDataset`, `records_to_segments`), `shared/training.py` (`masked_mse_loss` only — `train_model` and `_evaluate` are not reusable here due to the domain argument), `shared/inference.py` (`predict_last_position`, via `DomainRoutedModel` — see Step 3 §3), `shared/evaluate.py` (`predict_and_inverse` via `DomainRoutedModel`, `per_unit_metrics`), `shared/metrics.py`, `shared/plots.py`, `shared/tracking.py` (MLflow; gated by `mlflow.enabled` in config — off by default). The flux-only variant additionally reuses `domains/arctic_domain/_naming.py`'s `select_flux_target_columns`/`select_flux_scaler_stats` directly (plain functions, no side effects) — no re-preprocessing needed for Arctic; Rangeland's flux subset is a pure column-truncation (see below). Step 4's Arctic evaluation also reuses `_naming.py`'s `sample_test_pixels`/`save_prediction_sample` (see Step 4 point 5) for the deterministic 50-pixel comparison sample.

---

## Config Modes

Set `mode: dev | production` in `config/multi_domain.yaml`. Model and training hyperparameters are selected by mode. Production values mirror Arctic's individual production architecture (`common_dim=256`, `num_layers=6`, `num_heads=8`, `feedforward_dim=1024`) — Arctic is the most data-heavy domain, so its architecture sets the shared transformer's capacity floor. This is a settled choice, not a placeholder.

---

## Config (`config/multi_domain.yaml`)

Key hyperparameters:

| Key | Description |
|-----|-------------|
| `model.common_dim` | Shared embedding dimension D — all domains project to this |
| `model.hidden_dim` | Transformer's internal hidden dimension; set equal to `common_dim` |
| `model.num_heads` | Number of attention heads (passed to `TransformerModel` as `cfg.model.num_heads`) |
| `model.feedforward_dim` | Transformer feedforward dimension (default: 2 × `common_dim`) |
| `model.head_hidden_dim` | Hidden dim of per-domain MLP heads |
| `model.head_num_layers` | Number of hidden layers in MLP head (1 or 2) |
| `model.seq_len` | Sequence length — 12 (uniform across all domains and all individual pipeline configs, both dev and production) |
| `model.num_layers`, `model.dropout` | Remaining transformer kwargs passed to `shared/transformer.py` |
| `mlflow.enabled` | Whether to log training runs to MLflow (off by default) |
| `training.pretrain_epochs` | Max epochs for the pretrain stage |
| `training.finetune_epochs` | Max epochs per domain for the finetune stage. Production: 50 — reverted from a 2026-07-13 bump to 100 that regressed results (see config comment) |
| `training.early_stopping_patience` | Applies to both stages |
| `training.batch_size` | Per-domain sub-batch size; each optimizer step processes `batch_size` samples from each of the 3 domains = 3 × `batch_size` total samples per step. Used for training only — inference (`03_predict.py`) uses `shared/inference.py`'s own default batch size, not this value (see Step 3). |
| `training.finetune_lr` | Fallback learning rate for the finetune stage when the LR finder isn't reused |
| `training.eval_every_n_epochs` | |
| `training.steps_per_epoch` | Optimizer steps per epoch; each step covers all 3 domains simultaneously; default `len(arctic_train_loader)` means Arctic is fully exhausted each epoch while scarce domains cycle ~2–3× |
| `preprocessing.stride` | Window stride for DataLoaders (same in dev and production — both profiles are identical) |
| `paths.arctic.preprocessed_dir` | `outputs/arctic_domain/preprocessed` |
| `paths.arctic.scaler` | `outputs/arctic_domain/scaler.pkl` |
| `paths.arctic.train_label` | Which of Arctic's size/stride-labeled train variants to load — `"500K_s400"`, i.e. `train_500K_s400.pkl` (the settled production config: 500K windows, grid-level split, `stride=400`) |
| `paths.amazon.preprocessed_dir` | `outputs/amazon_domain/preprocessed` |
| `paths.amazon.scaler` | `outputs/amazon_domain/scaler.pkl` |
| `paths.rangeland.preprocessed_dir` | `outputs/rangeland_domain/preprocessed` |
| `paths.rangeland.scaler` | `outputs/rangeland_domain/scaler.pkl` |
| `paths.models_dir` | `outputs/multi_domain/models` — see "Outputs" for the `pretrained/`/`finetuned/` subfolder layout |
| `paths.predictions_dir` | `outputs/multi_domain/predictions` — same subfolder layout |
| `paths.evaluation_dir` | `outputs/multi_domain/evaluation` — same subfolder layout |

---

## Architecture (`model.py`)

**Class: `MultiDomainModel(nn.Module)`**

Constructor receives `cfg` and `domain_specs: dict[str, dict]`, where each entry maps a domain name to `{"nFeatures": int, "nTargets": int}`. `nTargets` depends on which target-set variant is being trained (see "Flux-Only Variant" below) — `model.py` itself is variant-agnostic; the caller (`02_train.py`) passes the right `nTargets` per variant.

Three `nn.ModuleDict` components:

1. **`self.projections`** — one `nn.Linear(nFeatures_d, cfg.model.common_dim)` per domain. Maps each domain's native feature dimension to the shared embedding space. No weight sharing here.

2. **`self.transformer`** — `TransformerModel(num_features=cfg.model.common_dim, num_targets=cfg.model.common_dim, cfg=cfg)` from `shared/transformer.py`, used as a shared encoder. `TransformerModel` reads `cfg.model.hidden_dim`, `cfg.model.num_heads`, and `cfg.model.feedforward_dim` — all present in the config. Setting `hidden_dim = common_dim` means the transformer's internal `input_proj` (common_dim → hidden_dim) is an identity-dimension linear sitting between the per-domain projection and the encoder; this is acceptable and learned. Setting `num_targets=common_dim` means the transformer's output head projects back to `common_dim`, giving `(batch, seq_len, common_dim)` embeddings for the per-domain heads. All domains share these weights.

3. **`self.heads`** — one per domain: `nn.Sequential` of `cfg.model.head_num_layers` blocks of `[nn.Linear(common_dim, head_hidden_dim), nn.GELU()]`, followed by `nn.Linear(head_hidden_dim, nTargets_d)`. No weight sharing; each domain learns its own target-space projection.

**`forward(x: Tensor[batch, seq_len, nFeatures_d], domain: str) → Tensor[batch, seq_len, nTargets_d]`:**
1. `h = self.projections[domain](x)` → `(batch, seq_len, common_dim)`
2. `enc = self.transformer(h)` → `(batch, seq_len, common_dim)`
3. `out = self.heads[domain](enc)` → `(batch, seq_len, nTargets_d)`

**Checkpoint format:** `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))`. The pretrain checkpoint contains all weights. Each finetune checkpoint is a complete `state_dict` with pretrain-stage weights frozen plus that domain's fine-tuned head.

---

## Domain Data Reference

Exact feature and target dimensions per domain, **full-target variant**. Scalers and pkl files are from individual domain pipelines (no multi-domain copies). `nFeatures_arctic` is inferred from loaded data as `records[0]["data"].shape[1] - 4`.

| Domain | nFeatures | nTargets | Feature order | Target col indices | Scaler | Train pkl |
|--------|-----------|----------|---------------|--------------------|--------|---------|
| Arctic | nStatic + 5 | 4 | `[static | CO2 | climate]` | last 4 (`-4:`) | `outputs/arctic_domain/scaler.pkl` | `outputs/arctic_domain/preprocessed/train_500K_s400.pkl` |
| Amazon | 14 | 3 | `[dynamic | climatological means]` | 14–16 (`14:17`) | `outputs/amazon_domain/scaler.pkl` | `outputs/amazon_domain/preprocessed/train.pkl` |
| Rangeland | 22 | 10 | `[dynamic | static | PFT | cyclical | site means]` | 22–31 (`22:32`) | `outputs/rangeland_domain/scaler.pkl` | `outputs/rangeland_domain/preprocessed/train.pkl` |

`val.pkl`/`test.pkl` are unlabeled (no size/stride suffix) for all three domains, living at
`outputs/{domain}_domain/preprocessed/{val,test}.pkl`. Full record schema (dict keys,
segment structure), the 60/20/20 split mechanism, and Arctic's grid scope/labeling are owned
by each domain's own `*_description.md`/`amazon_description.md`/`rangeland_description.md` —
not restated here. All three domains already use the same 60/20/20 fractions, so test sets
are identical across the Individual and Unified model variants.

---

## Flux-Only Variant

Mirrors the flux-only mode already adopted in the individual Arctic (`AR-c3aaf88b`) and Rangeland (`RG-5f0c3603`) pipelines: drop each domain's slow, accumulated-**pool** targets and keep only the fast, climate-driven **flux** targets. Selected via a `--flux-only` flag on `02_train.py`, `03_predict.py`, and `04_evaluate.py`. No re-preprocessing is needed for either domain — both reuse the existing full-target pkl/scaler and reduce columns in memory.

| Domain | Full-target `nTargets` | Flux-only `nTargets` | Flux-only targets | Mechanism |
|--------|------------------------|-----------------------|--------------------|-----------|
| Arctic | 4 | 2 | `[GPP, RECO]` | GPP/RECO sit in the *middle* of Arctic's target order (`arctic_description.md`), so records/scaler must be reordered, not just sliced — reuses `_naming.py`'s `select_flux_target_columns`/`select_flux_scaler_stats` (same functions the individual Arctic pipeline uses), reading `full_order` from `config/arctic_domain.yaml` at call time so it can't drift. |
| Amazon | 3 | 3 (unchanged) | `[discharge, active_fire_count, burned_area]` | No flux/pool distinction for Amazon — `--flux-only` has no effect. |
| Rangeland | 10 | 4 | `[GPP, RECO, Rm, Rg]` | Targets are already flux-first (`rangeland_description.md`), so this is a pure truncation to the first 4 target columns — no reordering needed. |

**Output separation:** the flux-only variant runs through the exact same pretrain → finetune → predict → evaluate pipeline as the full-target variant, but writes to sibling `_fluxonly`-suffixed folders so the two variants' checkpoints, predictions, and metrics never collide (e.g. `models/pretrained_fluxonly/` alongside `models/pretrained/` — see "Outputs"). `DOMAIN_TARGET_NAMES` in `02_train.py` gets a flux-only lookup (`arctic: ["GPP","RECO"]`, `rangeland: ["GPP","RECO","Rm","Rg"]`, `amazon` unchanged) so plots/metrics label columns correctly under either variant.

---

## Multi-Seed Publication Runs

`--seed N` on `02_train.py`/`03_predict.py`/`04_evaluate.py` seeds torch/numpy/random (weight
init + minibatch shuffle order; the data split itself is fixed) and appends `_seedN` to
output folder/file names (`checkpoint_path`/`stage_output_dir` in `flux_only.py`), so all
seeds' pretrain/finetune checkpoints, predictions, and metrics coexist. Use the same seed for
a `--stage pretrain` run and its matching `--stage finetune` run, so finetune loads that
seed's pretrain checkpoint. `shared/seed_aggregation.py::aggregate_seed_metrics` rolls up
each seed's metrics CSV into a mean/std-across-seeds summary (fails loudly if any seed is
missing a unit/target combination another seed has). **Current production methodology runs
the flux-only variant across 5 seeds** (`run_seed_sweep.py` at the repo root); the
full-target variant has not been through the seed sweep.

---

## Step 1 — Pre-flight check (`01_preprocess.py`)

**Goal:** Verify all individual domain preprocessed files exist and report dataset sizes before launching training. Produces no output files.

1. For each domain, verify the following paths exist (raise `FileNotFoundError` with the specific path if missing):
   - Val/test: `{paths.{domain}.preprocessed_dir}/{split}.pkl` for `split` in `[val, test]`, for all three domains.
   - Train: `{paths.arctic.preprocessed_dir}/train_{paths.arctic.train_label}.pkl` for Arctic (e.g. `train_500K_s400.pkl`); `{paths.{domain}.preprocessed_dir}/train.pkl` for Amazon and Rangeland (unlabeled — their pipelines don't use size/stride variants).
   - `{paths.arctic.scaler}` and the same pattern for Amazon and Rangeland.
2. **Log Arctic grid coverage:** collect the set of unique `grid` values present in the Arctic train pkl records and log them as an informational summary. Raise a warning if the Arctic train pkl appears to contain only one grid (likely a dev-mode run), but do not abort. Use `cfg.model.seq_len` (not `cfg.preprocessing.seq_len` — location differs from individual domain configs).
3. Load each train pkl; count records and compute approximate window count as `sum(len(segments) × ⌊(T_seg − seq_len) / stride + 1⌋ for each segment)`. For Arctic, each record has one segment of length T, so windows = `⌊(T − seq_len) / stride + 1⌋`. **`stride` here is per-domain** (Arctic reads its own pkl sidecar; Amazon/Rangeland use `cfg.preprocessing.stride`) — same rule and same failure mode as Step 2 §3, which this check exists to catch before training starts.
4. Log a summary table: domain, split, record count, approx window count.

No output files written.

---

## Step 2 — Training (`02_train.py`)

The script contains both pretrain and finetune logic, separated by a `--stage {pretrain,finetune}` CLI flag, and both target-set variants, selected by an orthogonal `--flux-only` flag.

### Pretrain stage — Joint pre-training

**Goal:** Train the full `MultiDomainModel` via mixed-step batching across all three domains. All hyperparameters from `config/multi_domain.yaml`.

1. **Load** per-domain train and val pkl files from the paths in `config/multi_domain.yaml` (Arctic's train file selected via `paths.arctic.train_label`, see Step 1). Infer `nFeatures_arctic` as `train_records_arctic[0]["data"].shape[1] - 4`. `nFeatures_amazon = 14` (8 dynamic + 6 climatological means), `nFeatures_rangeland = 22` (10 dynamic + 1 static + 4 PFT one-hot + 2 cyclical + 5 site means) — these are structural constants derived from the column lists in their domain configs; update if those column lists change. If `--flux-only`, apply Arctic's `select_flux_target_columns`/`select_flux_scaler_stats` and Rangeland's column truncation (see "Flux-Only Variant") to both train and val records before building datasets.

   **Normalization:** Each domain's data is normalized independently using its own domain-specific scaler — the same scaler produced by that domain's individual preprocessing pipeline (fit on that domain's training split only). There is no global or cross-domain scaler. Load all three scalers from the config paths under `paths.arctic.scaler`, `paths.amazon.scaler`, and `paths.rangeland.scaler`; flux-only scaler slicing happens after loading (see above).

2. **Build `domain_specs`** — `nTargets` depends on `--flux-only`:
   ```
   # full-target
   domain_specs = {
       "arctic":    {"nFeatures": nFeatures_arctic, "nTargets": 4},
       "amazon":    {"nFeatures": 14,               "nTargets": 3},
       "rangeland": {"nFeatures": 22,               "nTargets": 10},
   }
   # --flux-only
   domain_specs = {
       "arctic":    {"nFeatures": nFeatures_arctic, "nTargets": 2},
       "amazon":    {"nFeatures": 14,               "nTargets": 3},
       "rangeland": {"nFeatures": 22,               "nTargets": 4},
   }
   ```

3. **Build per-domain `WindowedDataset`** (from `shared/dataset.py`) for train and val, using `seq_len=cfg.model.seq_len`. **Stride is per-domain, not a single global value**: Amazon/Rangeland use `cfg.preprocessing.stride` (they have no size/stride-labeling system); **Arctic's train and val pkls each carry their own stride in a sidecar** (`domains/arctic_domain/_naming.py`'s `load_stride_seq_len`) and that sidecar value must be used instead — `train_500K_s400.pkl` only actually contains ~500K windows when windowed at its real stride (400); windowing it at `cfg.preprocessing.stride` (1, meant for Amazon/Rangeland) would silently inflate the window count by ~400x, making one epoch computationally intractable. Arctic records have a single `data` array (treated as one segment); Amazon/Rangeland records have a `segments` list. Each dataset item: `(input, target)` of shape `(seq_len, nFeatures_d)` and `(seq_len, nTargets_d)` respectively.

   **⚠️ Config path difference:** In single-domain configs, sequence length is at `cfg.preprocessing.seq_len`. In the multi-domain config it is at `cfg.model.seq_len`. All dataset construction in the multi-domain pipeline must read `cfg.model.seq_len`, not `cfg.preprocessing.seq_len`.

4. **Build per-domain `DataLoader`** for train and val with `training.batch_size`.

5. **Initialise** `MultiDomainModel(cfg, domain_specs)` from `domains/multi_domain/model.py` (shared transformer feedforward activation: GELU). AdamW optimizer (`training.initial_lr`/`training.optimized_lr` via the LR finder, `training.weight_decay`); linear warmup for `training.warmup_epochs` epochs then cosine decay (`T_max = training.pretrain_epochs − training.warmup_epochs`). Device: `cuda` if available, else `cpu`.

6. **Mixed-step training loop** — each optimizer step accumulates gradients from all three domains before updating weights. This enforces cross-domain representation pressure at every update, which is the correct inductive bias for knowledge transfer. `batch_size` samples are drawn per domain per step (equal mixing); scarce domains cycle via `itertools.cycle` and are seen ~2–3× per epoch.
   ```
   domain_iters = {d: iter(itertools.cycle(loader)) for d, loader in train_loaders.items()}
   num_steps_per_epoch = cfg.training.steps_per_epoch or len(train_loaders["arctic"])  # null in config → len(arctic_train_loader)
   ```
   **Note on domain imbalance:** Arctic's multi-century data contributes on the order of 10–15× more windows per epoch than Amazon or Rangeland. Scarce domains cycle via `itertools.cycle` to ensure they are seen each epoch, but the shared transformer will be exposed to disproportionately more Arctic-like sequences. Whether this biases learned representations should be assessed empirically in post-training evaluation.
   - Outer loop: `for epoch in range(cfg.training.pretrain_epochs):`
   - Inner loop: `for step in range(num_steps_per_epoch):`
     - `optimizer.zero_grad()`
     - `domain_losses = {}`
     - For each `domain` in `["arctic", "amazon", "rangeland"]`:
       - `inputs, targets = next(domain_iters[domain])`
       - `pred = model(inputs.to(device), domain=domain)` → `(batch, seq_len, nTargets_d)`
       - `domain_losses[domain] = masked_mse_loss(pred, targets)` (from `shared/training.py`; masks NaN positions)
     - `total_loss = sum(domain_losses.values()) / 3`
     - `total_loss.backward()`
     - `optimizer.step()`

7. **Loss logging per epoch:** mean loss per domain separately (accumulated from `domain_losses` each step) plus overall mean `total_loss` per epoch.

8. **Validation** every `training.eval_every_n_epochs` epochs:
   - Write an inline val loop — the shared `_evaluate` helper in `shared/training.py` calls `model(x)` without a domain argument and cannot be used here:
     ```python
     model.eval()
     with torch.no_grad():
         for domain in ["arctic", "amazon", "rangeland"]:
             for x, y in val_loaders[domain]:
                 pred = model(x.to(device), domain=domain)
                 # accumulate masked MSE per domain
     ```
   - Report per-domain val losses + their mean
   - **Early stopping** monitors the **unweighted mean validation loss** across all three domains, computed at every `training.eval_every_n_epochs` epochs. Patience counts the number of consecutive evaluations (not raw epochs) without improvement. Training stops when patience is exhausted.
   - If mean val loss improved → `torch.save(model.state_dict(), cfg.paths.models_dir / ("pretrained_fluxonly" if flux_only else "pretrained") / "best.pt")`

9. **Post-training plots** using the best pretrain-stage checkpoint on the val set: loss curves per domain + overall mean; scatter (predicted vs actual) per domain per target; boxplots of RMSE, NSE, KGE, PBIAS per domain. Use `shared/metrics.py` and `shared/plots.py`. Save to `cfg.paths.evaluation_dir / ("pretrained_fluxonly" if flux_only else "pretrained")`. Wrap the model in `DomainRoutedModel` before calling `predict_and_inverse`/`predict_last_position` — see Step 3 §3.

---

### Finetune stage — Per-domain head fine-tuning

**Goal:** Freeze shared weights from the pretrain stage; fine-tune each domain's MLP head independently. Produces one checkpoint per domain (per target-set variant).

1. **Load** the pretrain-stage checkpoint: reconstruct `MultiDomainModel(cfg, domain_specs)` (with the variant's `domain_specs`, per `--flux-only`) and call `model.load_state_dict(torch.load(cfg.paths.models_dir / ("pretrained_fluxonly" if flux_only else "pretrained") / "best.pt"))`.

2. **Freeze shared weights:** freeze **both** the shared transformer (`model.transformer.parameters()`) and all domain projection layers (`model.projections.parameters()`). Only the domain-specific MLP heads (`model.heads[domain].parameters()`) remain trainable for each domain's fine-tuning run.
   ```python
   for param in model.transformer.parameters():
       param.requires_grad = False
   for param in model.projections.parameters():
       param.requires_grad = False
   ```

3. **Fine-tune each domain in sequence** (Arctic → Amazon → Rangeland). For each domain `d`:
   a. Build `WindowedDataset` and `DataLoader` for `d`'s full training split (all available training windows; no round-robin; stride from config), with the same `--flux-only` column reduction applied as in the pretrain stage.
   b. AdamW optimizer with **only** `list(model.heads[d].parameters())`, `lr=cfg.training.finetune_lr` (or the pretrain stage's found LR, if reused), `weight_decay=cfg.training.weight_decay`; linear warmup for `training.warmup_epochs` epochs then cosine decay (`T_max = training.finetune_epochs − training.warmup_epochs`). The other domains' heads retain `requires_grad=True` but have no parameters in this optimizer, so they are not updated — this is correct.
   c. Training loop for `training.finetune_epochs`:
      - `pred = model(inputs, domain=d)` — forward pass with frozen projections + transformer, trainable head only
      - `loss = masked_mse_loss(pred, targets)`
      - Backward + step
   d. Validation every `training.eval_every_n_epochs` epochs: compute val loss for domain `d`'s val set; early stopping per domain on that domain's val loss.
   e. On best val loss → `torch.save(model.state_dict(), cfg.paths.models_dir / ("finetuned_fluxonly" if flux_only else "finetuned") / f"{d}_best.pt")` — full `state_dict` (includes frozen weights + fine-tuned head; self-contained for inference).

4. **Post-training plots** per domain: same as pretrain-stage post-training plots but for the finetune-stage checkpoint. Save to `cfg.paths.evaluation_dir / ("finetuned_fluxonly" if flux_only else "finetuned") / d`. Wrap in `DomainRoutedModel` as in Step 3 §3.

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set for a specified domain, checkpoint stage, and target-set variant. Only run when training performance is satisfactory.

**⚠ Arctic caution — large output:** like the individual Arctic pipeline's own step 3 (`arctic_description.md`), `--domain arctic` reconstructs a full dense `(time, y, x)` grid per test grid tile and can reach many GB. It is **not required for evaluation** — Step 4 recomputes predictions directly from the checkpoint and never reads this step's output — and `run_multi_domain.py` has no "run everything" bundle that would trigger it accidentally (every stage, including `predict`, is always invoked explicitly). Only run `--stage predict --domain arctic` if you specifically need the gridded NetCDF files; confirm disk headroom first. Amazon/Rangeland predictions are cheap parquet files with no such caution.

CLI: `--domain {arctic,amazon,rangeland}`, `--checkpoint {pretrained,finetuned}` (default: `finetuned`), `--flux-only` (optional flag)

1. **Load checkpoint** (folder selected by `--flux-only`, see "Outputs"):
   - `pretrained`: `cfg.paths.models_dir / ("pretrained_fluxonly" if flux_only else "pretrained") / "best.pt"`
   - `finetuned`: `cfg.paths.models_dir / ("finetuned_fluxonly" if flux_only else "finetuned") / f"{domain}_best.pt"`
   Reconstruct `MultiDomainModel(cfg, domain_specs)` (variant-appropriate `domain_specs`, per "Flux-Only Variant") and load state dict. Set `model.eval()`.

2. **Load** `test.pkl` from `cfg.paths.{domain}.preprocessed_dir / "test.pkl"`. Load scaler from `cfg.paths.{domain}.scaler` as `{"mean": np.ndarray, "std": np.ndarray}`. If `--flux-only`, apply the same column reduction as Step 2 to both records and scaler before building the dataset.

3. **Inference** — build domain `WindowedDataset` with **stride = 1** over the test pkl records (dense coverage). Use `predict_last_position` from `shared/inference.py`: for each window, record the prediction only at the last time step (`window_start + seq_len − 1`); first `seq_len − 1` steps of each sequence have no prediction and are filled with NaN.
   - `predict_last_position`/`predict_and_inverse` call `.eval()`/`.train()` on whatever model they're given and expect a plain `forward(x)` signature — neither works with `MultiDomainModel.forward(x, domain=...)` directly (and a bare `lambda x: model(x, domain=d)` doesn't work either, since a function has no `.eval()`). `domains/multi_domain/model.py::DomainRoutedModel` is an `nn.Module` wrapper that fixes this: `DomainRoutedModel(model, domain)` holds `model` as a real submodule so `.eval()`/`.to()` propagate correctly, and its `forward(x)` calls `model(x, domain=domain)` internally. Use it everywhere a domain-specific model needs to be passed to shared inference/evaluation code (here, and in Step 2's post-training plots). Uses `shared/inference.py`'s own default inference batch size (8192), not `training.batch_size`.

4. **Inverse-transform** predictions using the domain scaler (full-target column ranges; flux-only ranges are the `select_flux_scaler_stats`/truncated equivalents from "Flux-Only Variant"):
   - Arctic: `pred * std[-4:] + mean[-4:]` (last 4 scaler entries = target columns)
   - Amazon: `pred * std[14:17] + mean[14:17]`
   - Rangeland: `pred * std[22:32] + mean[22:32]`

5. **Save** predictions to `cfg.paths.predictions_dir / {stage}[_fluxonly] / {domain}/` — always nested by domain, including at the `pretrained` stage (the one shared pretrain checkpoint is still evaluated separately per domain, and nesting removes any ambiguity between domains' output filenames instead of relying on them happening not to collide) (see "Outputs" for the exact layout):
   - **Arctic:** NetCDF per variable per grid per SSP, matching TEM naming (`ALD_yearly`, `GPP_monthly`, etc. — flux-only only ever writes `GPP_monthly`/`RECO_monthly`) — same format as `arctic_domain/03_predict.py`. Filename convention: `{grid}_{ssp}_{variable}.nc`.
   - **Amazon:** parquet with columns `station_id, year, month, discharge_pred, active_fire_count_pred, burned_area_pred`. Filename: `amazon_predictions.parquet`.
   - **Rangeland:** parquet with columns `site, date, GPP_predicted, RECO_predicted, Rm_predicted, Rg_predicted` (+ `AGB_predicted, BGB_predicted, AGL_predicted, BGL_predicted, POC_predicted, HOC_predicted` for the full-target variant only). Also derive `NEE_predicted = RECO_predicted − GPP_predicted`. Filename: `rangeland_predictions.parquet`.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute per-domain metrics for the pretrain and finetune stages, for both target-set variants, and produce diagnostic figures. Individual domain evaluation is handled by the individual pipelines; full cross-model comparison (Individual vs. Unified-joint vs. Unified-fine-tuned) is deferred to the future `compare_models.py` script (not yet implemented — out of scope for this pipeline and for the initial dev/production runs).

CLI: `--flux-only` (optional flag) selects which target-set variant to evaluate.

1. **Recompute predictions** for both `pretrained` and `finetuned` checkpoints directly (no dependency on saved prediction files). For each stage: reconstruct `MultiDomainModel(cfg, domain_specs)` (variant-appropriate), load the stage checkpoint (see Step 2/3's `_fluxonly`-suffixed folder convention), call `predict_and_inverse` with a `DomainRoutedModel` wrapper (Step 3 §3) on `test.pkl`; the same call provides both predictions and inverse-transformed observations.
   - Arctic: extract January positions only for ALD and VEGC (ground truth is NaN at non-January timesteps for these variables — they are yearly outputs); use all monthly positions for GPP and RECO. (Flux-only Arctic has no ALD/VEGC — use all monthly positions for both targets.)
   - Amazon/Rangeland: use all time steps.

2. **Compute per-unit metrics** for each domain × stage using `shared/metrics.py` (RMSE, NSE, KGE, PBIAS):
   - Arctic: per-pixel per-target per-SSP per-period (`historical` = time < 2025, `projected` = time ≥ 2025)
   - Amazon: per-station per-target
   - Rangeland: per-site per-target
   Schema: `{domain_id_cols, target, stage, RMSE, NSE, KGE, PBIAS}`

3. **Save** metrics CSV per domain per stage, always nested by domain: `cfg.paths.evaluation_dir / {stage}[_fluxonly] / {domain} / f"{domain}_metrics.csv"` (see "Outputs").

4. **Produce diagnostic plots** using `shared/plots.py` for each stage independently (same style as individual domain pipelines), saved alongside the metrics CSV in the same per-domain folder:
   - Scatter (predicted vs actual) per target variable per stage
   - Boxplots of RMSE, NSE, KGE, PBIAS across units per target per stage
   - Cross-stage and cross-model comparison is handled by the future `compare_models.py` script (not yet implemented)

5. **Arctic only — deterministic 50-pixel prediction sample.** Uses `domains/arctic_domain/_naming.py`'s `sample_test_pixels`/`save_prediction_sample` directly (moved there from the individual Arctic `04_evaluate.py` specifically so both pipelines share one implementation) with the seed from `config/arctic_domain.yaml`'s `preprocessing.random_seed` — **not** multi-domain's own `preprocessing.random_seed` (documented as unused/reused-splits-only) — so the sampled sites are byte-for-byte identical to the individual Arctic pipeline's own `prediction_sample.parquet`, making the two runs directly comparable at the same sites. Saved to `{eval_dir}/{stage}[_fluxonly]/arctic/prediction_sample.parquet`. This is deliberately the only per-pixel-detail Arctic output multi-domain produces — dense NetCDF predictions for every test pixel are not generated here, mirroring the individual Arctic pipeline's own opt-in-only stance on that (see `arctic_description.md`'s caution that it can reach hundreds of GB).

---

## Entry Point (`run_multi_domain.py`)

`--stage {preprocess, pretrain, finetune, predict, evaluate}`
`--domain {arctic, amazon, rangeland}` — required for `predict`
`--checkpoint {pretrained, finetuned}` — for `predict`, default `finetuned`
`--flux-only` — optional, selects the flux-only target-set variant for `pretrain`, `finetune`, `predict`, and `evaluate`

**`run_multi_domain.py` does not forward `--seed`** — for multi-seed publication runs, use
`run_seed_sweep.py` instead, which invokes `02_train.py`/`03_predict.py`/`04_evaluate.py`
directly (see "Multi-Seed Publication Runs" above).

Intended workflow (each stage is run independently; run both target-set variants):
```
python run_multi_domain.py --stage preprocess

# Full-target variant
python run_multi_domain.py --stage pretrain
# Inspect pretrain-stage plots/val metrics before proceeding
python run_multi_domain.py --stage finetune
python run_multi_domain.py --stage predict --domain arctic --checkpoint pretrained
python run_multi_domain.py --stage predict --domain arctic --checkpoint finetuned
# Repeat predict for amazon and rangeland
python run_multi_domain.py --stage evaluate

# Flux-only variant (Arctic GPP/RECO, Rangeland GPP/RECO/Rm/Rg; Amazon unchanged)
python run_multi_domain.py --stage pretrain --flux-only
python run_multi_domain.py --stage finetune --flux-only
python run_multi_domain.py --stage predict --domain arctic --checkpoint finetuned --flux-only
# Repeat predict for amazon and rangeland
python run_multi_domain.py --stage evaluate --flux-only
```

---

## Outputs

Pretrain-stage and finetune-stage outputs are kept in separate subfolders (rather than same-folder, prefixed filenames) across `models/`, `predictions/`, and `evaluation/`, so a stage × target-set-variant grid is always easy to navigate and compare later. The flux-only variant mirrors the same layout under `_fluxonly`-suffixed sibling folders.

| Path | Contents |
|------|----------|
| `outputs/arctic_domain/preprocessed/train_500K_s400.pkl` | Reused from the individual Arctic pipeline — settled production config (500K windows, grid-level split, `stride=400`) |
| `outputs/{amazon,rangeland}_domain/preprocessed/train.pkl` | Reused from the individual pipelines — no size/stride labeling |
| `outputs/{domain}_domain/preprocessed/{val,test}.pkl` | Reused from individual pipelines — no multi-domain copies |
| `outputs/{domain}_domain/scaler.pkl` | Reused from individual pipelines |
| `outputs/multi_domain/models/pretrained/best.pt` | Best pretrain-stage checkpoint, full-target variant (full model state dict) |
| `outputs/multi_domain/models/pretrained_fluxonly/best.pt` | Same, flux-only variant |
| `outputs/multi_domain/models/finetuned/{domain}_best.pt` | Finetune-stage checkpoint per domain, full-target variant (frozen base + fine-tuned head) |
| `outputs/multi_domain/models/finetuned_fluxonly/{domain}_best.pt` | Same, flux-only variant |
| `outputs/multi_domain/predictions/pretrained/{domain}/...` | Pretrain-stage predictions, full-target variant, nested by domain (per-domain formats — see Step 3) |
| `outputs/multi_domain/predictions/pretrained_fluxonly/{domain}/...` | Same, flux-only variant |
| `outputs/multi_domain/predictions/finetuned/{domain}/...` | Finetune-stage predictions per domain, full-target variant |
| `outputs/multi_domain/predictions/finetuned_fluxonly/{domain}/...` | Same, flux-only variant |
| `outputs/multi_domain/evaluation/pretrained/{domain}/{domain}_metrics.csv` | Pretrain-stage per-domain metrics + `{domain}_scatter.png`/`{domain}_boxplot.png`, full-target variant |
| `outputs/multi_domain/evaluation/pretrained_fluxonly/{domain}/` | Same, flux-only variant |
| `outputs/multi_domain/evaluation/finetuned/{domain}/{domain}_metrics.csv` | Finetune-stage per-domain metrics + plots, full-target variant |
| `outputs/multi_domain/evaluation/finetuned_fluxonly/{domain}/` | Same, flux-only variant |
| `outputs/multi_domain/evaluation/{stage}[_fluxonly]/arctic/prediction_sample.parquet` | Arctic-only: 50-pixel deterministic obs-vs-predicted time series, same seed/sites as the individual Arctic pipeline's own `prediction_sample.parquet` (see Step 4) |
| `outputs/multi_domain/evaluation/pretrained[_fluxonly]/lr_finder.png` | LR-finder plot from the pretrain stage (not domain-specific — Arctic-routed probe, see Step 2; lives directly in the stage folder, not nested under a domain) |
