# Multi-Domain: Unified Transformer Across Arctic, Amazon, and Rangeland

## Overview

Train a single `MultiDomainModel` — per-domain input projections → shared causal transformer → per-domain MLP heads — to make causal same-step monthly predictions across all three domains simultaneously. The scientific goal is to test whether the data-rich Arctic domain regularizes and improves predictions in the data-scarce Amazon and Rangeland domains via shared transformer representations.

Training is two-stage: **Stage 1 (pretrain)** — joint training with mixed-step domain batching; **Stage 2 (finetune)** — frozen shared weights, each domain head fine-tuned independently. Three model variants are compared: **Individual** (per-domain baseline, already implemented), **Unified-joint** (Stage 1), and **Unified-fine-tuned** (Stage 2). Full cross-model comparison is done in a separate root-level script; this pipeline evaluates Stage 1 and Stage 2 only.

Evaluation is by **spatial generalization** — held-out pixels/stations/sites never seen in training, scored across the full time range.

**Config:** `config/multi_domain.yaml`  
**Model:** `domains/multi_domain/model.py`

**Pipeline steps:**
| Step | File | Role |
|------|------|------|
| Pre-flight check | `01_preprocess.py` | Verify individual domain pkl files exist; log sizes |
| Training | `02_train.py` | Stage 1 (pretrain) + Stage 2 (finetune) |
| Prediction | `03_predict.py` | Inference per domain × checkpoint |
| Evaluation | `04_evaluate.py` | Stage 1 vs Stage 2 metrics + plots |

**Prerequisites:** Before running any multi-domain step, complete the individual domain pipelines in full. All three individual domain configs use 60/20/20 train/val/test splits (already set). Arctic uses all grids in production (auto-discovered from GCS bucket) — no grid list needed. Multi-domain reuses their preprocessed pkl files and scalers directly — no separate preprocessing.

**Shared modules used:** `shared/transformer.py`, `shared/dataset.py` (`WindowedDataset`, `records_to_segments`), `shared/training.py` (`masked_mse_loss` only — `train_model` and `_evaluate` are not reusable here due to the domain argument), `shared/inference.py` (`predict_last_position`, via domain-curried wrapper — see Step 3), `shared/evaluate.py` (`predict_and_inverse` via domain-curried wrapper, `per_unit_metrics`), `shared/metrics.py`, `shared/plots.py`, `shared/tracking.py` (MLflow; gated by `mlflow.enabled` in config — off by default).

---

## Config Modes

Set `mode: dev | production` in `config/multi_domain.yaml`. Model and training hyperparameters are selected by mode. Production values are TBD — revisit after initial dev runs.

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
| `training.pretrain_epochs` | Max epochs for Stage 1 |
| `training.finetune_epochs` | Max epochs per domain for Stage 2 |
| `training.early_stopping_patience` | Applies to both stages |
| `training.batch_size` | Per-domain sub-batch size; each optimizer step processes `batch_size` samples from each of the 3 domains = 3 × `batch_size` total samples per step |
| `training.learning_rate` | |
| `training.eval_every_n_epochs` | |
| `training.steps_per_epoch` | Optimizer steps per epoch; each step covers all 3 domains simultaneously; default `len(arctic_train_loader)` means Arctic is fully exhausted each epoch while scarce domains cycle ~2–3× |
| `preprocessing.stride` | Window stride for DataLoaders (same in dev and production — both profiles are identical) |
| `paths.arctic.preprocessed_dir` | `outputs/arctic_domain/preprocessed` |
| `paths.arctic.scaler` | `outputs/arctic_domain/scaler.pkl` |
| `paths.amazon.preprocessed_dir` | `outputs/amazon_domain/preprocessed` |
| `paths.amazon.scaler` | `outputs/amazon_domain/scaler.pkl` |
| `paths.rangeland.preprocessed_dir` | `outputs/rangeland_domain/preprocessed` |
| `paths.rangeland.scaler` | `outputs/rangeland_domain/scaler.pkl` |
| `paths.models_dir` | `outputs/multi_domain/models` |
| `paths.predictions_dir` | `outputs/multi_domain/predictions` |
| `paths.evaluation_dir` | `outputs/multi_domain/evaluation` |

---

## Architecture (`model.py`)

**Class: `MultiDomainModel(nn.Module)`**

Constructor receives `cfg` and `domain_specs: dict[str, dict]`, where each entry maps a domain name to `{"nFeatures": int, "nTargets": int}`.

Three `nn.ModuleDict` components:

1. **`self.projections`** — one `nn.Linear(nFeatures_d, cfg.model.common_dim)` per domain. Maps each domain's native feature dimension to the shared embedding space. No weight sharing here.

2. **`self.transformer`** — `TransformerModel(num_features=cfg.model.common_dim, num_targets=cfg.model.common_dim, cfg=cfg)` from `shared/transformer.py`, used as a shared encoder. `TransformerModel` reads `cfg.model.hidden_dim`, `cfg.model.num_heads`, and `cfg.model.feedforward_dim` — all present in the config. Setting `hidden_dim = common_dim` means the transformer's internal `input_proj` (common_dim → hidden_dim) is an identity-dimension linear sitting between the per-domain projection and the encoder; this is acceptable and learned. Setting `num_targets=common_dim` means the transformer's output head projects back to `common_dim`, giving `(batch, seq_len, common_dim)` embeddings for the per-domain heads. All domains share these weights.

3. **`self.heads`** — one per domain: `nn.Sequential` of `cfg.model.head_num_layers` blocks of `[nn.Linear(common_dim, head_hidden_dim), nn.GELU()]`, followed by `nn.Linear(head_hidden_dim, nTargets_d)`. No weight sharing; each domain learns its own target-space projection.

**`forward(x: Tensor[batch, seq_len, nFeatures_d], domain: str) → Tensor[batch, seq_len, nTargets_d]`:**
1. `h = self.projections[domain](x)` → `(batch, seq_len, common_dim)`
2. `enc = self.transformer(h)` → `(batch, seq_len, common_dim)`
3. `out = self.heads[domain](enc)` → `(batch, seq_len, nTargets_d)`

**Checkpoint format:** `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))`. Stage 1 checkpoint contains all weights. Each Stage 2 checkpoint is a complete `state_dict` with Stage 1 frozen weights plus that domain's fine-tuned head.

---

## Domain Data Reference

Exact feature and target dimensions per domain. Scalers and pkl files are from individual domain pipelines (no multi-domain copies). `nFeatures_arctic` is inferred from loaded data as `records[0]["data"].shape[1] - 4`.

| Domain | nFeatures | nTargets | Feature order | Target col indices | Scaler | PKL dir |
|--------|-----------|----------|---------------|--------------------|--------|---------|
| Arctic | nStatic + 5 | 4 | `[static | CO2 | climate]` | last 4 (`-4:`) | `outputs/arctic_domain/scaler.pkl` | `outputs/arctic_domain/preprocessed/` |
| Amazon | 14 | 3 | `[dynamic | climatological means]` | 14–16 (`14:17`) | `outputs/amazon_domain/scaler.pkl` | `outputs/amazon_domain/preprocessed/` |
| Rangeland | 22 | 10 | `[dynamic | static | PFT | cyclical | site means]` | 22–31 (`22:32`) | `outputs/rangeland_domain/scaler.pkl` | `outputs/rangeland_domain/preprocessed/` |

**Record formats** (as stored in pkl files, carried over from individual domain preprocessing):

- **Arctic:** `List[Dict]` with keys `{grid: str, ssp: str, y: int, x: int, ny: int, nx: int, lat: float, lon: float, data: np.ndarray(T, nFeatures+4)}` — single contiguous array, targets in last 4 columns.
- **Amazon:** `List[Dict]` with keys `{station_id: str, segments: List[np.ndarray(T_seg, 17)], segment_starts: List[Tuple[int,int]]}` — multiple contiguous segments per station.
- **Rangeland:** `List[Dict]` with keys `{site: str, pft: str, segments: List[np.ndarray(T_seg, 32)], segment_starts: List[Tuple[int,int]]}` — multiple contiguous segments per site.

**Data splits (all domains):** 60% train / 20% val / 20% test, split by spatial unit (pixel / station / site). All individual domain configs already use these fractions, so test sets are identical across Individual and Unified model variants.

**Arctic grid scope:** all grids discovered by the individual Arctic pipeline (production: all grids auto-discovered from the GCS bucket). Multi-domain loads the Arctic pkl files as-is.

---

## Step 1 — Pre-flight check (`01_preprocess.py`)

**Goal:** Verify all individual domain preprocessed files exist and report dataset sizes before launching training. Produces no output files.

1. For each domain, verify the following paths exist (raise `FileNotFoundError` with the specific path if missing):
   - `{paths.arctic.preprocessed_dir}/{split}.pkl` for `split` in `[train, val, test]`
   - `{paths.arctic.scaler}`
   - Same pattern for Amazon and Rangeland
2. **Log Arctic grid coverage:** collect the set of unique `grid` values present in the Arctic train pkl records and log them as an informational summary. Raise a warning if the Arctic train pkl appears to contain only one grid (likely a dev-mode run), but do not abort. Use `cfg.model.seq_len` (not `cfg.preprocessing.seq_len` — location differs from individual domain configs).
3. Load each train pkl; count records and compute approximate window count as `sum(len(segments) × ⌊(T_seg − seq_len) / stride + 1⌋ for each segment)`. For Arctic, each record has one segment of length T, so windows = `⌊(T − seq_len) / stride + 1⌋`.
4. Log a summary table: domain, split, record count, approx window count.

No output files written.

---

## Step 2 — Training (`02_train.py`)

The script contains both Stage 1 and Stage 2 logic, separated by a `--stage {pretrain,finetune}` CLI flag.

### Stage 1 — Joint pre-training

**Goal:** Train the full `MultiDomainModel` via mixed-step batching across all three domains. All hyperparameters from `config/multi_domain.yaml`.

1. **Load** per-domain train and val pkl files from the paths in `config/multi_domain.yaml`. Infer `nFeatures_arctic` as `train_records_arctic[0]["data"].shape[1] - 4`. `nFeatures_amazon = 14` (8 dynamic + 6 climatological means), `nFeatures_rangeland = 22` (10 dynamic + 1 static + 4 PFT one-hot + 2 cyclical + 5 site means) — these are structural constants derived from the column lists in their domain configs; update if those column lists change.

   **Normalization:** Each domain's data is normalized independently using its own domain-specific scaler — the same scaler produced by that domain's individual preprocessing pipeline (fit on that domain's training split only). There is no global or cross-domain scaler. Load all three scalers from the config paths under `paths.arctic.scaler`, `paths.amazon.scaler`, and `paths.rangeland.scaler`.

2. **Build `domain_specs`:**
   ```
   domain_specs = {
       "arctic":    {"nFeatures": nFeatures_arctic, "nTargets": 4},
       "amazon":    {"nFeatures": 14,               "nTargets": 3},
       "rangeland": {"nFeatures": 22,               "nTargets": 10},
   }
   ```

3. **Build per-domain `WindowedDataset`** (from `shared/dataset.py`) for train and val, using `seq_len=cfg.model.seq_len` and `stride=cfg.preprocessing.stride`. Arctic records have a single `data` array (treated as one segment); Amazon/Rangeland records have a `segments` list. Each dataset item: `(input, target)` of shape `(seq_len, nFeatures_d)` and `(seq_len, nTargets_d)` respectively.

   **⚠️ Config path difference:** In single-domain configs, sequence length is at `cfg.preprocessing.seq_len`. In the multi-domain config it is at `cfg.model.seq_len`. All dataset construction in the multi-domain pipeline must read `cfg.model.seq_len`, not `cfg.preprocessing.seq_len`.

4. **Build per-domain `DataLoader`** for train and val with `training.batch_size`.

5. **Initialise** `MultiDomainModel(cfg, domain_specs)` from `domains/multi_domain/model.py` (shared transformer feedforward activation: GELU). AdamW optimizer (`training.learning_rate`, `training.weight_decay`); linear warmup for `training.warmup_epochs` epochs then cosine decay (`T_max = training.pretrain_epochs − training.warmup_epochs`). Device: `cuda` if available, else `cpu`.

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
   - If mean val loss improved → `torch.save(model.state_dict(), cfg.paths.models_dir / "stage1_best.pt")`

9. **Post-training plots** using best Stage 1 checkpoint on the val set: loss curves per domain + overall mean; scatter (predicted vs actual) per domain per target; boxplots of RMSE, NSE, KGE, PBIAS per domain. Use `shared/metrics.py` and `shared/plots.py`. Save to `cfg.paths.evaluation_dir/stage1/`. When calling `predict_and_inverse` or `predict_last_position`, pass the domain-curried wrapper (same pattern as Step 3): `domain_model = lambda x: model(x, domain=domain)` — the bare `model` cannot be passed directly as it requires a `domain` argument.

---

### Stage 2 — Per-domain head fine-tuning

**Goal:** Freeze shared weights from Stage 1; fine-tune each domain's MLP head independently. Produces one checkpoint per domain.

1. **Load** Stage 1 checkpoint: reconstruct `MultiDomainModel(cfg, domain_specs)` and call `model.load_state_dict(torch.load(cfg.paths.models_dir / "stage1_best.pt"))`.

2. **Freeze shared weights:** In Stage 2, freeze **both** the shared transformer (`model.transformer.parameters()`) and all domain projection layers (`model.projections.parameters()`). Only the domain-specific MLP heads (`model.heads[domain].parameters()`) remain trainable for each domain's fine-tuning run.
   ```python
   for param in model.transformer.parameters():
       param.requires_grad = False
   for param in model.projections.parameters():
       param.requires_grad = False
   ```

3. **Fine-tune each domain in sequence** (Arctic → Amazon → Rangeland). For each domain `d`:
   a. Build `WindowedDataset` and `DataLoader` for `d`'s full training split (all available training windows; no round-robin; stride from config).
   b. AdamW optimizer with **only** `list(model.heads[d].parameters())`, `lr=cfg.training.learning_rate`, `weight_decay=cfg.training.weight_decay`; linear warmup for `training.warmup_epochs` epochs then cosine decay (`T_max = training.finetune_epochs − training.warmup_epochs`). The other domains' heads retain `requires_grad=True` but have no parameters in this optimizer, so they are not updated — this is correct.
   c. Training loop for `training.finetune_epochs`:
      - `pred = model(inputs, domain=d)` — forward pass with frozen projections + transformer, trainable head only
      - `loss = masked_mse_loss(pred, targets)`
      - Backward + step
   d. Validation every `training.eval_every_n_epochs` epochs: compute val loss for domain `d`'s val set; early stopping per domain on that domain's val loss.
   e. On best val loss → `torch.save(model.state_dict(), cfg.paths.models_dir / f"stage2_{d}_best.pt")` — full `state_dict` (includes frozen weights + fine-tuned head; self-contained for inference).

4. **Post-training plots** per domain: same as Stage 1 post-training plots but for Stage 2 checkpoint. Save to `cfg.paths.evaluation_dir/stage2_{d}/`. Use the domain-curried wrapper inside the loop — `domain_model = lambda x: model(x, domain=d)` — created and used immediately within each iteration (do not store across iterations).

---

## Step 3 — Prediction (`03_predict.py`)

**Goal:** Run inference on the test set for a specified domain and checkpoint stage. Only run when training performance is satisfactory.

CLI: `--domain {arctic,amazon,rangeland}`, `--checkpoint {stage1,stage2}` (default: `stage2`)

1. **Load checkpoint:**
   - Stage 1: `cfg.paths.models_dir / "stage1_best.pt"`
   - Stage 2: `cfg.paths.models_dir / f"stage2_{domain}_best.pt"`
   Reconstruct `MultiDomainModel(cfg, domain_specs)` and load state dict. Set `model.eval()`.

2. **Load** `test.pkl` from `cfg.paths.{domain}.preprocessed_dir / "test.pkl"`. Load scaler from `cfg.paths.{domain}.scaler` as `{"mean": np.ndarray, "std": np.ndarray}`.

3. **Inference** — build domain `WindowedDataset` with **stride = 1** over the test pkl records (dense coverage). Use `predict_last_position` from `shared/inference.py`: for each window, record the prediction only at the last time step (`window_start + seq_len − 1`); first `seq_len − 1` steps of each sequence have no prediction and are filled with NaN.
   - `predict_last_position` calls `model(x)` internally and cannot be used directly with `MultiDomainModel`. Wrap the model with a domain-curried callable before passing:
     ```python
     domain_model = lambda x: model(x, domain=domain)
     preds = predict_last_position(domain_model, dataset, device, batch_size=cfg.training.batch_size)
     ```
     `domain` here is a single CLI string — no closure risk. When the same wrapper is created inside a `for d in [...]` loop (Stage 2 plots), write `lambda x: model(x, domain=d)` and use it immediately in the same iteration; never store lambdas across iterations.
   - The same `domain_model` wrapper must be used anywhere `predict_and_inverse` is called for post-training plots (Step 2).

4. **Inverse-transform** predictions using the domain scaler:
   - Arctic: `pred * std[-4:] + mean[-4:]` (last 4 scaler entries = target columns)
   - Amazon: `pred * std[14:17] + mean[14:17]`
   - Rangeland: `pred * std[22:32] + mean[22:32]`

5. **Save** predictions to `cfg.paths.predictions_dir`:
   - **Arctic:** NetCDF per variable per grid per SSP, matching TEM naming (`ALD_yearly`, `GPP_monthly`, etc.) — same format as `arctic_domain/03_predict.py`. Filename convention: `{domain}_{checkpoint}_{grid}_{ssp}_{variable}.nc`.
   - **Amazon:** parquet with columns `station_id, year, month, discharge_pred, active_fire_count_pred, burned_area_pred`. Filename: `amazon_{checkpoint}_predictions.parquet`.
   - **Rangeland:** parquet with columns `site, date, GPP_predicted, RECO_predicted, Rm_predicted, Rg_predicted, AGB_predicted, BGB_predicted, AGL_predicted, BGL_predicted, POC_predicted, HOC_predicted`. Also derive `NEE_predicted = RECO_predicted − GPP_predicted`. Filename: `rangeland_{checkpoint}_predictions.parquet`.

---

## Step 4 — Evaluation (`04_evaluate.py`)

**Goal:** Compute per-domain metrics for Stage 1 and Stage 2 and produce diagnostic figures. Individual domain evaluation is handled by the individual pipelines; full cross-model comparison (Individual vs Unified-joint vs Unified-fine-tuned) is done in a separate root-level intercomparison script.

1. **Recompute predictions** for both `stage1` and `stage2` directly from checkpoints (no dependency on saved prediction files). For each stage: reconstruct `MultiDomainModel(cfg, domain_specs)`, load the stage checkpoint, call `predict_and_inverse` with the domain-curried wrapper on `test.pkl`; the same call provides both predictions and inverse-transformed observations.
   - Arctic: extract January positions only for ALD and VEGC (ground truth is NaN at non-January timesteps for these variables — they are yearly outputs); use all monthly positions for GPP and RECO.
   - Amazon/Rangeland: use all time steps.

2. **Compute per-unit metrics** for each domain × stage using `shared/metrics.py` (RMSE, NSE, KGE, PBIAS):
   - Arctic: per-pixel per-target per-SSP per-period (`historical` = time < 2025, `projected` = time ≥ 2025)
   - Amazon: per-station per-target
   - Rangeland: per-site per-target
   Schema: `{domain_id_cols, target, stage, RMSE, NSE, KGE, PBIAS}`

3. **Save** metrics CSV per domain per stage: `cfg.paths.evaluation_dir/{domain}_stage1_metrics.csv` and `cfg.paths.evaluation_dir/{domain}_stage2_metrics.csv`.

4. **Produce diagnostic plots** using `shared/plots.py` for each stage independently (same style as individual domain pipelines):
   - Scatter (predicted vs actual) per target variable per stage
   - Boxplots of RMSE, NSE, KGE, PBIAS across units per target per stage
   - Stage 1 plots → `cfg.paths.evaluation_dir/stage1/`; Stage 2 plots → `cfg.paths.evaluation_dir/stage2_{domain}/`
   - Cross-stage and cross-model comparison is handled by the root-level `compare_models.py` script

---

## Entry Point (`run_multi_domain.py`)

`--stage {preprocess, pretrain, finetune, predict, evaluate}`
`--domain {arctic, amazon, rangeland}` — required for `predict`
`--checkpoint {stage1, stage2}` — for `predict`, default `stage2`

Intended workflow (each stage is run independently):
```
python run_multi_domain.py --stage preprocess
python run_multi_domain.py --stage pretrain
# Inspect Stage 1 plots/val metrics before proceeding
python run_multi_domain.py --stage finetune
python run_multi_domain.py --stage predict --domain arctic --checkpoint stage1
python run_multi_domain.py --stage predict --domain arctic --checkpoint stage2
# Repeat predict for amazon and rangeland
python run_multi_domain.py --stage evaluate
```

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/{domain}_domain/preprocessed/{train,val,test}.pkl` | Reused from individual pipelines — no multi-domain copies |
| `outputs/{domain}_domain/scaler.pkl` | Reused from individual pipelines |
| `outputs/multi_domain/models/stage1_best.pt` | Best Stage 1 checkpoint (full model state dict) |
| `outputs/multi_domain/models/stage2_{domain}_best.pt` | Stage 2 checkpoint per domain (frozen base + fine-tuned head) |
| `outputs/multi_domain/predictions/amazon_{stage}_predictions.parquet` | Amazon predictions: station_id, year, month, 3 target columns |
| `outputs/multi_domain/predictions/rangeland_{stage}_predictions.parquet` | Rangeland predictions: site, date, 11 columns (10 targets + NEE) |
| `outputs/multi_domain/predictions/{domain}_{stage}_{grid}_{ssp}_{var}.nc` | Arctic predictions: NetCDF per variable per grid per SSP |
| `outputs/multi_domain/evaluation/stage1/` | Stage 1 post-training diagnostic plots |
| `outputs/multi_domain/evaluation/stage2_{domain}/` | Stage 2 per-domain diagnostic plots |
| `outputs/multi_domain/evaluation/{domain}_stage1_metrics.csv` | Per-unit metrics for Stage 1 (RMSE, NSE, KGE, PBIAS) |
| `outputs/multi_domain/evaluation/{domain}_stage2_metrics.csv` | Per-unit metrics for Stage 2 (RMSE, NSE, KGE, PBIAS) |
