# Code Audit & Remediation Report — 2026-06-26

**Scope:** Full adversarial read of all Python files and configs in the repo (shared core, all domain pipelines, multi-domain model). Conducted prior to starting any production runs.

**Result summary:** Two critical bugs found and fixed in the multi-domain code (neither has executed yet — multi-domain is "Not Started"). Single-domain pipelines are clean. Two low-severity items corrected.

---

## Bug 1 — CRITICAL (runtime crash): Multi-domain MLP head dimension mismatch

**File changed:** `domains/multi_domain/model.py`

**Root cause:** The head-builder loop in `MultiDomainModel.__init__` used `common_dim` as the input dimension for every hidden layer. After the first hidden layer, the actual tensor flowing through has shape `head_hidden_dim`, not `common_dim`. With `head_num_layers = 2` (production config), the second `nn.Linear(common_dim, head_hidden_dim)` receives a tensor of the wrong shape and raises a `RuntimeError`. Dev mode (`head_num_layers = 1`) ran through one iteration only, so the bug never triggered in dev testing.

**Incorrect code:**
```python
for _ in range(head_num_layers):
    layers += [nn.Linear(common_dim, head_hidden_dim), nn.GELU()]
layers.append(nn.Linear(head_hidden_dim, spec["nTargets"]))
```

**Corrected code:**
```python
in_dim = common_dim
for _ in range(head_num_layers):
    layers += [nn.Linear(in_dim, head_hidden_dim), nn.GELU()]
    in_dim = head_hidden_dim
layers.append(nn.Linear(in_dim, spec["nTargets"]))
```

The fix also correctly handles `head_num_layers = 0` (no hidden layers): the final `nn.Linear` receives `common_dim` directly as input, which is the correct dimension.

**Impact if undetected:** Multi-domain training would crash immediately on the first forward pass in production mode, before any training occurred. No silent data corruption — pure crash.

**Verified:** The fix was applied. Confirmed by reading the corrected file.

---

## Bug 2 — CRITICAL (silent scientific error): Arctic target labels wrong in multi-domain code

**Files changed:** `domains/multi_domain/02_train.py`, `domains/multi_domain/03_predict.py`, `domains/multi_domain/04_evaluate.py`

**Root cause:** Three multi-domain files hardcoded the arctic target names as `["GPP", "RECO", "ALD", "VEGC"]`. The arctic pkl files store targets in config order (`config/arctic_domain.yaml` order): `[ALD, GPP, RECO, VEGC]`. The loss and gradient computation are positional (column 0 of model output vs column 0 of target tensor), so training itself would have been mathematically correct. The bug only affected labels.

**Consequences:**
1. **Logged per-target losses** during training/evaluation would show "GPP loss" when tracking ALD, and vice versa — misleading monitoring.
2. **Saved NetCDF files** (`03_predict.py → _save_arctic()`): prediction column 0 (actual ALD values) would be written as a variable named "GPP" and saved to `..._GPP_...nc`. All four arctic target output files would have wrong variable names.
3. **Evaluation temporal filter** (`04_evaluate.py → arctic_metrics()`): the function uses `ARCTIC_YEARLY = {"ALD", "VEGC"}` to decide whether to apply a January-only mask. With wrong label order:
   - Column 0 = ALD (yearly target) was labeled "GPP" → NOT in ARCTIC_YEARLY → evaluated at ALL months instead of January-only → wrong metrics for ALD
   - Column 2 = RECO (monthly target) was labeled "ALD" → IN ARCTIC_YEARLY → evaluated at JANUARY ONLY → 11/12 of RECO's data silently discarded → wrong metrics for RECO

This would have produced scientifically incorrect NSE/KGE/RMSE/PBIAS for all four arctic targets in any multi-domain evaluation run.

**Fix applied:** Changed `["GPP", "RECO", "ALD", "VEGC"]` to `["ALD", "GPP", "RECO", "VEGC"]` in `DOMAIN_TARGET_NAMES` in all three files. Also fixed the local `target_names = [...]` inside `_save_arctic()` in `03_predict.py`.

**Verified:** Bug was caught before any multi-domain training run, so no output files were produced with wrong labels. No data to re-generate.

---

## Item 3 — Config correction: Multi-domain production batch size

**File changed:** `config/multi_domain.yaml`

**Rationale:** In the multi-domain pretrain loop, `steps_per_epoch = len(arctic_train_loader)`, so the epoch length is keyed to arctic data volume. Arctic is the dominant domain by data volume (millions of windows in production). The individual arctic model uses `batch_size = 1024` (validated for A100 throughput). The multi-domain config had `batch_size = 256`, which would be unnecessarily slow on the A100 for a dataset of arctic scale.

**Change:** `batch_size: 256` → `batch_size: 1024` in the production training block. A comment was added: `# matches arctic individual model — arctic dominates epoch length`.

---

## Item 4 — Spec-code divergence: Amazon description Step 1 §5

**File changed:** `domains/amazon_domain/amazon_description.md`

**Issue:** Step 1 §5 originally said "Ensure temporal completeness — reindex each station to its full monthly range; insert NaN rows for any gaps; log a warning per station with count of inserted rows." The actual `01_preprocess.py` does not insert NaN rows. Instead it detects breaks in the monthly ordinal sequence (`np.diff(ords) != 1`) and splits data into contiguous segments at those breaks. A single global gap count is logged (not per-station).

**The code is correct and better than the original spec.** Inserting NaN rows into predictors that are fully observed would be wrong — it would introduce synthetic missing values requiring imputation. The contiguous-segment approach correctly handles gaps by treating each run of consecutive months as an independent segment for the sliding-window dataset.

**Fix:** Updated §5 in the description to accurately describe the break-detection + segment-splitting approach. No code changes.

---

## Items Reviewed and Found Clean

The following were carefully checked and are scientifically correct:

| Area | Finding |
|---|---|
| Causal masking (`transformer.py`) | Upper-triangular `−inf` mask, `diagonal=1` — correct |
| Sinusoidal PE for odd `hidden_dim` | Slice on div term handles odd dim — correct |
| Masked MSE loss | Graph-connected zero on empty batch; no silent NaN propagation — correct |
| Scaler inverse-transform | `pred * std[-n:] + mean[-n:]` with targets always in last n columns — correct |
| Arctic grid-stratified split | Both SSP records for a pixel land in the same split — correct |
| Arctic scaler fit before subsampling | Scaler always fit on full train pool, then learning-curve subsampling occurs — correct |
| Arctic CO2 interpolation | `ffill().bfill()` correctly covers Dec 2100 extrapolation — correct |
| Arctic ALD/VEGC projected time labels | Wrong 1901-label override to 2025– implemented correctly — correct |
| `records_to_segments` dual format | Arctic (`data`) and Amazon/Rangeland (`segments`) both handled — correct |
| `predict_last_position` ordering | No shuffle in inference DataLoader, sequential `wi` index maps each element back — correct |
| NSE/KGE/PBIAS edge cases | Zero-variance obs, zero mean obs, zero pred std all return NaN explicitly — correct |
| Multi-domain Stage 2 freeze | Freezes transformer + projections; trains only domain heads sequentially per domain — correct |
| Multi-domain pretrain loss aggregation | Sum of 3 domain losses / 3 in normalized space → comparable magnitudes across domains — correct |
| MLflow sidecar run_id | Written before training loop begins, so a crash still records the run ID — correct |
| Config mode resolution | Fails loudly on missing mode key or missing profile block — correct |
| Amazon station split | Station-level (not temporal), so station climatology computed from own data is not leakage — correct |

---

## Scientific Concerns (No Code Change — Monitor in Production)

### ALD/VEGC training signal imbalance
With `seq_len = 12`, every 12-month window contains exactly one January. `masked_mse_loss` averages over all non-NaN positions: GPP and RECO each contribute 12 positions per window; ALD and VEGC each contribute 1. Effective gradient share: ALD ~4%, VEGC ~4%, GPP ~46%, RECO ~46%.

This is by design (the spec is explicit — no resampling). Monitor `val_loss_ALD` and `val_loss_VEGC` per-target curves after the first arctic production run. If these targets underfit significantly relative to GPP/RECO, revisit with a weighted loss or upsampling strategy.

### Variance formula in arctic `fit_scaler`
The streaming formula `std = sqrt(clip(ss/c − mean², 0, None))` is mathematically equivalent to the population std but susceptible to catastrophic cancellation when mean is large and variance is small. Float64 arithmetic and the `clip(..., 0, None)` guard mitigate the risk in practice for climate variables. Log `min(feature_std)` after the scaler fit as a sanity check during the production run.

---

## What to Do Before Starting Multi-Domain Training

1. ✅ Bug 1 fixed — production `head_num_layers = 2` will no longer crash
2. ✅ Bug 2 fixed — arctic target labels correct; evaluation filters correct
3. ✅ Batch size updated — multi-domain production will use 1024
4. Complete all three single-domain production runs first (per CLAUDE.md: "Work strictly in order")
5. When starting multi-domain, run a dev smoke test first to confirm the model instantiates and forward pass runs without error
