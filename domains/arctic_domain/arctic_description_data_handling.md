# Arctic Preprocessing — Data Handling

How `01_preprocess.py` fetches, splits, sizes, caches, and saves data (companion to `arctic_description.md` Step 1). Results live in `key_findings_log.md`.

## 1. The problem

- ~263 grid tiles; each up to ~10,000 land pixels × decades of monthly data × ~22 variables. Full dataset = hundreds of GB, slow GCS I/O.
- Excluded during auto-discovery (lists in `01_preprocess.py`) → **252 active grids**:
  - `KNOWN_BROKEN_GRIDS` (permanent): `H15_V13`, `H17_V18`, `H19_V17`.
  - `FLAKY_GRIDS_20260710` (failed several retry cycles on one day, incl. extended `fetch_timeout_seconds`; re-test later): `H11_V16`, `H11_V19`, `H14_V15`, `H16_V7`, `H17_V3`, `H19_V13`, `H19_V18`, `H9_V19`.
- Goal: small labeled datasets (50K, 500K windows) that still span the whole circumpolar region, enabling a learning-curve sweep.
- **Train is always size-capped.** `train_size` must be a positive int (else fails loudly). Default 50K keeps accidental runs cheap; larger sizes via `--train-size`.

## 2. Pipeline: two passes

| Pass | Does | Keeps |
|---|---|---|
| **1a — gather** | Visit every grid (concurrent, isolated subprocess per grid, §9): land-pixel keys, lat/lon, SSPs present, scaler contribution. No split decision. | Small per-grid summary |
| **1b — decide** (in memory) | Grid centroids → latitude-stratified whole-grid split (§3) → pixel split map; sum scaler contributions of train grids only. | — |
| **2 — fetch & build** | Re-fetch only wanted (round-robin-selected) pixels, normalise, bucket into the split from 1b. No split decisions. | Save-ready records |

Whole-grid splitting needs every centroid first, hence 1a/1b are separate; no grid is re-fetched for that.

**Inventory** (fixed by grid roster + split fractions; recomputable from the pass-1 cache):

| | count |
|---|---|
| grids visited | 252 |
| grids with land pixels | 246 |
| train / val / test grids | 150 (61.0%) / 48 (19.5%) / 48 (19.5%) |
| train / val / test pixel pools | 771,832 / 230,977 / 284,643 |

## 3. Split — whole grids, latitude-stratified (phase 1b)

Replaced an earlier per-pixel-within-grid split, where held-out pixels sat next to training pixels in the same tile (interpolation, not spatial generalization).

1. Grid centroid = mean lat/lon of its land pixels.
2. Bin grids into `split_lat_bins` (default 6) latitude quantile strata — keeps val/test spread across climate zones. Auto-clamped to `max(1, num_grids // 5)` so small `--grids` runs don't leave val/test empty.
3. Per stratum: RNG seeded `(random_seed, crc32(f"lat_stratum_{stratum}"))`, shuffle, cut at `train_frac/val_frac/test_frac` (60/20/20) with `round()`.
4. `split[(grid, y, x)] = grid_split[grid]` for every land pixel.

- A pixel's full series (both SSPs) is in exactly one split — the causal emulator needs unbroken history.
- Every grid is in exactly one split. Cost: val/test draw from fewer grids (~43–48 each); round-robin (§5) keeps them representative.
- Assignment (here) ≠ selection into a capped output (§5).

## 3a. Staggered windowing

`WindowedDataset` starts windows at `range(0, T − seq_len + 1, stride)`, so unshifted pixels would all sample the same calendar months. Each pixel's series is front-trimmed by `crc32(f"{seed}:{grid}:{y}:{x}") % stride` before windowing (deterministic phase shift). Beneficial at 50K and 500K; unconditional (`--stagger` flag removed).

## 4. Leakage guarantees

- No pixel in more than one split (`split[(grid,y,x)]` is single-valued).
- No window crosses a split: windows slide within one pixel's contiguous series.
- Scaler fit only on train-assigned grids.
- Val/test independent of `train_size`/train stride (§6).
- Train outputs only contain train-assigned grids.
- Train and val/test are spatially independent (whole grids), not just deduplicated.

## 5. Representativeness of capped datasets

One pixel at stride 1 yields thousands of windows, so a naive 50K set could come from 2–3 grids. Fixes:

- **(a) Coarse counting stride.** Windows per pixel are counted at `capped_stride`, so the budget needs many more pixels/grids. Val/test always use the config default (24). Train stride is swept independently (`--train-capped-stride`/`--sweep-strides`); 400 won at 50K over {50, 100, ..., 500} (best val loss and NSE+RMSE on all 4 targets) and at 500K → production choice (CLI/config, not hardcoded).
- **(b) Round-robin per split.** One pixel per grid per pass from that split's own pool until the budget is met. Pools are disjoint (§3), so no cross-split exclusion is needed.

Result: a 50K train set spans most/all of train's ~150 grids; val/test likewise across ~43–48. Pass 1 visits every grid (no early exit).

## 6. Val/test: fixed and frozen

- `val.pkl`/`test.pkl`: `val_size`/`test_size` (50K windows each) at `capped_stride`, from val/test grids only; `train_size`/`train_capped_stride` never touch their code path.
- Must stay identical across all experiments so every model is scored on the same population.
- **Enforced:** the existing sidecar (`seed, stride, seq_len, size_target, grids_hash, train_frac, val_frac, test_frac, split_unit, split_lat_bins`) is compared with the current run. Match → reused (`"val.pkl already exists and matches config — skipping (cached)"`). Mismatch → **fails loudly with a field-level diff**; rebuild only with `--force-recompute`. (Added after a silent rebuild forced retraining a whole stride sweep.)
- `grids_hash` = CRC32 of the sorted grid list (catches `--grids` debug runs or bucket changes).
- **Sidecar** `{name}.meta.json` on every pkl: the fields above plus `size_label, actual_window_count, num_grids_covered, num_pixels` (counted from what was written). `02_train.py` reads `stride`/`seq_len` from it.
- **Train stride sweeps:** `--train-capped-stride` or `--sweep-strides 50,100,...` decouple train from val/test stride; `--label` names variants (e.g. `train_50K_s200.pkl`). `--sweep-strides` computes each stride's wanted pixels from the pass-1 cache, fetches the union once per grid, and writes one `train_{label}_s{stride}.pkl` per stride.

## 7. Resumability caches (`outputs/arctic_domain/preprocessed/`)

| Cache dir | Holds, per grid | Invalidated when |
|---|---|---|
| `.grid_failed_cache/` | Empty `{grid}.failed` marker | After `failed_marker_ttl_seconds` (1 h) |
| `.grid_pass1_summary_cache/` | Pixel keys, lat/lon, SSPs, scaler-sum contribution | `SUMMARY_SCHEMA_VERSION` bump only (not seed/fractions/`split_lat_bins`/`train_size` — those only re-run 1b) |
| `.grid_pass2_records_cache/` | Normalised records for wanted pixels | Wanted `(y, x)` set changes (exact match only; growing it re-fetches the whole grid) or scaler fingerprint changes |

- Pass-1 caches are small; the pass-2 cache is overwritten in place and reached ~15 GB (50K sweep + 500K run). Budget disk for it (§9).
- `val.pkl`/`test.pkl` use sidecar comparison (§6), not these caches.

## 8. Saved files

- `train_{label}.pkl`: new per `--train-size`/`--label`; variants coexist.
- `val.pkl`, `test.pkl`: never regenerated silently (§6).
- `scaler.pkl`: recomputed every run from that run's train pool (can differ if grid fetches differ) — **copy it with the train pkl** to the training VM.

## 9. Where this runs

- CPU-only `vm-cpu-sandeep` (32 vCPU / 128 GB RAM, no swap); see `environment_spec.md` "Compute placement policy".
- **Data flow:** preprocess → verify sidecar → `scp` VM-to-VM to `vm-sandeep` (resolve its internal IP from the laptop first) → `md5sum` both sides → train. Always copy `scaler.pkl` and the `.meta.json` with the train pkl.
- **Disk:** train pkl ≈ 28.5 KB/window (stride 400: 50K → 1.3 GB, 500K → 12.9 GB), plus the pass-2 cache. Plan resizes ahead (GCE disks can't shrink). Old `train_{label}.pkl` files can be deleted once trained and results pulled; the pass-2 cache regenerates them for the same seed/size/stride.
- **Memory:** ≈ `workers × (~4 GB worker + ~3–4 GB in-flight data)`. `--max-workers` 8 (default) and 12 (~56 GB RSS peak) are safe; 24 OOM-locked the VM.

## 10. Resilience

Runs can be killed by the OS; per-grid caches (§7) make restarts near-instant. `run_preprocess_resilient.sh` relaunches until success.
