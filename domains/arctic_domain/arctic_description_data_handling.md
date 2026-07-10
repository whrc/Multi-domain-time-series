# Arctic Preprocessing — Data Handling Strategy

Companion to `arctic_description.md` step 1. Covers only **how data is
fetched, split, sized, and saved** — not experiment results (those live in
`key_findings_log.md`).

## 1. The problem

The Arctic domain spans ~263 grid tiles covering the circumpolar region. 3 —
`H15_V13`, `H17_V18`, `H19_V17` (`KNOWN_BROKEN_GRIDS`) — are permanently
unfetchable. 8 more — `H11_V16`, `H11_V19`, `H14_V15`, `H16_V7`, `H17_V3`,
`H19_V13`, `H19_V18`, `H9_V19` (`FLAKY_GRIDS_20260710`) — are currently
excluded after failing to fetch across ~5 real retry cycles on 2026-07-10,
including one with `fetch_timeout_seconds` raised 180→300; this is a
single-day observation, not confirmed permanent, and is worth re-testing
without the exclusion on a future date. Both lists live in
`01_preprocess.py` and are applied during grid auto-discovery, leaving 252
grids in active use. Each grid holds up to ~10,000 land pixels, each with
decades of monthly data across ~22 variables. Fetching and windowing *all* of
it at once is:

- **Too big** — full production dataset is hundreds of GB.
- **Too slow to iterate on** — raw data lives on GCS, so I/O is slow; waiting
  hours per iteration burns time and (if run on the GPU VM) GPU cost.
- **Easy to make unrepresentative** — grabbing the first N pixels found yields
  a training set from a handful of nearby grids, not a fair sample of the
  whole Arctic.

So preprocessing must produce **smaller, labeled datasets** (e.g. "50K
windows," "500K windows") that are still spread fairly across the whole
circumpolar region — so a model trained on a small dataset sees the same
*kind* of diversity as one trained on a larger one, enabling a learning-curve
sweep (does validation performance saturate as train size grows?).

**Train is always size-capped; there is no uncapped mode.** An uncapped train
set (every pixel, full density) would be hundreds of GB and take hours to
fetch. `preprocessing.train_size` must be a positive int; `01_preprocess.py`
fails loudly otherwise. The config default is the smallest capped size (50K)
so a bare/accidental run stays cheap. Larger learning-curve points (500K, 2M,
...) are opt-in via `--train-size`.

## 2. Pipeline overview: two passes, pass 1 itself in two phases

`01_preprocess.py` runs in two passes over the 252 active grids:

| Pass | Does | Keeps around |
|---|---|---|
| **1a — gather** | Visit every grid once (concurrent, isolated subprocess per grid — see §9). Learn its land-pixel keys, lat/lon, SSP scenarios present, and its own scaler contribution — unconditionally, with no split decision yet. | A small per-grid summary — never the full raw fetch. |
| **1b — decide** (in-memory, no I/O) | Once every grid's summary is in: compute each grid's centroid from its `lat_lon`, latitude-stratify all grids, assign **whole grids** to train/val/test (§3), then do a second in-memory pass over the already-collected summaries to build the pixel-level split map and sum only the train-assigned grids' scaler contributions. | Nothing new fetched — pure computation over pass 1a's output. |
| **2 — fetch & build** | Using pass 1's split assignment and each split's round-robin pixel selection, re-fetch *only the wanted pixels'* data, normalize it, and bucket it into train/val/test. | Filtered, normalized, save-ready records for wanted pixels only. |

Splitting by whole grid (rather than by individual pixel within every grid)
needs every grid's centroid before any split decision can be made — hence
phase 1a (gather everything first) has to be fully separate from phase 1b
(decide once, using all of it). No grid is ever re-fetched just because its
split wasn't known yet; 1b is pure in-memory computation over 1a's cached
output.

Pass 2 makes **no split decisions** — it only filters to the pixels pass 1
already decided are "wanted," normalizes them, and buckets each record into
the split pass 1 already assigned. This lets pass 1 answer "how much data
exists, and how should it be divided?" cheaply, before pass 2 spends
bandwidth on the real fetch.

**Total inventory** (as of the 2026-07-10 grid-level-split run — stable
across train-size/stride changes since it depends only on the grid roster
and split fractions, both fixed; recomputable instantly from the pass-1
cache with zero GCS calls):

| | count |
|---|---|
| grids visited (pass 1a) | 252 |
| grids with land pixels (assigned a split) | 246 |
| grids assigned to train / val / test | 150 (61.0%) / 48 (19.5%) / 48 (19.5%) |
| train pixels (pool, before subsampling) | 771,832 |
| val pixels (pool, before subsampling) | 230,977 |
| test pixels (pool, before subsampling) | 284,643 |

## 3. How the train/val/test split works (pass 1b) — whole grids, latitude-stratified

**History:** the original design (through `AR-sspfix0708` and earlier) split
*pixels within every grid* independently — each grid contributed ~60% of its
own pixels to train, ~20% to val, ~20% to test, so a held-out pixel commonly
sat immediately next to a training pixel in the same ~4km tile. Given strong
spatial autocorrelation in environmental conditions, this made val/test
scores closer to "interpolation within an already-seen region" than genuine
extrapolation to unseen terrain — a real weakness, confirmed by
`AR-gridsplitsweep0710`'s results being much noisier/less decisive than what
replaced it. As of 2026-07-10 (`feat/arctic-grid-level-split`), the split
assigns **whole grids** to train/val/test instead:

1. Compute every grid's centroid `(lat, lon)` from its pass-1a land-pixel
   `lat_lon` values (mean of all its pixels' coordinates).
2. Bin all grids into `preprocessing.split_lat_bins` (config default 6)
   latitude quantile strata — this keeps held-out grids spread across
   climate zones instead of clustering (e.g. val/test both landing entirely
   in the high Arctic).
3. Within each stratum independently: seed an RNG deterministically from
   `(random_seed, crc32(f"lat_stratum_{stratum}"))`, shuffle that stratum's
   grid list, then cut at `train_frac`/`val_frac`/`test_frac` (default
   60/20/20) using `round()` on the stratum's own grid count.
4. Every pixel in a grid inherits that grid's split label:
   `split[(grid, y, x)] = grid_split[grid]` for every land pixel in that
   grid.

`split_lat_bins` auto-clamps to `max(1, num_grids // 5)` so a small `--grids`
debug run (e.g. 6 grids) can't end up with ~1 grid per stratum, which would
send 100% of a stratum to train via `round()` and leave val/test empty.

**A pixel's entire time series — both SSP scenarios — goes to exactly one
split**, inherited from its grid. This matters because the model is a causal
emulator: it needs unbroken history leading up to the point it predicts, so a
pixel's sequence is never cut short or reassigned mid-stream.

Pass 2 reads `split[(grid, y, x)]` back to bucket each fetched record — it
never re-derives or overrides this assignment.

**Every grid lands in exactly one of train/val/test — never split across
them.** This is a deliberate reversal of the old per-grid design: a genuine
spatial-generalization test needs held-out *regions* the model never saw any
part of, not just held-out pixels within regions it trained on elsewhere.
The practical cost is that val/test each draw from a smaller set of grids
(~43-48 each, vs. up to ~246 under the old per-grid split) — §5 covers why
this isn't a representativeness problem once combined with round-robin
subsampling.

**Whether a given train-assigned grid's pixels are actually *selected* into
a size-capped train output additionally depends on round-robin subsampling
(§5)** — assignment (this section) and selection are two different steps.

## 3a. Staggered windowing — spreading pixels across different calendar months

Independent of the grid-level split above: `WindowedDataset` (and the
window-count formula in §5) generates window starts via `range(0, T -
seq_len + 1, stride)`, always starting at position 0 of a pixel's own
segment. Left alone, every pixel in a dataset would sample windows from the
*same* fixed set of calendar months — stride buys pixel/grid diversity, not
temporal diversity, since every pixel's windows land on identical positions
in the calendar.

To fix this, every pixel's raw time series is **trimmed at the front** by a
small, deterministic, pixel-specific offset before windowing:
`offset = crc32(f"{seed}:{grid}:{y}:{x}") % stride`. This shifts where that
pixel's fixed-stride windows fall in the calendar, without any randomness in
window *placement* itself (still `range(0, T', seq_len, stride)` on the
trimmed series) — different pixels just end up phase-shifted relative to
each other. Confirmed beneficial at both 50K and 500K scale under the old
split (`AR-stagger0709`, `AR-500Kstagger0709`) and made **unconditional** as
part of the grid-level-split redesign (no more `--stagger` flag or
vanilla/staggered comparison — every run staggers).

## 4. Leakage guarantees

Given the split mechanism in §3, the following hold:

- **No pixel appears in more than one split's saved file.**
  `split[(grid,y,x)]` is single-valued; pass 2 looks each record up under
  that one key and buckets it once.
- **No window crosses a split boundary.** Splitting is per-pixel, not
  per-timestep — a pixel's full, uncut time series belongs to one split, and
  `WindowedDataset` only slides within one pixel's own contiguous segment. A
  window spanning two pixels (or two splits) is not structurally possible.
- **The scaler (mean/std) is fit only on train-assigned grids.** In phase 1b,
  only grids assigned `"train"` contribute to the running scaler sums;
  val/test grids are skipped for that purpose. Val/test information never
  leaks into normalization stats.
- **Val/test are unaffected by train-side choices.** See §6 — `train_size`
  and train's stride never change which pixels or windows land in val/test.
- **Training data only ever comes from train-assigned grids.** Pass 2's
  train bucket is populated exclusively from records whose grid was assigned
  `"train"` in phase 1b; there is no code path that pulls a val/test-assigned
  grid's pixel into a train output file.
- **Train and val/test are now spatially independent, not just
  deduplicated.** Because whole grids (not pixels within a grid) are assigned
  to a single split (§3), a held-out val/test pixel is never geographically
  adjacent to a training pixel in the same tile — this is the property the
  old per-grid split explicitly lacked (see §3's "History" note). This is a
  genuinely stronger generalization test, not just a leakage guarantee.

## 5. Representativeness: capped datasets that still cover the whole Arctic

A single pixel, fetched at full (monthly, stride=1) density, can produce
**thousands** of windows on its own. A naive "50K window" dataset could come
from a dozen pixels in 2-3 grids — technically 50,000 windows, but
geographically meaningless.

Two mechanisms fix this:

**(a) Coarser stride, applied at counting time.** Every split uses
`capped_stride` instead of stride=1 when counting how many windows each
pixel contributes. A coarser stride makes each pixel "cost" far fewer
windows, so hitting a fixed window budget requires pulling in many more
pixels — and therefore many more grids. **val/test always use the config
default (`capped_stride: 24`) and never change** — see §6. **Train's stride
is swept independently** via `--train-capped-stride`/`--sweep-strides`; a
7-then-9-point sweep at a 50K window budget (`AR-gridsplitsweep0710`,
`AR-gridsplit4005000710`) found `stride=400` the clear winner (best val loss
and best NSE+RMSE on all 4 targets simultaneously among {50, 100, ..., 500}),
confirmed to also win decisively when scaled to 500K
(`AR-500Kstride400-0710`) — this is the current production choice, though it
is a per-run CLI/config choice, not a hardcoded constant.

  *Note:* for a given pixel, window starts are **not** randomized —
  `WindowedDataset` always generates them via `range(0, T - seq_len + 1,
  stride)`, starting at position 0 relative to that pixel's own segment.
  Since every pixel's segment begins at the same calendar date, every pixel
  in a dataset samples windows from the *same* set of calendar months —
  stride buys pixel/grid diversity, not extra temporal diversity beyond those
  fixed positions.

**(b) Round-robin subsampling, run separately per split.** Once a split's
pixel pool is known (from §3's assignment), round-robin subsampling cycles
one pixel from grid A, one from grid B, one from grid C, ... — pulling from
that split's own pool only — until the split's window budget is met. Because
the three pools are already mutually exclusive (§3), no separate "exclude
val/test picks from train candidates" step is needed.

Together, (a) and (b) mean a "50K" train set ends up with a handful of
pixels spread across *most or all* of train's ~150 assigned grids, instead
of thousands of pixels from a few — and likewise for val/test across their
own ~43-48 assigned grids (§3).

**Every run visits every grid during pass 1** — there's no early exit once
"enough" data is found, because stopping early would break representativeness
before window-budget subsampling even runs.

## 6. Val/test: fixed size, generated once, frozen — comparisons depend on this

`val.pkl` and `test.pkl` are always size-capped (default 50,000 windows each,
at `capped_stride`, spread across all grids assigned to that split) —
**regardless of train's size or stride**. Concretely:

- Val/test use their own `val_size`/`test_size` config keys and their own
  grid/pixel pool (filtered to grids assigned `"val"` / `"test"` in §3),
  windowed at `capped_stride` only.
- `train_size` and `train_capped_stride` never appear anywhere in val/test's
  selection code path.

**Val/test must stay byte-identical across every future run** — a stride
sweep, a `500K` vs `50K` comparison, a future `2M` run — so results from
different experiments are ever evaluated against the same held-out
population. This is enforced, not just intended: `01_preprocess.py` compares
the existing `val.pkl`/`test.pkl`'s sidecar (`seed`, `stride`, `seq_len`,
`size_target`, `grids_hash`, `train_frac`/`val_frac`/`test_frac`,
`split_unit`, `split_lat_bins`) against what the current run's config would
produce. If they match, the existing file is reused untouched — logged as
`"val.pkl already exists and matches config — skipping (cached)"`. **If they
don't match, the run fails loudly with a field-level diff of the mismatched
keys and refuses to proceed**, rather than silently rebuilding a different
val/test population; pass `--force-recompute` to intentionally rebuild.

*History:* earlier in the grid-level-split effort, a mismatch was silently
resolved by rebuilding val/test from scratch — this went unnoticed until a
`stride=400,500` extension run ended up evaluated against a population with
a very different pixel count than the original 7-point sweep, discovered
only via manual inspection (`AR-gridsplit4005000710`). The 7 already-trained
checkpoints had to be retrained from scratch just to get a valid
apples-to-apples comparison. The loud-failure behavior above (commit
`3d19d6e`) exists specifically to make this impossible to miss again.

This is what makes model comparisons fair: every model in a learning-curve
sweep (50K -> 500K -> 2M train pixels) or a stride/density sweep is scored
against the exact same held-out data — and any run since 2026-07-10 that
didn't hit an error was, by construction, scored against it too.

**Train's stride is independently sweepable.** `--train-capped-stride` (or
`--sweep-strides 50,100,150,200,...` to sweep several in one pass) decouples
train's stride from val/test's `--capped-stride`, so comparing
training-density settings never silently changes the held-out population
underneath the comparison. Use `--label` to keep each variant's output files
distinct (e.g. `train_50K_s200.pkl`).

`--sweep-strides` computes each stride's wanted-pixel set up front (cheap,
from the pass-1 cache), takes the union across all strides, fetches each
wanted grid **once**, then splits the union-filtered records locally into one
`train_{label}_s{stride}.pkl` per stride — one GCS pass regardless of how
many strides are compared.

## 7. Caching for resumability

Each pass caches its own per-grid work under
`outputs/arctic_domain/preprocessed/`, so a restart resumes almost instantly
instead of re-fetching from scratch:

| Cache dir | Holds, per grid | Invalidated when |
|---|---|---|
| `.grid_failed_cache/` | An empty `{grid}.failed` marker (timestamp only) | 1 hour after creation — so an exhausted-retries grid isn't retried every re-run, but isn't excluded forever either |
| `.grid_pass1_summary_cache/` | Pixel keys, lat/lon, SSP scenarios present, this grid's scaler-sum contribution (unconditional — no split-gating happens in pass 1a anymore) | A bare `SUMMARY_SCHEMA_VERSION` constant only — **not** `random_seed`, split fractions, or `train_size`. Since the summary is now a pure function of the grid's own raw data, changing the seed, the split fractions, or `split_lat_bins` no longer forces a full 252-grid re-fetch — only phase 1b (cheap, in-memory) re-runs. Bump the schema version to auto-invalidate old-shape entries after a `_grid_summary_from_records` change. |
| `.grid_pass2_records_cache/` | Filtered, normalized, save-ready records for this grid's *wanted* pixels | The wanted `(y, x)` pixel set for this grid changes (automatic whenever `--train-size`, stride, seed, or fractions change the selection — **exact match only, no subset reuse**, so growing a selection re-fetches the whole grid rather than fetching only the delta), **or** the fitted scaler's mean/std fingerprint changes |

`val.pkl`/`test.pkl`/`scaler.pkl` use a separate mechanism (sidecar
comparison, not per-grid caching) — see §6 for their invalidation keys.

Neither pass-1 cache holds the full raw per-grid fetch (multi-GB) — only
small derived data. `.grid_pass2_records_cache/` is the exception: it holds
one fully-fetched, filtered copy of *wanted* records per grid (overwritten
in place as the wanted set grows, not accumulated per distinct request), and
in practice reached ~15GB after a 50K sweep plus a 500K run touching most
train-pool grids. Budget real disk headroom for this cache on top of the
train/val/test `.pkl` files themselves — see §9's disk planning note.

## 8. What gets saved

Every run writes into `outputs/arctic_domain/preprocessed/`:

| File | What it is | Regenerated when |
|---|---|---|
| `train_{label}.pkl` | This run's train set (e.g. `train_50K.pkl`, or `train_50K_s200.pkl` with `--label`) | Every new `--train-size` or `--label` |
| `val.pkl` | Fixed 50K-window validation set | **Never, silently** — only via explicit `--force-recompute` after a sidecar mismatch (§6) |
| `test.pkl` | Fixed 50K-window test set | **Never, silently** — only via explicit `--force-recompute` after a sidecar mismatch (§6) |
| `scaler.pkl` | Train-only mean/std, fit only on train-assigned grids | Recomputed every run from that run's pass-1 pool; **must be copied to the training VM alongside the train pkl** (§9) — it can differ run to run if train-pool grid fetch success differs |

Multiple `train_{label}.pkl` variants can sit on disk side by side — a new
size never deletes or overwrites another. Every `.pkl` gets a
`{name}.meta.json` sidecar recording seed, stride, window length, actual
window count, grids/pixels covered, and split fractions — this is what
`02_train.py` reads to know how the file was built, and what preprocessing
itself checks to decide whether a file needs rebuilding.

## 9. Where this runs

Preprocessing never touches a GPU — it's network- and CPU-bound (fetching +
windowing), not compute-bound — so it runs on the CPU-only `vm-cpu-sandeep`
(32 vCPU / 128GB RAM), not the GPU `vm-sandeep`. See `environment_spec.md`'s
"Compute placement policy."

**Data flow:** preprocess on `vm-cpu-sandeep` -> verify the new pkl's sidecar
-> `scp` directly to `vm-sandeep` (VM-to-VM, never through the laptop,
resolve the destination's internal IP from the laptop first since
`vm-cpu-sandeep` can't resolve it on its own) -> `md5sum` check on both sides
-> train on `vm-sandeep`. `scaler.pkl` and the train pkl's `.meta.json`
sidecar must be copied alongside the train pkl every time.

**Disk planning:** train pkl size scales linearly with actual window count
at ~28.5KB/window (validated at `stride=400`: 50K → 1.3GB, 500K → 12.9GB, a
consistent ratio) — extrapolate for a target size before launching, and
remember `.grid_pass2_records_cache/` grows alongside it (§7, §10) and isn't
reclaimed automatically. A `train_size` bump big enough to need a disk resize
should be planned deliberately, not discovered mid-run — GCE persistent
disks can only grow, never shrink back down. Old `train_{label}.pkl` files
for sizes/strides no longer being iterated on are safe to delete once their
checkpoint is trained and their `val_metrics_*.csv`/figures are pulled — the
pass-2 cache can regenerate them without a fresh GCS fetch, as long as the
exact same wanted-pixel selection (same seed/size/stride) is requested again.

**Memory:** pass 2's fetch concurrency (`--max-workers`) trades off against
RAM — each concurrent grid-fetch worker can peak ~4GB for the largest grids,
and the main process's thread pool holds each in-flight grid's full decoded
data too, so memory scales roughly as `workers x (~4GB + ~3-4GB)`.
`--max-workers 8` (the config default) and `--max-workers 12` (used for
every run in the grid-level-split effort, up to ~56GB RSS observed on a
128GB no-swap VM) both run safely; 24 workers OOM-locked a VM with no swap
configured.

## 10. Resilience

Long runs can get killed unexpectedly by the OS. Because pass 1 and pass 2
each cache their per-grid work as they go (§7), a restart resumes almost
instantly instead of re-fetching everything. `run_preprocess_resilient.sh`
automates relaunching until a run finishes successfully.
