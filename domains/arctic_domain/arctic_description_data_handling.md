# Arctic Preprocessing — Data Handling Strategy

Companion to `arctic_description.md` step 1. Covers only **how data is
fetched, split, sized, and saved** — not experiment results (those live in
`key_findings_log.md`).

## 1. The problem

The Arctic domain spans ~263 grid tiles covering the circumpolar region (260
are fetchable; 3 — `H15_V13`, `H17_V18`, `H19_V17` — are permanently broken).
Each grid holds up to ~10,000 land pixels, each with decades of monthly data
across ~22 variables. Fetching and windowing *all* of it at once is:

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

## 2. Pipeline overview: two passes

`01_preprocess.py` runs in two passes over the 260 grids:

| Pass | Does | Keeps around |
|---|---|---|
| **1 — scan & decide** | Visit every grid once. Learn its land-pixel count. Assign every pixel to train/val/test. Fold train pixels into the running scaler sums. | A small per-grid summary (pixel keys, lat/lon, SSP scenarios present, scaler contribution) — never the full raw fetch. |
| **2 — fetch & build** | Using pass 1's split assignment and each split's round-robin pixel selection, re-fetch *only the wanted pixels'* data, normalize it, and bucket it into train/val/test. | Filtered, normalized, save-ready records for wanted pixels only. |

Pass 2 makes **no split decisions** — it only filters to the pixels pass 1
already decided are "wanted," normalizes them, and buckets each record into
the split pass 1 already assigned. This lets pass 1 answer "how much data
exists, and how should it be divided?" cheaply, before pass 2 spends
bandwidth on the real fetch.

**Total inventory** (stable across runs, same seed/grid list — recomputable
instantly from the pass-1 cache with zero GCS calls):

| | count |
|---|---|
| total land pixels | 1,319,153 |
| train pixels | 791,498 |
| val pixels | 263,829 |
| test pixels | 263,826 |

## 3. How the train/val/test split works (pass 1)

The split is decided **once per pixel, in pass 1, and never revisited**:

1. For each grid, list its unique `(y, x)` pixel keys.
2. Seed a per-grid RNG deterministically from `(random_seed,
   crc32(grid_id))` — the same seed always reproduces the same split for
   that grid, regardless of run order or how many times the assignment is
   recomputed.
3. Shuffle the grid's pixel keys with that RNG, then cut them into
   train/val/test at the configured fractions (default 60/20/20), using
   `round()` on that grid's own pixel count.
4. Record the result as `split[(grid, y, x)] -> "train" | "val" | "test"`.

This runs **independently per grid** — each grid shuffles and splits its own
pixel list, rather than one global split over the pooled circumpolar pixel
pool. That's what makes (nearly) every grid contribute pixels to every split,
instead of one split ending up dominated by a handful of large grids.
*Caveat:* because `round()` operates on each grid's own (sometimes tiny)
pixel count, a grid with only 1-2 land pixels can end up contributing to only
one or two splits — not guaranteed all three.

**A pixel's entire time series — both SSP scenarios — goes to exactly one
split.** The split key is `(grid, y, x)` only, not `(grid, y, x, ssp)`, so
both scenarios for a pixel are always looked up under the same split label.
This matters because the model is a causal emulator: it needs unbroken
history leading up to the point it predicts, so a pixel's sequence is never
cut short or reassigned mid-stream.

Pass 2 reads `split[(grid, y, x)]` back to bucket each fetched record — it
never re-derives or overrides this assignment.

**Does every grid end up in every split's *final* dataset?** Every grid is
*assigned* pixels to train/val/test (step 3 above). Whether a given grid's
pixels are actually *selected* into a size-capped split's output additionally
depends on round-robin subsampling (§5) — assignment and selection are two
different steps.

**This is per-grid stratification, not a global grid-level split.** There is
no notion of "these are train grids, these are val grids, these are test
grids" — every grid (down to the round() caveat above) contributes pixels to
train *and* val *and* test. This is deliberate: a global grid-level split
would risk val/test being dominated by whichever regions those grids happen
to fall in (e.g. mostly one climate zone), which would undercut
representativeness — the same problem §5 solves for dataset size.

*Caveat this creates:* because splitting happens within each grid, a val/test
pixel can sit immediately next to a training pixel in the same tile. Given
spatial autocorrelation in environmental conditions, this is a weaker
generalization test than holding out entire unseen regions — the model may
partly succeed on val/test because it saw very similar nearby conditions
during training, not because it generalized to genuinely novel climate
regimes. "Leakage-free" (§4) means no duplicated data and no cross-split
information in the scaler — it does not mean spatial independence between
splits. This matches the project's stated evaluation approach (held-out
*pixels*, per `CLAUDE.md`), but is worth stating plainly rather than leaving
implicit.

## 4. Leakage guarantees

Given the split mechanism in §3, the following hold:

- **No pixel appears in more than one split's saved file.**
  `split[(grid,y,x)]` is single-valued; pass 2 looks each record up under
  that one key and buckets it once.
- **No window crosses a split boundary.** Splitting is per-pixel, not
  per-timestep — a pixel's full, uncut time series belongs to one split, and
  `WindowedDataset` only slides within one pixel's own contiguous segment. A
  window spanning two pixels (or two splits) is not structurally possible.
- **The scaler (mean/std) is fit only on train pixels.** While iterating a
  grid's records in pass 1, only records whose pixel is assigned `"train"`
  contribute to the running scaler sums; val/test records are skipped for
  that purpose. Val/test information never leaks into normalization stats.
- **Val/test are unaffected by train-side choices.** See §6 — `train_size`
  and train's stride never change which pixels or windows land in val/test.
- **Training data only ever comes from train-assigned pixels.** Pass 2's
  train bucket is populated exclusively from records whose
  `split[(grid,y,x)] == "train"`; there is no code path that pulls a
  val/test-assigned pixel into a train output file.

These guarantees rule out data duplication and information leaking through
the scaler. They do **not** mean train and val/test pixels are spatially
independent — splitting happens *within* each grid (§3), so a held-out pixel
can be geographically adjacent to a training pixel. See §3's caveat for what
that implies about the strength of the generalization test.

## 5. Representativeness: capped datasets that still cover the whole Arctic

A single pixel, fetched at full (monthly, stride=1) density, can produce
**thousands** of windows on its own. A naive "50K window" dataset could come
from a dozen pixels in 2-3 grids — technically 50,000 windows, but
geographically meaningless.

Two mechanisms fix this:

**(a) Coarser stride, applied at counting time.** Every split uses
`capped_stride` (config default 24; production runs typically use a wider
value chosen from sweep results — see `key_findings_log.md`) instead of
stride=1 when counting how many windows each pixel contributes. A coarser
stride makes each pixel "cost" far fewer windows, so hitting a fixed window
budget requires pulling in many more pixels — and therefore many more grids.

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
pixels from *most or all* 260 fetchable grids, instead of thousands of
pixels from a few. The same applies to val and test.

**Every run visits every grid during pass 1** — there's no early exit once
"enough" data is found, because stopping early would break representativeness
before window-budget subsampling even runs.

## 6. Val/test: fixed size, generated once, independent of train

`val.pkl` and `test.pkl` are always size-capped (default 50,000 windows each,
at `capped_stride`, spread across all grids) — **regardless of train's size
or stride**. Concretely:

- Val/test use their own `val_size`/`test_size` config keys and their own
  pixel pool (filtered to `split == "val"` / `"test"`), windowed at
  `capped_stride` only.
- `train_size` and `train_capped_stride` never appear anywhere in val/test's
  selection code path.
- Val/test regenerate **only** when something that would change their
  content changes: `random_seed`, split fractions, the grid list,
  `capped_stride`, `seq_len`, or `val_size`/`test_size` — detected
  automatically via a sidecar comparison, not something to track by hand.

This is what makes model comparisons fair: every model in a learning-curve
sweep (50K -> 500K -> 2M train pixels) or a stride/density sweep is scored
against the exact same held-out data.

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
| `.grid_pass1_summary_cache/` | Pixel keys, lat/lon, SSP scenarios present, this grid's scaler-sum contribution | `random_seed`, `train_frac`, `val_frac`, `test_frac` change — **not** `train_size`, since the summary doesn't depend on how big train will be, so it's reused across every learning-curve size |
| `.grid_pass2_records_cache/` | Filtered, normalized, save-ready records for this grid's *wanted* pixels | The wanted `(y, x)` pixel set for this grid changes (automatic whenever `--train-size`, seed, or fractions change the selection), **or** the fitted scaler's mean/std fingerprint changes |

`val.pkl`/`test.pkl`/`scaler.pkl` use a separate mechanism (sidecar
comparison, not per-grid caching) — see §6 for their invalidation keys.

None of these cache the full raw per-grid fetch (multi-GB) — only small
derived data — so resuming after an interruption never risks filling the
VM's disk.

## 8. What gets saved

Every run writes into `outputs/arctic_domain/preprocessed/`:

| File | What it is | Regenerated when |
|---|---|---|
| `train_{label}.pkl` | This run's train set (e.g. `train_50K.pkl`, or `train_50K_s200.pkl` with `--label`) | Every new `--train-size` or `--label` |
| `val.pkl` | Fixed 50K-window validation set | Only if config/seed/grids change (§6) |
| `test.pkl` | Fixed 50K-window test set | Only if config/seed/grids change (§6) |
| `scaler.pkl` | Train-only mean/std | Recomputed every run from that run's pass-1 pool |

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
-> `scp` directly to `vm-sandeep` (VM-to-VM, never through the laptop) ->
`md5sum` check on both sides -> train on `vm-sandeep`. `scaler.pkl` never
needs copying — both VMs share the same disk lineage.

**Before a big run:** check disk headroom (`df -h`) against the run's
expected pixel count and stride (~82.6KB/pixel at `stride=200`, from
empirical measurement).

**Memory:** pass 2's fetch concurrency (`--max-workers`) trades off against
RAM — each concurrent grid-fetch worker can peak ~4GB for the largest grids,
and the main process's thread pool holds each in-flight grid's full decoded
data too, so memory scales roughly as `workers x (~4GB + ~3-4GB)`.
`--max-workers 8` runs safely; 24 workers OOM-locked a VM with no swap
configured.

## 10. Resilience

Long runs can get killed unexpectedly by the OS. Because pass 1 and pass 2
each cache their per-grid work as they go (§7), a restart resumes almost
instantly instead of re-fetching everything. `run_preprocess_resilient.sh`
automates relaunching until a run finishes successfully.
