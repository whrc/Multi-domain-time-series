# Arctic Preprocessing — Data Handling Strategy

This is a companion to `arctic_description.md` step 1. It focuses only on
**how data is fetched, sized, and saved**, and why the design looks the way it does.

## Total pixel inventory

Across all 260 fetchable grid tiles (of 263 total — 3 are permanently broken/unfetchable:
`H15_V13`, `H17_V18`, `H19_V17`), there are **1,319,153 distinct land pixels** in total.
Pass 1's per-grid stratified 60/20/20 split (see below) divides these into:

| split | pixels |
|---|---|
| train | 791,498 |
| val   | 263,829 |
| test  | 263,826 |

These counts are stable across runs (same seed, same grid list) and can be recomputed
instantly from the pass-1 summary cache (`.grid_pass1_summary_cache/`) with zero GCS calls.

## The problem

The Arctic domain has ~263 grid tiles covering the whole circumpolar region, and each grid
can hold up to ~10,000 land pixels, each with decades of monthly data across ~22 variables.
Fetching and building windows from *all* of that at once is:

- **Too big** — the full production dataset is hundreds of GB.
- **Too slow to iterate on** — the raw data stays on GCS cloud storage so I/O is slow, and you don't want to wait hours just to preprocess and burn pricey GPU time.
- **Easy to make unrepresentative** — if you just grab the first N pixels you find, you'll
  get a training set from a handful of nearby grids, not a fair representative sample of the whole Arctic.

We need a way to build **smaller, labeled datasets** (e.g. "50K windows," "500K windows")
that are still spread fairly and **representative** of the whole circumpolar region, so a model trained on a
small dataset sees the same *kind* of diversity as one trained on a much larger one — such that we can experiment
with increasing training dataset sizes, and use where the validation performance saturates. At the same time, we want
to **avoid wasting GPU time** on preprocessing — it runs on a dedicated CPU-only VM instead (see
"The two-VM workflow" below), with the finished datasets moved to the GPU VM only when ready
for training.

**Train is always size-capped — there is no "full/uncapped" mode.** A truly uncapped train set
(all pixels, full production density) would be hundreds of GB and take hours to fetch, so it's
not supported at all: `preprocessing.train_size` must be a positive int, and `01_preprocess.py`
fails loudly if it isn't. The config default is deliberately the *smallest* capped size (50K),
so a bare/accidental run (no `--train-size` passed) stays cheap — minutes, not hours — instead
of silently becoming the most expensive possible run. Larger learning-curve points (500K, 2M,
...) are opt-in via `--train-size`.

## The two-pass process

Preprocessing (`01_preprocess.py`) runs in two passes:

**Pass 1 — scan and decide the split.** Visit every grid once: fetch it, learn how many land
pixels it has, **assign every pixel to train/val/test** (see "Train/val/test split" below), and
fold its train pixels into the running scaler sums. This pass never keeps the full raw data
around — it only keeps a small summary per grid (a few hundred KB, not GBs) plus the global
per-pixel split assignment.

**Pass 2 — fetch what's actually needed.** Using pass 1's split assignment and each split's
round-robin pixel selection (see below), re-fetch *only those pixels'* data from cloud storage,
normalize it, and save it. Pass 2 makes no split decisions of its own — it only filters to the
already-decided "wanted" pixels, normalizes, and buckets each record into the split pass 1
already assigned it to.

Splitting the work this way means pass 1 can answer "how much do we have, and how should it
be divided?" cheaply, before pass 2 spends time/bandwidth fetching the real data.

### Three caches, one per resumability concern

Each phase caches its own per-grid work under `outputs/arctic_domain/preprocessed/`, so a
restart resumes almost instantly instead of re-fetching from scratch. They're kept separate
because each holds a different kind of data and is invalidated by different things:

| Cache dir | Holds, per grid | Invalidated when |
|---|---|---|
| `.grid_failed_cache/` | Nothing but an empty `{grid}.failed` marker file (its timestamp is the only payload) | 1 hour after creation — so a grid that exhausted fetch retries isn't retried on every single re-run, but also isn't excluded forever in case the failure was transient |
| `.grid_pass1_summary_cache/` | Pass 1's small summary: pixel keys, lat/lon, which SSP scenarios are present, and this grid's contribution to the train-scaler sums (a few hundred KB even for a large grid) | `random_seed`, `train_frac`, `val_frac`, `test_frac` change — **not** `train_size`, since pass 1's summary doesn't depend on how big the train set will be, so it's reused across every learning-curve size |
| `.grid_pass2_records_cache/` | The already-filtered, normalized, save-ready records for this grid's *wanted* pixels only | The exact set of `(y, x)` pixels wanted from that grid changes — which happens automatically whenever `--train-size` (or the seed/fractions) changes the selection |

None of these cache the full raw per-grid fetch (which can be multi-GB) — only small derived
data, so re-running or resuming after an interruption never risks filling up the VM's disk. See
`arctic_description.md` step 1 for the exact code paths.

## Train/val/test split: who decides, and when

**Pass 1 decides the split, once, per pixel — pass 2 never re-decides anything.** For each
grid, pass 1 calls `_grid_split_labels()` on that grid's unique `(y, x)` pixel keys and assigns
every one of them to exactly one of train/val/test. Pass 2 later reads that assignment back out
to bucket each fetched record — it does no split logic of its own.

- **Per-pixel, not per-window.** A pixel's *entire* time series (both SSP scenarios) goes to
  one split — never split within a pixel. This matters because the model is a causal emulator:
  it needs real, unbroken history leading up to the point it predicts, so a pixel's sequence is
  never cut short or reassigned mid-stream.
- **Split fractions are applied per grid, independently** (default 60/20/20) — each grid
  shuffles and splits *its own* pixel list, rather than one global 60/20/20 split over the
  pooled circumpolar pixel pool. This grid-level stratification is what makes every grid
  contribute pixels to (nearly) every split, instead of, say, one split ending up dominated by
  a handful of large grids. Caveat: because `round()` is applied to each grid's own (sometimes
  very small) pixel count, a grid with only 1-2 land pixels can end up contributing to only one
  or two of the three splits, not guaranteed all three.
- **Does every grid end up in every split's final dataset?** Every grid is *assigned* pixels to
  train/val/test (previous point). Whether a given grid's pixels are *selected* into a
  size-capped split's final output additionally depends on round-robin subsampling, next.

## Representativeness: how a "50K" dataset still covers the whole Arctic

A single pixel, fetched at full density, can produce **thousands** of training windows on
its own. If we built a "50K window" dataset naively, it could come from just a dozen pixels
in 2-3 grids — technically 50,000 windows, but geographically meaningless.

Instead, every split (train, val, test) uses a deliberately **coarser stride**
(`capped_stride` in the config, default 24 months instead of 1 — but see "Train's stride vs
val/test's stride" below; a 2026-07 sweep found `stride=200` performs best in practice) when
counting how many windows each pixel contributes. This means each pixel "costs" far fewer
windows, so hitting a window budget requires pulling in **many more pixels — and therefore
many more grids**.

### Window starts are NOT randomized per pixel

For a given pixel, `WindowedDataset` (`shared/dataset.py`) generates window starts via
`range(0, T - seq_len + 1, stride)` — always starting at position 0, relative to that pixel's
own segment. Since every pixel's segment for a given SSP scenario begins at the same absolute
calendar date (a shared time axis across the whole grid), **every pixel in the dataset samples
windows from the identical set of calendar months** — there is no per-pixel phase offset or
randomization. At `stride=200`, pixel A's windows start at months 0, 200, 400, ... and so does
every other pixel's — not staggered. More pixels buys spatial diversity, but *not* additional
temporal diversity beyond whatever those fixed positions happen to cover. Randomizing each
pixel's phase (a random offset in `[0, stride)` before striding) would add temporal diversity
across the population without changing the total window count — a candidate future improvement,
not yet implemented.

On top of that, once a split's pixels are known (from the per-pixel assignment above),
**round-robin subsampling runs separately for train, val, and test** — each already scoped to
that split's own pixel pool, cycling one pixel from grid A, one from grid B, one from grid C,
and so on, until that split's window budget is met. Because the three pools are already
mutually exclusive (a pixel belongs to exactly one split, decided in pass 1), there's no
separate "exclude val/test picks from train candidates" step needed — the exclusivity already
exists before any subsampling happens. So a "50K" train set ends up with a handful of pixels
from *most or all* 263 grids, instead of thousands of pixels from a few — and the same is true
of val and test.

**Important:** pixel time series are never cut short or resampled out of order — a pixel's
data always stays a continuous monthly sequence. "Coarser stride" only changes how many
overlapping windows we slice out of that same continuous sequence, never the sequence
itself.

Because reaching even a modest window budget at this coarser density already requires more
pixels than there are grids, **every run visits every grid during pass 1** — there's no shortcut
that stops early once "enough" data is found. That's intentional: stopping early is exactly
what would break representativeness.

## Wide vs deep: what the sweeps found

Two knobs affect this differently:
- **`capped_stride`** trades width (more pixels/grids) against depth (more windows per pixel)
  *at a fixed window budget* — a coarser stride forces more pixels in to hit the same budget.
- **`train_size`** (at a stride already fixed) mostly widens further — since depth-per-pixel is
  unchanged, a bigger budget just pulls in more pixels at that same depth.

**Density sweep** (`AR-controlledsweep0708` in `key_findings_log.md`): sweeping `stride` from 50
to 250 at a fixed 50K budget found **`stride=200` wins on every target simultaneously** — both
extremes underperform (too deep/narrow at 50, too wide/shallow at 250).

**Size sweep** (50K → 500K done, 2M in progress, all at the winning `stride=200`): widening
further continued to help — 500K beat 50K on 3 of 4 targets and on overall validation loss. 2M
is testing whether this keeps helping or plateaus.

**Caveat:** because depth doesn't currently add temporal diversity (every pixel already samples
the same fixed calendar positions — see "Window starts are NOT randomized" above), width's edge
over depth so far may partly reflect that today's "depth" isn't buying genuinely new
information, just more overlapping views of the same calendar snapshots.

### Pixels touched at `stride=200`, by train size

| train_size | pixels @ stride=200 | windows | grids | source |
|---|---|---|---|---|
| 50K | 2,584 | 43,641 | 221 | actual (`train_50K_s200.meta.json`) |
| 500K | 26,692 | 451,048 | 229 | actual (`train_500K_s200.meta.json`) |
| 2M | ~118,000 (estimate) | ~2,000,000 | ~230 (estimate) | estimated from 500K's empirical ratio (~16.9 windows/pixel); replace with the real sidecar once that run completes |

Even 2M's estimated ~118K pixels is only ~15% of the 791,498-pixel train pool — plenty of
headroom to scale further later if wanted.

## Val and test: fixed size, generated once

`val.pkl` and `test.pkl` are **always size-capped** (50,000 windows each, at `capped_stride`,
spread across all grids) — regardless of how large your training set is, and regardless of what
stride train uses. They are generated **once** and then reused unchanged across every
experiment, including both the learning-curve sweep (train sizes 50K → 500K → 2M → ...) and the
density sweep (train stride 50/100/150/200/250/...). This is what makes model comparisons fair:
every model is judged against the exact same held-out data.

They're only regenerated if something that would invalidate them changes — a different
random seed, different split fractions, or a different grid list — which is detected
automatically (see the cache/sidecar table above), not something you need to track by hand.

### Train's stride vs val/test's stride

`--train-capped-stride` decouples train's stride from val/test's `--capped-stride` (which
always governs val/test, regardless of what train uses). This was added after a 2026-07 bug
was found and fixed: an earlier version tied val/test's pixel subsampling to the *same*
`capped_stride` as train, so every time a training-density experiment changed `capped_stride`,
the held-out val/test population silently changed too — confounding genuine training-density
effects with the held-out population's own composition changing underneath each comparison. Now
val/test are locked once at `--capped-stride` and train can be swept independently via
`--train-capped-stride` (a single value) or `--sweep-strides 50,100,150,200,250` (many values in
one pass — see next section). Use `--label` to give each variant's outputs a distinct name
(e.g. `50K_s200`) so different density/size experiments don't overwrite each other's checkpoints.

### Sweeping many train strides in one pass

Comparing several `capped_stride` values used to mean a full separate GCS fetch per value. The
`--sweep-strides` mode instead computes each stride's wanted-pixel set up front (cheap, from
pass 1's cache), takes the **union** across all of them, fetches each wanted grid **once**, then
splits the union-filtered records locally into one `train_{label}_s{stride}.pkl` per stride —
paying for one GCS pass regardless of how many stride values are compared.

## What gets saved, and where

For every run, `01_preprocess.py` saves into `outputs/arctic_domain/preprocessed/`:

| File | What it is | Regenerated when |
|---|---|---|
| `train_{label}.pkl` | Training set for this run's size/density (e.g. `train_50K.pkl`, `train_500K.pkl`, or a stride-labeled variant like `train_50K_s200.pkl` via `--label`) | Every time you pick a new `--train-size` or `--label` |
| `val.pkl` | Fixed 50K-window validation set | Only if config/seed/grids change |
| `test.pkl` | Fixed 50K-window test set | Only if config/seed/grids change |
| `scaler.pkl` | Normalization stats (mean/std), fit once on the full train pixel pool | Only if config/seed/grids change |

Multiple `train_{label}.pkl` variants can sit on disk side by side — generating a new size
never deletes or overwrites another size, so you can copy just the one you need to the GPU
VM for training.

Every pkl also gets a small `{name}.meta.json` **sidecar** recorded next to it, storing the
seed, stride, window length, actual window count, how many grids/pixels it covers, and the
split fractions used. This sidecar is what training (`02_train.py`) reads to know exactly
how that file was built (rather than assuming), and what preprocessing itself checks to
decide "is this file still valid, or does it need to be rebuilt?"

## The two-VM workflow

Preprocessing never touches a GPU — it's just fetching data from cloud storage and
building/normalizing windows, which is network- and CPU-bound, not compute-bound. Running it
on the GPU VM (`vm-sandeep`) would tie up an expensive GPU-hour resource for work that never
uses the GPU at all.

As of 2026-07-08, preprocessing runs on a dedicated CPU-only VM (`vm-cpu-sandeep`,
`n2-standard-32`, 32 vCPU / 128GB RAM, ~$1.55/hr vs `vm-sandeep`'s ~$3.67/hr) — it gets a fast
in-cloud network path to the GCS bucket (vs. a laptop's home/office connection) and far more
real parallel fetch workers, at a fraction of the GPU VM's cost. See `environment_spec.md`'s
"Compute placement policy" for the general CPU-work/GPU-work split this follows.

**Data flow:** preprocess on `vm-cpu-sandeep` → verify the new pkl's sidecar → direct VM-to-VM
`scp` to `vm-sandeep`, over the internal network, never routed through the laptop (a one-time
SSH key trust was set up between the two VMs for this) → `md5sum` checked equal on both sides →
train on `vm-sandeep`. `scaler.pkl` never needs copying, since both VMs share the same disk
lineage and already have an identical one.

**Before a big run, check disk headroom:** `df -h` on the target VM, then use the pixel-count
table above (and its pixel-to-storage ratio, ~82.6KB/pixel at `stride=200`) to estimate a new
run's footprint before launching it — this is exactly how the 2M run's storage was
sanity-checked beforehand.

**Memory caution:** pass 2's fetch concurrency (`--max-workers`) trades off against RAM — each
concurrent grid-fetch worker can peak around ~4GB for the largest grids (dense 100x100 tiles),
and the main process's own thread pool (same `--max-workers` count) holds each in-flight grid's
full decoded data too, so total memory scales roughly as `workers x (~4GB + ~3-4GB)`. A
24-worker run OOM-locked the VM entirely (no swap configured) in 2026-07; `--max-workers 8` has
run safely and repeatably since.

Along the way, only small **derived** data is ever cached to disk (per-grid summaries and
per-grid selected-pixel records — megabytes to low GB, not the multi-GB raw fetch) so
re-running or resuming after an interruption never risks filling up the VM's disk.

## Resilience

Long preprocessing runs can occasionally get killed unexpectedly by the OS. Because pass 1 and pass
2 each cache their per-grid work to disk as they go (see the three-cache table above), a
restart resumes almost instantly from where it left off instead of re-fetching everything
from scratch. See `arctic_description.md` step 1 for the exact cache layout, and
`run_preprocess_resilient.sh` for the wrapper script that automates relaunching until a run
finishes successfully.
