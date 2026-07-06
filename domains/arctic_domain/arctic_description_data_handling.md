# Arctic Preprocessing — Data Handling Strategy

This is a companion to `arctic_description.md` step 1. It focuses only on
**how data is fetched, sized, and saved**, and why the design looks the way it does.

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
to **avoid wasting GPU time** on preprocessing, which can be done on a
laptop instead, and the preprocessed datasets moved to the GPU VM only when ready for training.

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
data, so re-running or resuming after an interruption never risks filling up local disk. See
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
(`capped_stride` in the config, currently 24 months instead of 1) when counting how many
windows each pixel contributes. This means each pixel "costs" far fewer windows, so hitting a
window budget requires pulling in **many more pixels — and therefore many more grids**.

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

## Val and test: fixed size, generated once

`val.pkl` and `test.pkl` are **always size-capped** (50,000 windows each, at the coarser
stride, spread across all grids) — regardless of how large your training set is. They are
generated **once** and then reused unchanged across every experiment, including the
learning-curve sweep (train sizes 50K → 500K → 2M → ...). This is what makes model
comparisons across different training sizes fair: every model is judged against the exact
same held-out data.

They're only regenerated if something that would invalidate them changes — a different
random seed, different split fractions, or a different grid list — which is detected
automatically (see the cache/sidecar table above), not something you need to track by hand.

## What gets saved, and where

For every run, `01_preprocess.py` saves into `outputs/arctic_domain/preprocessed/`:

| File | What it is | Regenerated when |
|---|---|---|
| `train_{label}.pkl` | Training set for this run's size (e.g. `train_50K.pkl`, `train_500K.pkl`, `train_2M.pkl`) | Every time you pick a new `--train-size` |
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

## Why this runs on a laptop, not the GPU VM

Preprocessing never touches a GPU — it's just fetching data from cloud storage and
building/normalizing windows, which is network- and CPU-bound, not compute-bound. It's also
slow: a run can take a while end-to-end (mostly waiting on data fetches, not
crunching numbers). Running that on the GPU VM would tie up an expensive GPU-hour resource
for work that never uses the GPU at all — a waste of money for no benefit.

So preprocessing is done locally instead: any size-capped variant (50K, 500K, 2M) is small
(tens of MB to a few GB), so it's practical to build and copy it from a laptop. Once a variant
is ready, the finished `.pkl`/`.meta.json`/`scaler.pkl` files are simply copied over to the VM,
where `02_train.py` loads them directly — the VM's GPU time is spent only on training, never
on data prep.

Along the way, only small **derived** data is ever cached to disk (per-grid summaries and
per-grid selected-pixel records — megabytes, not the multi-GB raw fetch) so re-running or
resuming after an interruption never risks filling up local storage.

## Resilience

Long local runs can occasionally get killed unexpectedly by the OS. Because pass 1 and pass
2 each cache their per-grid work to disk as they go (see the three-cache table above), a
restart resumes almost instantly from where it left off instead of re-fetching everything
from scratch. See `arctic_description.md` step 1 for the exact cache layout, and
`run_preprocess_resilient.sh` for the wrapper script that automates relaunching until a run
finishes successfully.
