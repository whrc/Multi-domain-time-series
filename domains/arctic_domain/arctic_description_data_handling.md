# Arctic Preprocessing — Data Handling Strategy

This is a plain-language companion to `arctic_description.md` step 1. It focuses only on
**how data is fetched, sized, and saved**, and why the design looks the way it does.

## The problem

The Arctic domain has ~263 grid tiles covering the whole circumpolar region, and each grid
can hold up to ~10,000 land pixels, each with decades of monthly data across ~22 variables.
Fetching and building windows from *all* of that at once is:

- **Too big for a laptop** — the full production dataset is tens of GB.
- **Too slow to iterate on** — you don't want to wait hours just to test a smaller model.
- **Easy to make unrepresentative** — if you just grab the first N pixels you find, you'll
  get a training set from a handful of nearby grids, not a fair sample of the whole Arctic.

We need a way to build **smaller, labeled datasets** (e.g. "50K windows," "500K windows")
that are still spread fairly across the whole circumpolar region, so a model trained on a
small dataset sees the same *kind* of diversity as one trained on the full dataset — just
less of it.

## The two-pass process

Preprocessing (`01_preprocess.py`) runs in two passes:

**Pass 1 — scan.** Visit every grid once, just to learn about it: how many land pixels does
it have, and how would those pixels split into train/val/test? This pass never keeps the
full raw data around — it only keeps a small summary per grid (a few hundred KB, not GBs).

**Pass 2 — fetch what's actually needed.** Using the summaries from pass 1, decide exactly
which pixels are wanted for train/val/test (see "Representativeness" below), then re-fetch
*only those pixels'* data from cloud storage, normalize it, and save it.

Splitting the work this way means pass 1 can answer "how much do we have, and how should it
be divided?" cheaply, before pass 2 spends time/bandwidth fetching the real data.

## Representativeness: how a "50K" dataset still covers the whole Arctic

A single pixel, fetched at full density, can produce **thousands** of training windows on
its own. If we built a "50K window" dataset naively, it could come from just a dozen pixels
in 2-3 grids — technically 50,000 windows, but geographically meaningless.

Instead, for any **size-capped** split (train when you pass `--train-size`, and val/test
always — see below), we deliberately use a **coarser stride** (`capped_stride` in the
config, currently 24 months instead of 1) when counting how many windows each pixel
contributes. This means each pixel "costs" far fewer windows, so hitting a window budget
requires pulling in **many more pixels — and therefore many more grids**.

On top of that, pixel selection is **round-robin across grids**: one pixel from grid A, one
from grid B, one from grid C, and so on, cycling back around, until the window budget is
met. So a "50K" dataset ends up with a handful of pixels from *most or all* 263 grids,
instead of thousands of pixels from a few.

**Important:** pixel time series are never cut short or resampled out of order — a pixel's
data always stays a continuous monthly sequence. "Coarser stride" only changes how many
overlapping windows we slice out of that same continuous sequence, never the sequence
itself. This matters because the model is a *causal* emulator — it needs real, unbroken
history leading up to the point it's predicting.

Because reaching even a modest window budget at this coarser density already requires more
pixels than there are grids, **every capped run visits every grid** — there's no shortcut
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
automatically (see "Sidecars" below), not something you need to track by hand.

## What gets saved, and where

For every run, `01_preprocess.py` saves into `outputs/arctic_domain/preprocessed/`:

| File | What it is | Regenerated when |
|---|---|---|
| `train_{label}.pkl` | Training set for this run's size (e.g. `train_50K.pkl`, `train_500K.pkl`, `train_2M.pkl`) | Every time you pick a new `--train-size` |
| `train_full.pkl` | The full, uncapped training set (all grids, full density) | Only for an uncapped run — VM-only, tens of GB |
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
slow: a run can take several hours end-to-end (mostly waiting on data fetches, not
crunching numbers). Running that on the GPU VM would tie up an expensive GPU-hour resource
for work that never uses the GPU at all — a waste of money for no benefit.

So preprocessing is done locally instead: any size-capped variant (50K, 500K, 2M) is small
(tens of MB to a few GB), so it's practical to build it on a laptop. Once a variant is
ready, the finished `.pkl`/`.meta.json`/`scaler.pkl` files are simply copied over to the VM,
where `02_train.py` loads them directly — the VM's GPU time is spent only on training, never
on data prep. Only the fully uncapped `train_full.pkl` (tens of GB) needs to be generated on
the VM directly, since it's too large to comfortably build and transfer from a laptop.

Along the way, only small **derived** data is ever cached to disk (per-grid summaries and
per-grid selected-pixel records — megabytes, not the multi-GB raw fetch) so re-running or
resuming after an interruption never risks filling up local storage.

## Resilience

Long local runs can occasionally get killed unexpectedly by the OS. Because pass 1 and pass
2 each cache their per-grid work to disk as they go, a restart resumes almost instantly from
where it left off instead of re-fetching everything from scratch. See `arctic_description.md`
step 1 for the exact cache layout, and `run_preprocess_resilient.sh` for the wrapper script
that automates relaunching until a run finishes successfully.
