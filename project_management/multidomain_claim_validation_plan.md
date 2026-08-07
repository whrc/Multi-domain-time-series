# Multi-Domain Claim — Validation Plan

<!-- Claude Instructions ─────────────────────────────────────────────────────
This is a PLAN document, not a results log. It fixes the paper's central claim,
lists the experiments that establish it, and records the adversarial review the
claim must survive. Results go to key_findings_log.md as usual, not here.
Update this file when the claim or the experiment set changes — not when a run
finishes.
──────────────────────────────────────────────────────────────────────────── -->

**Status:** Plan only — nothing here has been executed.
**Created:** 2026-08-07 · **Branch:** `docs/claim-validation-plan`

---

## Context

The manuscript's §4.4 currently plans to argue "multi-domain beats single-domain," supported by
`MD-seedsweep0714` as corrected by `MD-unitsbugfix0716`. 5-seed median NSE, individual →
multi-domain fine-tuned:

| domain | target | individual | MD fine-tuned |
| --- | --- | --- | --- |
| Amazon | discharge | 0.356 | 0.760 |
| Amazon | active_fire_count | 0.368 | 0.707 |
| Amazon | burned_area | 0.047 | 0.521 |
| Rangeland | GPP / RECO / Rg / Rm | 0.904 / 0.880 / 0.887 / 0.873 | 0.972 / 0.931 / 0.968 / 0.905 |
| Arctic | GPP (hist / proj) | 0.859 / 0.874 | 0.938 / 0.935 |
| Arctic | RECO (hist / proj) | 0.472 | 0.745 / 0.653 |

The direction supports the claim. The evidence, as it stands, would not survive an adversarial
review: three separate confounds are bundled into the word "multi-domain," the statistics are
medians of seed means with no paired test, the effective test n is 19 Amazon stations and 8
Rangeland sites from a single split, and `MD-unitsbugfix0716` already demonstrated that a silent
unit-space mismatch between the two arms is a live failure mode. The result is currently
*consistent with* cross-domain transfer, not *evidence for* it against the obvious alternatives.

This document fixes the claim, lists the minimum experiment set that establishes it, and records
the peer-review attacks each experiment answers.

**Filter applied to the experiment list:** every entry must (a) support the claim, (b) produce
something that appears in the manuscript, even as a single data point, and (c) be non-optional.
Items failing any of the three were cut and are handled by disclosure in text instead — see Part 3.

---

## Part 0 — Literature review (precedes and constrains the claim)

The claim cannot be finalized before knowing what is already published. Part 1 is **provisional**
until this review is done.

### Deliverables of the review

| File | Contents |
| --- | --- |
| `paper/references.bib` | DOI-verified BibTeX; every entry's metadata pulled from a registry, never written from memory |
| `paper/literature_review.md` | Annotated review by theme — what each work found, what it implies for our claim — ending in a positioning statement |

### DOI verification protocol (non-negotiable)

**Purpose: prevent hallucinated references. It gates what gets cited, not what gets searched.**
Discovery is unrestricted — search broadly across ML and environmental-science venues, follow
citation trails, use whatever surfaces the work. The DOI check is the final gate every candidate
passes before entering `references.bib`.

1. **Crossref** — `https://api.crossref.org/works?query.bibliographic=<title + first author>&rows=5`.
   Journals and most Earth-science literature.
2. **DataCite** — `https://api.datacite.org/dois?query=<title>`. arXiv preprints
   (`10.48550/arXiv.*`), Zenodo, datasets. arXiv assigns a DOI to every submission, so ML preprints
   and the arXiv versions of NeurIPS / ICLR / PMLR papers are verifiable here.
3. **Match rule.** Accept only on normalized-title match **and** first-author surname match **and**
   year match. Short of that, search again — do not accept a near-miss.
4. **Version of record.** Crossref search returns preprints, discussion papers, and dataset records
   alongside the published article. Prefer the version of record and check the `container-title`
   before accepting. *(Observed in practice: the seed search below returned the HESS discussion
   DOI instead of the HESS paper for Kratzert 2018, a HydroShare dataset DOI instead of the paper
   for Kratzert 2019b, and an EarthArXiv preprint instead of the WRR article for Nearing 2021 —
   all three silently plausible.)*
5. **Never guess a DOI.** A guessed DOI can resolve to a real but *different* paper, which is worse
   than no citation. *(Observed: a guessed DOI for Nearing 2021 resolved cleanly to Clark et al.
   2021 on performance-metric abuse.)* Always search, then verify.
6. **Entry construction.** Build every entry by content negotiation on the resolved DOI —
   `https://doi.org/{DOI}` with `Accept: application/x-bibtex`, which works for both registries.
   Record which registry verified each entry.
7. **Failures are excluded, not patched.** Anything with no resolvable DOI stays out of
   `references.bib` and goes in an `## Unverified — excluded` appendix of `literature_review.md`
   with the reason. The review may discuss such a work in prose; the manuscript cannot cite it.

### Themes to cover, and why each bears on this study

| Theme | Why it matters here |
| --- | --- |
| **Pretraining on process-model output, fine-tuning on observations** | **Closest prior art to C4 — the novelty question lives here.** Partially surveyed already; see the finding below. |
| **Cross-region transfer to data-sparse areas** | **Closest prior art to C3.** Same finding. |
| Process-model emulation and surrogate modeling | Establishes what Arctic/Rangeland emulation is and what accuracy is normal — i.e. whether our per-domain numbers are competitive |
| Deep learning for streamflow, prediction in ungauged basins | Amazon's exact task and evaluation protocol; source of the drainage-area normalization already adopted (`AZ-5e809245`) |
| Multi-task learning, cross-domain transfer, negative transfer | The mechanism C2 claims and the risk the Introduction raises |
| Transfer scaling and data-efficiency measurement | **C3's exchange-rate framing** — prior work measures transfer in units of "effective data transferred"; E6 should adopt that metric rather than invent one |
| Foundation models for Earth observation, weather, climate | Positions CLAUDE.md Goal 3 and the "one shared encoder" framing |
| Transformers for time series, including the skeptical literature | Justifies the architecture; pre-empts "did you need a transformer" |
| Evaluation metrics and spatial cross-validation | NSE / KGE / PBIAS provenance and spatial-autocorrelation leakage — directly relevant to the whole-grid Arctic split rationale |
| Domain background: TEM, rangeland carbon, Amazon hydrology and fire | Data-source and target-variable citations for §2 |

### Novelty risk — partially resolved already, and it is real

A seed search of the first three themes (22 DOI-verified works, Appendix A) shows **C4 as
originally drafted is not novel**. Two lines of prior work land directly on it:

- **Process-model pretraining → observational fine-tuning already exists.** Read et al. (2019,
  `10.1029/2019wr024922`) pretrain a deep model on process-model output and fine-tune on sparse
  observations for lake temperature; Jia et al. (2021, `10.1145/3447814`) extend it; Willard et al.
  (2021, `10.1029/2021wr029579`) transfer to *unmonitored* lakes, which is our spatial-holdout
  protocol.
- **Cross-region transfer to data-sparse regions already exists.** Ma et al. (2021,
  `10.1029/2020wr028600`) transfer hydrologic data across continents specifically to improve
  prediction in data-sparse regions — C3's structure, inside one domain.

**The review must therefore sharpen the claim rather than assert it.** What remains genuinely new
is the *distance* the transfer crosses. The prior work transfers between instances of the **same
variable, same process, same task family** (lake → lake, basin → basin). This study transfers
across **different biomes, different process models, and different target variables**: Arctic TEM
carbon fluxes → Amazon river discharge and fire activity. That is a defensible and interesting
distinction — but the paper must make it deliberately, in the Introduction, rather than be caught
on it in review.

The review ends with a **positioning statement**: one paragraph on what is new here relative to the
verified prior art. Part 1's claim ladder must then match it.

---

## Part 1 — The claim

*Provisional pending Part 0. The reframe below is the study's own logic; its novelty depends on the
completed review — see above.*

### The reframe that raises the ceiling

The three domains are not interchangeable. **Arctic (emulating TEM)** and **Rangeland (emulating
RangeSTAR)** have *process-model output* as targets: deterministic, noise-free, effectively
unlimited. **Amazon** has *real observations* — ANA gauge discharge and satellite fire counts and
burned area — noisy, and limited to 98 stations.

So the headline is not "multi-task learning helps." It is:

> **Pretraining on abundant process-model emulation data measurably improves real-world
> observational prediction in a data-scarce domain — across biomes, process models, and target
> variables that share nothing.**

This tells any group with a process model and a data-poor region what to do: process-model output
is a *manufacturable pretraining corpus*. The current numbers are consistent with it — Amazon, the
observational domain, is where the gain is largest.

### Claim ladder

| | Claim | Impact | Established by |
| --- | --- | --- | --- |
| **C1** | One shared encoder serves three biomes with no loss to any (no negative transfer) | Low-moderate | E1 — **declared fallback if C2 fails** |
| **C2** | Cross-domain pretraining beats *fairly matched* dedicated training | Moderate | E1 + E2 + E3 + E4 |
| **C3** | Cross-domain pretraining **substitutes for in-domain data**, at a quantifiable exchange rate | High | C2 + E6 |
| **C4** | **Process-model emulation data is an effective pretraining corpus for real-world observational prediction, across unrelated biomes and variables** | **Highest** | C2 + E5 |

**Lead with C4, support with C3, keep C1 as the declared fallback.** C3 and C4 are the same
experiments viewed from two angles, so nothing extra is spent. This also sets up CLAUDE.md's Goal 3
(foundation-model fine-tuning) as the natural follow-on paper.

---

## Part 2 — Experiments

### G0 — Cross-arm equivalence gate *(prerequisite, not an experiment)*

**Motivation.** `MD-unitsbugfix0716` — multi-domain Amazon metrics computed in log1p /
area-normalized space while the individual arm was in physical units, moving active_fire_count
0.886 → 0.707 — was found by eye while building an unrelated figure, *after* the numbers had been
human-verified and written into the status doc. Every number below is meaningless if the two arms
are not measuring the same quantity. Answers R6.

**Design.** Assertions, not inspection. Four checks:

1. **Observation identity.** For every held-out unit × target, the observations produced by the
   individual pipeline's `04_evaluate.py` and by `domains/multi_domain/04_evaluate.py` must be
   numerically identical — both claim physical units, so any difference is a transform bug.
2. **Test-unit identity.** Both arms hold out exactly the same pixels / stations / sites, and no
   multi-domain training window originates from any individual pipeline's test unit.
3. **Scaler provenance.** Every scaler the multi-domain pipeline consumes was fit on training units
   only; per-unit climatology features are computed identically in both arms.
4. **Metric-code identity.** Both arms route through `shared/metrics.py` and
   `shared/evaluate.py::per_unit_metrics` with the same masking and the same Arctic January-only
   handling for ALD/VEGC.

**Implementation.** New tests under `tests/multi_domain/`, following `tests/arctic_domain/
test_grid_split.py`. Check 1 is the load-bearing one and would have caught the log1p bug the day it
was introduced.

**Manuscript destination.** Reproducibility statement; Supplement S4.
**Criteria.** All four pass, or the offending pipeline is fixed and affected numbers regenerated.
**Cost.** Zero GPU.

---

### E1 — Paired comparison harness and statistics

**Motivation.** The comparison is currently reported as medians of seed means, with no paired test,
no confidence interval, and no per-unit win/loss count, on 19 Amazon stations and 8 Rangeland
sites. `compare_models.py` has been listed as unimplemented since `MD-devsmoke0711`. Answers R3, R4.
Establishes C1.

**Design.** Build `compare_models.py` (Individual vs Unified-joint vs Unified-fine-tuned), the
script `multi_description.md` §Step 4 defers to. It joins per-unit metrics across arms on the shared
id columns — **asserting identical unit sets before comparing** — and produces, per target:

- Paired per-unit ΔNSE with **Wilcoxon signed-rank**, bootstrap CIs on the median delta, and
  **Holm-Bonferroni** correction across the target family. The unit-level test is the powerful one;
  the seed-level paired test (n = 5) is reported as an explicitly underpowered secondary check.
- **Fraction of held-out units improved.** 17-of-19 and 3-of-19 are different scientific statements
  with the same median.
- Agreement across all four metrics — the claim must hold on NSE, KGE, RMSE *and* PBIAS;
  disagreements get named, not dropped.
- Effect size against the across-seed std already in the `*_seedavg.csv` files.
- Full negative-transfer audit: every domain × target × period where multi-domain loses.
- Per-stratum breakdown: Arctic by SSP × period, Rangeland by PFT.
- Paired verdict on **pretrained vs fine-tuned**, which `MD-seedsweep0714` reports as nearly
  identical (Arctic GPP 0.925/0.926 vs 0.935/0.938) and which the findings log still lists open.

**Implementation.** No retraining — every input already exists on disk:

```
outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed{1..5}/metrics_test.csv
outputs/amazon_domain/evaluation_seed{1..5}/metrics_test.csv
outputs/rangeland_domain/evaluation_fluxonly_seed{1..5}/metrics_test.csv
outputs/multi_domain/evaluation/{pretrained,finetuned}_fluxonly_seed{1..5}/{domain}/{domain}_metrics.csv
```

Reuse `run_seed_sweep.py`'s `ARCTIC_ID_COLS` / `AMAZON_ID_COLS` / `RANGELAND_ID_COLS` for the join
keys, and `shared/seed_aggregation.py` for the across-seed rollup. `scipy.stats` for the tests.

**Manuscript destination.** Table 4, Table 5, Figure 6, §4.3, §4.4 transfer classification.
**Criteria.** Gains that are significant after correction *and* backed by a clear majority of units
support C1/C2; gains carried by a handful of units get reported as such.
**Cost.** Zero GPU, hours of work.

---

### E2 — Fair individual baseline

**Motivation.** The single most dangerous objection (R1). The shared backbone is 256-dim / 6-layer /
ff-1024 / head 128×2 ([`config/multi_domain.yaml:20-29`](../config/multi_domain.yaml)), deliberately
sized "at least as large as the standalone Arctic model" (manuscript §3.5). Individual Amazon is
128/3/512 and individual Rangeland is 64/3/256 with dropout 0.3 — so the two domains carrying the
headline are compared against a backbone 4-16× wider than their own. Meanwhile an epoch is defined
by Arctic's batch count with the small domains `itertools.cycle`d
([`domains/multi_domain/02_train.py:162-163`](../domains/multi_domain/02_train.py)) at
`batch_size: 1024` against Amazon's own 256, so the small domains receive far more effective passes.
And every individual config states "no grid search" in its own comments while the shared model was
chosen large. Load-bearing for C2.

**Design.** Retrain individual Amazon and Rangeland removing all three asymmetries:

1. **Capacity-matched** — individual model at the shared backbone's dimensions.
2. **Budget-matched** — same gradient-step count and batch size the multi-domain pretrain
   effectively grants that domain. **Extract the real step counts from the `MD-seedsweep0714` logs
   under `outputs/_seed_sweep_logs/`** rather than estimating them.
3. **Tuning-parity** — a small documented hyperparameter search, so the baseline receives
   comparable tuning effort.

5 seeds each, via `run_seed_sweep.py`.

**Implementation.** New `production_matched` hyperparameter profiles in
`config/amazon_domain.yaml` and `config/rangeland_domain.yaml` (both already switch profiles on
`mode:`), selected by a `--profile` flag on each domain's `02_train.py` / `04_evaluate.py`.
Outputs to `evaluation_matched_seed{s}/` so they never collide with the published run.

**Manuscript destination.** New "Individual (matched)" column in Table 4; new Methods subsection;
regenerated Figure 6 and Figure 7.
**Criteria.** If the matched baseline closes most of the gap, the reported effect was capacity and
budget — restate as C1 (see stopping rules).
**Cost.** Small datasets; hours of A100 time. Note `burned_area` individual NSE 0.047 is at the
mean-prediction baseline — until this runs, "0.047 → 0.521" reads as "our baseline failed to train."

---

### E3 — Reference baselines

**Motivation.** **Verified: no random forest, gradient boosting, or linear baseline exists anywhere
in this repo.** In hydrology and fire modeling a per-station RF on the same features is the default
reference, and reviewers ask for it reflexively. If RF reaches NSE ~0.7 on Amazon discharge, the
transformer premise deflates. Answers R2.

**Design.** Two floors, on identical features, splits, and metric code:

1. **Per-unit monthly climatology** — mean of each held-out unit's own record by calendar month.
   (The per-unit mean is already implicit in NSE = 0, so it needs no separate run; seasonal
   climatology is the informative floor and is not.)
2. **Random forest / gradient boosting** per domain, flattening the same seq_len-12 window into a
   feature vector.

**Implementation.** A single `05_baselines.py` per domain reading the same `train.pkl` / `test.pkl`
and scoring through `shared/evaluate.py::per_unit_metrics`, so the numbers are directly comparable.
`scikit-learn` (not currently in `requirements.txt`).

**Manuscript destination.** Floor rows in Table 4; §4.2.
**Criteria.** Both deep arms must clearly beat both floors on every headline target; any target
where they do not is reported as such.
**Cost.** CPU, minutes. **Run early — it is the most likely source of an unwelcome surprise.**

---

### E4 — Solo-in-harness ablation

**Motivation.** The one experiment that isolates *cross-domain data* from capacity, schedule, and
the two-stage recipe simultaneously — the three things R1 says are confounded. **The single
experiment most likely to change the paper's conclusion.** Load-bearing for C2, C3, C4.

**Design.** Run the multi-domain pipeline with a single domain in the mix: identical 256/6/1024
backbone, identical `steps_per_epoch`, identical pretrain → freeze → head-finetune recipe,
identical batch size and LR-finder procedure. The only variable is whether the other domains' data
is present. 5 seeds × 3 domains. As a side benefit this holds the model-selection protocol constant
across arms, which also disposes of R8.

**Implementation.** A `--domains` flag on `domains/multi_domain/02_train.py` restricting the
`DOMAINS` list from `domains/multi_domain/flux_only.py`; `steps_per_epoch` must stay pinned to
Arctic's batch count via `training.steps_per_epoch` in `config/multi_domain.yaml` (currently `null`
= derived from Arctic) so the solo runs get the same schedule. Output paths need the domain-set in
the folder name. Small and non-architectural, but touches `02_train.py`, `03_predict.py`,
`04_evaluate.py`, and the path helpers in `flux_only.py`.

**Manuscript destination.** Table 4 column; Figure 6 panel.
**Criteria.** Cross-domain pretraining must beat solo-in-harness by a margin comparable to the
reported individual-vs-multi gap. If solo-in-harness already recovers most of the gain, C2/C3/C4
are unsupported — see stopping rules.
**Cost.** 15 pretrain + finetune runs. The largest single GPU line item; sequence after E1/E3.

---

### E5 — Donor ablation (leave-one-domain-out)

**Motivation.** **This is the experiment that establishes C4.** It tests whether Amazon's gain
tracks the large synthetic **Arctic** corpus specifically, or is generic extra data. Also the only
thing standing between the study and R10 ("two of your three domains are deterministic model
output") — that objection becomes the contribution if, and only if, the synthetic donor is shown to
drive the observational domain's gain.

**Design.** For Amazon and Rangeland as target domain, pretrain on four corpora — {self},
{self + Arctic}, {self + other small domain}, {all three} — then finetune and evaluate as usual.
5 seeds. The {self} arm is E4's solo run, so it is not paid for twice.

**Implementation.** Falls out of E4's `--domains` flag at no extra implementation cost; only extra
runs.

**Manuscript destination.** Figure 9 (new); Abstract and §4.4 headline.
**Criteria.** C4 holds if Amazon's gain rises with Arctic's inclusion specifically. If Rangeland
alone reproduces it, C4 is unsupported — retreat to C3.
**Cost.** ~6 additional pretrain configurations × 5 seeds.

---

### E6 — Data-scarcity dose-response

**Motivation.** **This is the experiment that establishes C3**, and the strongest single item for
the Discussion. It converts a vague "better" into an exchange rate — how much in-domain data
cross-domain pretraining is worth — which is the form the transfer-scaling literature already uses
and the form other groups can apply to their own regions.

**Design.** Subsample Amazon and Rangeland **training units** (not windows — units, to preserve the
spatial-generalization protocol) to ~25 / 50 / 100%, and at each level train both the matched
individual model (E2) and the multi-domain pretrain+finetune. Held-out test set stays fixed
throughout, so the curves are comparable. 5 seeds per point.

**Implementation.** A `--train-frac-units` flag on Amazon's and Rangeland's `01_preprocess.py`,
subsampling the train unit list deterministically under the existing `preprocessing.random_seed`.
`domains/arctic_domain/05_learning_curve.py` is the closest existing precedent for the sweep
structure and reporting.

**Manuscript destination.** Figure 10 (new); §4.4 and Discussion.
**Criteria.** C3 holds if the transfer gain grows monotonically as in-domain data shrinks. A flat
relationship means the benefit is not about data scarcity, and the Discussion's central mechanism
claim must be dropped.
**Cost.** 2 domains × 3 levels × 2 arms × 5 seeds, small datasets throughout.

---

### E7 — Full-target parity

**Motivation.** The claim is currently established only for the flux-only variant, and flux-only
was itself adopted *because* the pool targets were failing. Undisclosed post-hoc target selection
is a reviewer kill shot. Answers R5.

**Design and implementation.** In order:

1. **Fix the units bug on the full-target path** — the outstanding follow-up recorded in
   `MD-unitsbugfix0716`. `inverse_amazon_log1p()` in `domains/multi_domain/flux_only.py` is already
   written and applied on the flux-only path; the full-target path (`pretrained/`, `finetuned/`, no
   `_fluxonly` suffix) still needs it. Re-evaluation only, no retraining.
2. **Run the 5-seed sweep** for the full-target variant via `run_seed_sweep.py`, then aggregate.

**Manuscript destination.** Supplement S1; §4.4 scope statement.
**Criteria.** The direction of the effect must hold for the full-target variant. If it does not,
the manuscript scopes the claim to flux targets **and states why flux-only was chosen**, which it
must do regardless.
**Cost.** Step 1 is ~1 min/seed inference. Step 2 is a full 5-seed pretrain + finetune sweep.

---

### E8 — Split robustness (Amazon + Rangeland only)

**Motivation.** Amazon test = 19 stations, Rangeland test = 8 sites, both from a single draw at
`preprocessing.random_seed: 42`. A median NSE difference over 8 sites is not, on its face,
distinguishable from split noise. Answers R3 alongside E1.

**Design.** Re-preprocess Amazon and Rangeland at 2-3 alternative `random_seed` values, keeping
Arctic's deliberately frozen split fixed, and rerun the individual and multi-domain arms per split.

**Implementation.** Amazon/Rangeland `01_preprocess.py` already take the split seed from config.
**Two constraints to respect:** (a) the multi-domain pretrain must rerun per split, since its
Amazon/Rangeland training data changes — this is the real cost, not the preprocessing; (b) Arctic's
val/test are frozen behind the fail-loud sidecar guard added in commit `3d19d6e`, and this
experiment must not trip it — Arctic is deliberately untouched.

**Manuscript destination.** Supplement S2; one sentence in §4.4.
**Criteria.** Sign and rough magnitude of the gain stable across splits. If the gain vanishes or
reverses on another split, the result is a split artifact and the paper cannot make C2.
**Cost.** Minutes of CPU per split for preprocessing; the pretrain reruns dominate.

---

### Cut from this program, and why

| Cut | Reason |
| --- | --- |
| Shuffled-donor negative control | Subsumed by E4 + E5 — if donor *identity* matters, generic regularization is already excluded |
| Model-selection-parity rerun | E4 holds the protocol constant across arms, so it cannot explain E4's result; handled by disclosure (R8) |
| Causal-climatology sensitivity | Does not support the claim and affects both arms equally; handled by disclosure (R7) |

---

## Part 2b — Figures and tables these experiments produce

The manuscript has Figures 1-8 and Table 4 today. This program revises three exhibits and adds
three, plus four supplements.

### Revised

- **Table 4 — master comparison.** Currently three columns (Individual / Pretrained / Fine-tuned).
  Becomes the **ladder**: Climatology · RF-GBM · Individual (as-published) · Individual (matched) ·
  Solo-in-harness · Pretrained · Fine-tuned. Rows = domain × target; cells = median across held-out
  units ± across-seed std. NSE in main text, RMSE/KGE/PBIAS as supplementary twins. *Feeds: E1, E2,
  E3, E4.* The column order is the table's argument — reading left to right is watching each
  confound get stripped away.
- **Figure 6 — the ladder figure (central figure of the paper).** One row per domain, targets
  grouped along x, boxplots across held-out units, one box per Table 4 ladder column. Replaces the
  current three-box version in `figures/scripts/make_remaining_figures.py`, which compares against
  the un-matched baseline. *Feeds: E1, E2, E3, E4.*
- **Figure 7 — per-site %-change maps.** `figures/scripts/make_figure7.py`, currently drawn against
  the as-published individual baseline; must be regenerated against **Individual (matched)** once
  E2 lands, or it maps the confound rather than the effect. *Feeds: E2.*

### New

- **Table 5 — paired significance (statistical spine of §4.4).** One row per domain × target:
  median ΔNSE (fine-tuned − matched individual), bootstrap 95% CI, Holm-adjusted Wilcoxon p,
  fraction of held-out units improved, and a transfer verdict (positive / negative / none). Turns
  §4.4's classification bullet into an actual result. *Feeds: E1, E2.*
- **Figure 9 — donor ablation (the C4 figure).** Amazon and Rangeland as target domains; x =
  pretraining corpus {self, +Arctic, +other small, all three}; y = held-out test NSE per target;
  points = seed mean with across-seed error bars. *Feeds: E5.*
- **Figure 10 — data-substitution curve (the C3 figure).** x = fraction of in-domain training
  units; y = held-out test NSE; two lines per target (Individual matched, multi-domain fine-tuned)
  with across-seed bands. The **horizontal** gap between curves is the headline quantity. *Feeds: E6.*

### Supplementary

| | Contents | Feeds |
| --- | --- | --- |
| **S1** | Full-target parity — Table 4's structure for the full-target variant | E7 |
| **S2** | Split robustness — median ΔNSE per target, one point per split seed | E8 |
| **S3** | Negative-transfer audit — every domain × target × period where multi-domain loses, in full | E1 |
| **S4** | Equivalence audit — G0's assertions and their outcome | G0 |

**Dependency worth flagging:** Table 4, Figure 6, and Figure 7 cannot be finalized until E2 lands,
because all three currently compare against the un-matched individual baseline. Everything in E1
can be computed today and revised once E2 exists.

---

## Part 3 — Adversarial review

Written as a reviewer would. Each objection maps to the experiment that answers it, or is
explicitly resolved by disclosure rather than by a run.

| | Objection | Covered by | Residual risk |
| --- | --- | --- | --- |
| **R1** | **"The baseline is a straw man."** Shared backbone 256/6/1024 vs Amazon's 128/3/512 and Rangeland's 64/3/256; an epoch is Arctic's batch count with the small domains cycled at batch 1024 vs 256; every individual config says "no grid search" while the shared model was chosen large. Sharpest form: Amazon `burned_area` individual NSE 0.047 *is* the mean-prediction baseline. | **E2, E4** | None if both run. **Weakest point in the paper — it fails without them.** |
| **R2** | "No non-deep baseline anywhere." Verified true across the repo. | **E3** | Result unknown — run early. |
| **R3** | "n = 19 stations and 8 sites, one split, medians only." No paired test, no CI, no win/loss count; Rangeland PFT strata hold 1-3 sites each. | **E1, E8** | Low after both. |
| **R4** | "Fine-tuning does nothing, which undercuts your framing." Pretrained ≈ fine-tuned pushes all explanatory weight back onto R1's confounds. | **E1** (paired verdict) | Answered either way; state it plainly. |
| **R5** | "You changed the target set until it worked." Flux-only was adopted because pool targets were failing. | **E7** | Needs the scope statement in §4.4 regardless. |
| **R6** | "Your headline number already moved once because of a silent bug." (`MD-unitsbugfix0716`: 0.886 → 0.707.) | **G0** | Low once the assertions exist. |
| **R7** | "Your 'causal' model uses features computed from the full record." Per-unit climatology spans each unit's entire series (Open Question 5). Drivers-only, standard for static catchment attributes, and equal across arms — so the comparison is safe, but the strict-causality wording in §3.1/§3.6 is not. | **Disclosure** — state the convention and its justification in Methods | Low; a wording fix, not an experiment. |
| **R8** | "Model selection differs between arms." Individual models early-stop on their own val loss; multi-domain pretrain early-stops on the cross-domain **mean** ([`02_train.py:196-217`](../domains/multi_domain/02_train.py)), dominated by Arctic. | **Disclosure** — E4 holds the protocol constant, so it cannot explain E4's result | Low. |
| **R9** | "'A single model' overstates it." Three checkpoints after Stage 2; each domain keeps its own projection and head. Only the encoder is shared. | **Disclosure** — state precisely in §3.5 | None. |
| **R10** | "Two of your three domains are deterministic process-model output, so transfer is easier than in the real world." | **Reframed as C4, tested by E5** | Becomes a genuine weakness only if E5 shows the synthetic donor is not what drives Amazon's gain. |
| **R11** | "This has been done." Read et al. 2019, Willard et al. 2021, Ma et al. 2021 — see Part 0. | **Positioning statement** from the completed review; the defense is transfer *distance* (cross-biome, cross-process-model, cross-variable), not the mechanism | Moderate — the Introduction has to make this argument explicitly. |

---

## Part 4 — Order and stopping rules

**Order** — cost and information gain differ by an order of magnitude across tiers:

1. **Part 0** — the literature review, because it can still change the claim.
2. **G0, then E1** — zero GPU, and G0 gates everything: if the arms do not measure the same
   quantity, every downstream number is suspect.
3. **E3** — cheap, and could reframe the paper. Run before spending A100 hours.
4. **E4** — most likely to change the conclusion.
5. **E5, E6** — the C4 and C3 evidence, i.e. the actual contribution.
6. **E2, E7, E8** — parity, scope, and robustness.

**Stopping rules, fixed before anything runs:**

- **E4** shows solo-in-harness recovering most of the gain → drop C2/C3/C4, publish **C1**, and
  report the capacity/schedule finding as the real result.
- **E2**'s matched baseline closes most of the gap → same outcome; the gain was capacity and budget.
- **E3** shows RF matching the transformer on Amazon → deep learning is demoted to a component of
  the transfer story, not the contribution.
- **E5** shows Rangeland alone reproduces Amazon's gain → **C4 unsupported**, retreat to C3.
- **E6** shows a flat gain-vs-scarcity relationship → **C3 unsupported**, drop the Discussion's
  central mechanism.
- **E8** shows the gain vanishing or reversing on another split → the result is a split artifact and
  C2 cannot be made.

**Paper mapping.** C4 → Abstract and §4.4 headline; C3 → §4.4 and Discussion; C1 → fallback and
§4.3; E1 → Tables 4-5 and supplement; E2/E4/E5/E6 → new Methods subsection on controls; G0/E7/E8 →
supplement and Data & Code Availability.

---

## Appendix A — DOI-verified seed references

Verified 2026-08-07 against Crossref via `api.crossref.org`, under the Part 0 protocol. These are a
head start on `paper/references.bib`, not the finished review — themes D-I of the table above are
not yet searched. Every DOI below returned a record whose title, first author, and year match.

### Process-model pretraining → observational fine-tuning (closest prior art to C4)

| Ref | DOI | Venue |
| --- | --- | --- |
| Read et al. 2019, *Process-Guided Deep Learning Predictions of Lake Water Temperature* | `10.1029/2019wr024922` | Water Resources Research |
| Jia et al. 2021, *Physics-Guided Machine Learning for Scientific Discovery: An Application in Simulating Lake Temperature Profiles* | `10.1145/3447814` | ACM/IMS Trans. Data Science |
| Willard et al. 2021, *Predicting Water Temperature Dynamics of Unmonitored Lakes With Meta-Transfer Learning* | `10.1029/2021wr029579` | Water Resources Research |
| Willard et al. 2022, *Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems* | `10.1145/3514228` | ACM Computing Surveys |
| Karpatne et al. 2017, *Theory-Guided Data Science* | `10.1109/tkde.2017.2720168` | IEEE TKDE |
| Tsai et al. 2021, *From calibration to parameter learning* | `10.1038/s41467-021-26107-z` | Nature Communications |

### Emulation and surrogate modeling

| Ref | DOI | Venue |
| --- | --- | --- |
| Reichstein et al. 2019, *Deep learning and process understanding for data-driven Earth system science* | `10.1038/s41586-019-0912-1` | Nature |
| Irrgang et al. 2021, *Towards neural Earth system modelling* | `10.1038/s42256-021-00374-3` | Nature Machine Intelligence |
| Rasp et al. 2018, *Deep learning to represent subgrid processes in climate models* | `10.1073/pnas.1810286115` | PNAS |
| Lu & Ricciuto 2019, *Efficient surrogate modeling methods for large-scale Earth system models* | `10.5194/gmd-12-1791-2019` | Geoscientific Model Development |
| Dagon et al. 2020, *A machine learning approach to emulation and biophysical parameter estimation with CLM* | `10.5194/ascmo-6-223-2020` | ASCMO |
| Beucler et al. 2021, *Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems* | `10.1103/physrevlett.126.098302` | Physical Review Letters |
| Watson-Parris 2021, *Machine learning for weather and climate are worlds apart* | `10.1098/rsta.2020.0098` | Phil. Trans. R. Soc. A |

### Deep learning hydrology, ungauged basins, cross-region transfer (closest prior art to C3)

| Ref | DOI | Venue |
| --- | --- | --- |
| Kratzert et al. 2018, *Rainfall–runoff modelling using LSTM networks* | `10.5194/hess-22-6005-2018` | HESS |
| Kratzert et al. 2019, *Toward Improved Predictions in Ungauged Basins* | `10.1029/2019wr026065` | Water Resources Research |
| Kratzert et al. 2019, *Towards learning universal, regional, and local hydrological behaviors* | `10.5194/hess-23-5089-2019` | HESS |
| Nearing et al. 2021, *What Role Does Hydrological Science Play in the Age of Machine Learning?* | `10.1029/2020wr028091` | Water Resources Research |
| Nearing et al. 2024, *Global prediction of extreme floods in ungauged watersheds* | `10.1038/s41586-024-07145-1` | Nature |
| Ma et al. 2021, *Transferring Hydrologic Data Across Continents* | `10.1029/2020wr028600` | Water Resources Research |
| Feng et al. 2020, *Enhancing Streamflow Forecast ... With Data Integration at Continental Scales* | `10.1029/2019wr026793` | Water Resources Research |
| Addor et al. 2017, *The CAMELS data set* | `10.5194/hess-21-5293-2017` | HESS |

### Metrics

| Ref | DOI | Venue |
| --- | --- | --- |
| Clark et al. 2021, *The Abuse of Popular Performance Metrics in Hydrologic Modeling* | `10.1029/2020WR029001` | Water Resources Research |

**Still to search:** multi-task learning and negative transfer; transfer scaling and data-efficiency
measurement; foundation models for Earth observation and weather; transformers for time series and
the skeptical literature; remaining evaluation-metric and spatial-cross-validation provenance
(Nash & Sutcliffe, Gupta/KGE, Moriasi/PBIAS, spatial-CV); domain background for TEM, RangeSTAR,
and Amazon hydrology and fire.
