# Methodology Audit — Multi-Domain Time Series Prediction

**Date:** 2026-06-17
**Branch:** `review/methodology-audit-20260617`
**Reviewer scope:** `arctic_domain`, `amazon_domain`, `rangeland_domain` (skipped `multi_domain`).
Reviewed: the three `*_description.md` specs, three `config/*_domain.yaml`, the four
shared modules (`shared/transformer.py`, `shared/metrics.py`, `shared/plots.py`,
`config/config.py`), and the governance docs (`CLAUDE.md`, `project_management/`).
`00_eda.ipynb` and the empty `01–04`/`run_*.py` scripts were not reviewed as code (their
docstring/pointer convention was checked).

This is an adversarial design review **before** the pipeline scripts are written. At the time this report was drafted, no codebase changes had been made beyond adding this document; remediation was applied in subsequent commits on the same branch.

**Severity legend:** Critical = silently-wrong science or a blocking contradiction ·
High = wrong/biased results or implementation blocker · Medium = real defect, localized ·
Low = minor/style.

> **Process note:** `proj_mgmt.md` is human-owned; per its own rule the only allowed edit is
> adding a Navigation row when a new management file is created. I did **not** edit it — if
> you want this report indexed, please add the nav row yourself.

---

## 1. Executive Summary — top findings ranked by risk

| # | Severity | Finding | Where |
|---|---|---|---|
| 1 | **Critical** | **Input→target temporal alignment contradicts the project's own framing.** All three specs slice `input` and `target` at the *same* indices (`segment[start:start+seq_len]` for both), and `shared/transformer.py` emits one output per input position. So the model learns **contemporaneous** mapping Xₜ→Ŷₜ (with causal history), i.e. emulation/nowcasting — **not** "predicts the next time step's output (next-token prediction)" as CLAUDE.md states. An implementer following CLAUDE.md literally would shift targets by one and build a different model. | CLAUDE.md "Project" §; `arctic_description.md:144,155`; `amazon_description.md:128`; `rangeland_description.md:169`; `shared/transformer.py:73-79` |
| 2 | **High** | **Amazon is labeled "forecasting" but designed as a nowcast.** Same-step Xₜ→Yₜ means predicting discharge/fire at month *t* requires climate at month *t*. That is gap-filling/emulation, not forecasting the future. Reporting it as forecast skill would overstate the science. | `amazon_description.md:1,5,128` |
| 3 | **High** | **No domain holds out a future time period.** All three split by spatial unit (pixel/station/site), sending each unit's *entire* time series to one split. This measures spatial generalization, not temporal-forecast skill. Arctic further reports a "projected period" metric that, because the split is spatial, was seen (for train pixels) during training and risks being read as future-projection skill. | `arctic_description.md:125,189`; `amazon_description.md:106`; `rangeland_description.md:129` |
| 4 | **High** | **Amazon & Rangeland substitute a global mean for val/test units' climatology features.** Per-unit climatological means are computed from *training units only*; val/test units get the training **global** mean. These features come from predictors (available for all units), so this is not leakage — it just creates a train/test feature-distribution mismatch that biases evaluation downward and wastes the feature. | `amazon_description.md:107`; `rangeland_description.md:131` |
| 5 | **High** | **SSOT contradicts reality on all three domains.** `current_project_status.md` omits `rangeland_domain` entirely, marks `amazon_domain` "Not Started / config + description not yet written" (both exist), and marks `arctic_domain` "Complete / pipeline fully implemented" — yet arctic `02–04` are empty and `metrics.py`/`plots.py` are unimplemented stubs. By the SSOT rules this table is authoritative, so it actively misleads. | `current_project_status.md:27-29,54-58` |
| 6 | **High** | **The "SSOT for all logging decisions" is Arctic-only.** `log_experiments.md` declares "if a metric or artifact is not listed here, it is not logged," then hardcodes Arctic (SSP IDs, NetCDF prediction format, per-pixel `metrics.csv` with `grid/y/x/lat/lon`, spatial NSE maps). Amazon (parquet, per-station) and Rangeland (parquet, per-site) have no logging spec at all. | `log_experiments.md:9-11,57-130` |
| 7 | **Medium** | **`metrics.py` has no divide-by-zero/degenerate-series handling** (all four functions are `NotImplementedError` stubs). NSE blows up on constant `obs`; KGE's α/β and Pearson `r` are undefined when `std(obs)=0` or `mean(obs)=0`; PBIAS divides by `Σobs`. Amazon fire targets can be all-zero over a test station (`mean=Σ=0`) and Rangeland slow pools are near-constant at a single test site — both will trigger this. | `shared/metrics.py:22-44` |
| 8 | **Medium** | **Rangeland window (6 months) contradicts its own pool rationale.** The spec says pools have near-zero predictor correlation and must be learned "from sequences" of long-term accumulation, but `seq_len=6` in *both* dev and production. A 6-month causal window cannot carry multi-year accumulation state. | `rangeland_description.md:113,117`; `rangeland_domain.yaml:36-40` |

---

## 2. Per-domain findings

### 2.1 Arctic

| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| Contemporaneous Xₜ→Yₜ alignment (see Exec #1); emulation framing is scientifically correct for Arctic but conflicts with CLAUDE.md "next-token". | `arctic_description.md:144,155` | High | TEM output at *t* is a function of drivers at *t*; same-step mapping is right for an emulator, but the doc label is wrong. |
| Split is by pixel; entire per-pixel series → one split; no temporal holdout. | `arctic_description.md:125` | High | Measures spatial generalization, not forecasting. "Projected period" metric (`:189`) was trained-on for train pixels. Defensible as emulation, but must be labeled as spatial CV. |
| Normalization stats fit train-pixels only via `nanmean`/`nanstd`, `std=1` where 0; applied to all splits. | `arctic_description.md:127,129` | — (correct) | Explicit and leakage-free. Good. |
| Dev `stride=360`, `seq_len=10`. 360 is a multiple of 12, so every dev window starts in January and spans Jan–Oct → exactly one January per window. ALD/VEGC (yearly, January-only targets) are supervised only at window position 0, at a fixed phase; ~7 windows/pixel (SSP1), ~3 (SSP5), one grid. | `arctic_domain.yaml:62-63`; `arctic_description.md:111,121` | Low | Dev-only artifact (production `stride=1` fixes it); flagged so it isn't mistaken for a representative run. |
| LR config uses a single `learning_rate` that the spec tells you to **mutate in config** after the LR finder. | `arctic_description.md:151`; `arctic_domain.yaml:92` | Medium | Diverges from Amazon/Rangeland (`initial_lr`+`optimized_lr`); hand-editing the SSOT config each run is fragile. Unify (see §3). |
| `metrics.csv` columns in the spec (`grid,ssp,y,x,target,period,rmse,nse,kge,pbias`, lowercase) disagree with `log_experiments.md` (`grid,y,x,lat,lon,ssp,period,variable,RMSE,…`, uppercase, adds lat/lon, `variable` not `target`). | `arctic_description.md:191`; `log_experiments.md:82-97` | Medium | Two authoritative-sounding schemas for the same file; downstream/report code can't satisfy both. |
| Loss `((pred-target)[valid]**2).mean()` is NaN if a window has zero valid targets. | `arctic_description.md:155` | Low | GPP/RECO are monthly so `valid` is rarely empty, but a guard (skip/`n>0`) should be specified. |

**Implementability checklist — Arctic**

| Stage | Status | Notes |
|---|---|---|
| Preprocess | Ready (minor) | `seq_len/stride/random_seed/train_frac…` all in yaml and resolved by `config.py`. Unspecified but internally consistent: the *ordering* of the ~`nStatic` 2D vars across the 5 merged static files (affects feature/scaler column order — fine if deterministic). GCS grid auto-discovery mechanism implied, not detailed. |
| Train | Ready (deps) | Dataset slicing, masked-MSE loss, checkpoint-on-val all specified. Depends on `metrics.py`/`plots.py` (stubs) and the LR-finder package (not in requirements). `seq_len/stride` explicitly placeholders. |
| Predict | Ready | Last-position inference, NaN fill, inverse-transform `[-4:]`, NetCDF reconstruction specified. |
| Evaluate | Ready (schema) | Loads truth from GCS; resolve the `metrics.csv` schema conflict above first. |

---

### 2.2 Amazon

| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| "Forecasting" label vs contemporaneous Xₜ→Yₜ design (Exec #2). | `amazon_description.md:1,5,128` | High | Cannot forecast future discharge/fire without future climate inputs; the design is a nowcast/emulator. |
| Val/test stations receive the **global** training climatology instead of their own (Exec #4). | `amazon_description.md:107` | High | Per-station mean/std of `[precip,tmax,tmin]` are computed from predictors available for all stations; substituting a global constant for test stations creates train/test feature mismatch and biases metrics. |
| **Eval ground-truth source is contradictory.** Step 4 says load "ground truth … from parquet," but the prediction parquet (Step 3) holds only `*_pred` columns — no truth. | `amazon_description.md:155,163` | Medium | Blocks evaluation as written; truth must come from `test.pkl` (inverse-transformed) or a re-loaded CSV. Unspecified. |
| **Raw→clean column rename map lives only in the description table, not the yaml.** `columns.id`/`dynamic`/`targets` use the renamed names; the `EstacaoCod→station_id`, `Prec→precip`, `vazao→discharge`, `AF→active_fire_count`, `BA→burned_area`, `ET→et`, `DrangAr→drainage_area` mapping is prose only. | `amazon_description.md:43-65`; `amazon_domain.yaml:11-18` | Medium | Violates CLAUDE.md "no hardcoding — all in config." Preprocess can't rename without hardcoding the map. |
| `discharge` ~6% NaN handled by masked loss; only target with NaN. | `amazon_description.md:62,92` | — (correct) | Consistent with the shared masked-MSE. |
| KGE/PBIAS undefined for all-zero fire targets at a station (`mean=Σ=0`). | `amazon_description.md:164`; `shared/metrics.py:27-39` | Medium | Fire counts/burned area are frequently zero; metrics will produce inf/NaN without guards. |

**Implementability checklist — Amazon**

| Stage | Status | Notes |
|---|---|---|
| Preprocess | Needs clarification | Add the raw→clean rename map to config; decide the climatology policy (Exec #4); the rest (station filter, cyclical encoding, segmenting, train-only scaler) is fully specified. |
| Train | Ready | `nFeatures=14`, `nTargets=3` fixed; loss/checkpoint specified. Depends on shared stubs + LR finder. |
| Predict | Ready | Parquet schema and last-position inference specified. |
| Evaluate | Needs clarification | Resolve the ground-truth source; specify KGE/PBIAS behavior for zero-valued fire series. |

---

### 2.3 Rangeland

| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| `seq_len=6` (dev **and** production) vs the stated need to learn multi-year pool accumulation (Exec #8). | `rangeland_description.md:113,117`; `rangeland_domain.yaml:36-40` | Medium | A 6-step causal window cannot represent the long-term state the spec says pools require; with same-step mapping and weak predictor correlation, pools likely collapse to a per-site constant. |
| Val/test sites receive the **global** training climatology instead of their own (same as Amazon, Exec #4). | `rangeland_description.md:131` | High | Same train/test feature mismatch. |
| **Tiny per-PFT test sets.** desert-scrub 7 / sagebrush 7 / grass-tree 6 sites; a 70/15/15 site split yields ≈1 (or 0) test site per small PFT. | `rangeland_description.md:107,129` | Medium | Per-PFT test metrics would rest on n=1; 15% of 6 rounds below 1, so the "each PFT in all splits" guarantee needs an explicit rounding/min-count rule. |
| Pool targets near-zero predictor correlation (r≤0.27); same-step mapping. | `rangeland_description.md:113` | Medium | Combined with the 6-month window, expect low/negative NSE for pools; per-target loss logging (specified at `:185`) is the right mitigation to keep, but the design likely can't learn pools well. |
| Output column count mismatch: Step 5 lists 11 value columns (+site+date); Outputs table says "all 12 output columns" and `predictions.parquet`, while Step 3 says save to a directory. | `rangeland_description.md:201,230,191` | Low | Cosmetic count/path inconsistency. |
| NEE excluded as output, derived `RECO−GPP` at inference. | `rangeland_description.md:76,199` | — (correct) | Avoids a redundant, perfectly-collinear target. Good. |

**Implementability checklist — Rangeland**

| Stage | Status | Notes |
|---|---|---|
| Preprocess | Needs clarification | Specify per-PFT split rounding/min-count; decide climatology policy (Exec #4); `KEEP` columns, monthly aggregation rules, `min_records_per_month`, PFT one-hot order all specified in yaml/spec. |
| Train | Ready | `nFeatures=22`, `nTargets=10` fixed; loss/checkpoint specified. Depends on shared stubs + LR finder. (Window-length concern is methodology, not implementability.) |
| Predict | Ready | Last-position inference, NEE derivation, parquet specified (fix the column-count/path wording). |
| Evaluate | Ready | Truth from `test.pkl` (clear, unlike Amazon); metrics/plots depend on shared stubs + div-by-zero guards. |

---

## 3. Cross-domain consistency

Consistency across domains is a stated goal (ahead of S2 unification). Classification:
**(a)** justified by genuine domain differences · **(b)** unjustified — unify now ·
**(c)** unclear — needs your decision.

| Choice | Arctic | Amazon | Rangeland | Class | Note |
|---|---|---|---|---|---|
| Input→target alignment | same-step Xₜ→Yₜ | same-step | same-step | **consistent (a)** internally — but all three diverge from CLAUDE.md's "next-token." Fix the doc, not the specs. |
| Split philosophy | by pixel | by station | by site (PFT-stratified) | **(a)** | All spatial CV with train-only scaler. Rangeland's PFT stratification justified by imbalance. None holds out time (Exec #3). |
| Normalization | nan-aware z-score, train-only | z-score, train-only | z-score, train-only | **(a)** consistent | Arctic uses `nanmean/nanstd` (targets have NaN); others plain. Fine. |
| Per-unit climatology feature | none (uses native static layers for all pixels) | mean+std of 3 vars, train-only + **global sub** for val/test | mean of 5 vars, train-only + **global sub** | **(b)/(c)** | Presence is (a)-justified (tabular domains synthesize a site fingerprint Arctic gets from static rasters). The global-substitution is **(b) unjustified** (computable from each unit's own predictors). See Exec #4 / Open Q1. |
| LR config schema | `learning_rate` (mutated in place) | `initial_lr`+`optimized_lr` | `initial_lr`+`optimized_lr` | **(b)** | Unify to `initial_lr`+`optimized_lr`; avoid hand-mutating the SSOT config. |
| "Best model" criterion | min val masked-MSE | same | same | **(a)** consistent | Clear and identical; no yaml key, but unambiguously specified. |
| Loss | masked MSE on normalized targets | same | same | **(a)** consistent | Good. |
| Inference scheme | last-position, stride=1, lead NaN | same | same | **(a)** consistent | Note (Low): training supervises *all* positions while inference uses only the last — identical across domains, mild train/infer objective mismatch. |
| Prediction/eval file format | NetCDF + per-pixel `metrics.csv` | parquet + per-station `metrics.csv` | parquet + per-site `metrics.csv` | **(a)** for format (gridded vs tabular) | But `metrics.csv` column names/case diverge (`target` vs `target_variable`; lowercase vs uppercase) → **(b)**, and `log_experiments.md` only covers Arctic (Exec #6). |
| Shared plotting | uses Arctic-shaped `plot_metric_boxplot`/`plot_pred_vs_true` | needs per-station/per-target | needs per-site/10-target | **(b) leaking abstraction** | See §4. |

**Leaking abstractions (force shared code to special-case a domain):**
- `shared/plots.py` — **yes**: `plot_pred_vs_true` (4-panel, ALD/GPP/RECO/VEGC) and
  `plot_metric_boxplot` (SSP/period/pixels) are Arctic-shaped and don't fit Amazon (3 targets,
  per-station) or Rangeland (10 targets, per-site). `plot_spatial_map` is Arctic-only — that one
  is (a)-justified (only Arctic is gridded). `shared/plots.py:45-64,76-82`.
- `shared/transformer.py` — **no** (positive finding): `num_features`/`num_targets` are
  constructor args (`:40`), the causal mask is built from runtime `T` (`:75-77`), `max_len=5000`
  covers all `seq_len`. Cleanly serves 14/3, 22/10, and Arctic's variable feature count.
- `shared/metrics.py` — **no**: domain-agnostic 1-D arrays. (But it must gain div-by-zero guards
  that Amazon/Rangeland will exercise — Exec #7.)
- `config/config.py` — **no**: generic mode resolution.

---

## 4. Shared code findings

### `shared/transformer.py`
| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| Causal mask is genuinely causal. `triu(…, diagonal=1)→-inf` zeroes the future; additive mask means query *i* attends only to keys *j≤i*. | `shared/transformer.py:68-71,77` | — (correct) | Verified. With per-position output and same-step targets, position *t* predicts Yₜ from X₀..ₜ — matches the intended "inputs up to *t* → target at *t*." |
| Clean cross-domain abstraction; no baked-in feature/seq-len assumptions. | `:40,75-77` | — (positive) | Serves all three domains without special-casing. |
| Positional encoding breaks for **odd** `hidden_dim`: `pe[:,1::2]` (floor) vs `div` (ceil) shape mismatch. | `:15-19` | Low | Latent; all configs use even dims (8/128). Worth a guard or a comment. |
| `model.architecture` is read from yaml but the module always builds a transformer; `log_experiments.md` references an `lstm` arch with no implementation anywhere. | `arctic_domain.yaml:74`; `log_experiments.md:24` | Low | Orphaned/forward-looking config; harmless now, but don't log a non-existent arch. |

### `shared/metrics.py`
| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| Documented formulas are correct: RMSE, NSE (`1−Σe²/Σ(o−ō)²`), KGE (`1−√((r−1)²+(α−1)²+(β−1)²)`, original Gupta-2009 std-ratio α), PBIAS (`100·Σ(p−o)/Σo`). | `:18,23,28-33,38` | — (correct) | Standard. |
| No degenerate-input handling (all stubs): constant `obs`→NSE div0; `std(obs)=0`/`mean(obs)=0`→KGE/PBIAS div0; `r` undefined for constant series; `_clean` can empty the arrays (n<2). | `:11-44` | Medium | Will produce inf/NaN on Amazon zero-fire and Rangeland constant pools. Define the degenerate-case contract (return NaN vs raise) before implementing. |
| PBIAS sign convention (`p−o` → positive=overprediction) differs from the common Moriasi (`o−p`) convention. | `:38` | Low | Fine if documented; ensure plots/interpretation match. |

### `shared/plots.py`
| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| All functions are `NotImplementedError` stubs. | `:31-82` | Medium | Train/eval across all domains depend on these; they're a blocking scaffolding gap (expected, but the SSOT calls the Arctic pipeline "complete"). |
| `plot_pred_vs_true` (4-panel, ALD/GPP/RECO/VEGC) and `plot_metric_boxplot` (SSP/period/pixels) are Arctic-shaped. | `:45-64` | Medium | Leaking abstraction (see §3); must generalize to N targets and non-SSP grouping for Amazon/Rangeland. |
| Skewed targets need a log-scale option. Amazon `discharge`/`active_fire_count`/`burned_area` and Rangeland pools (60–7,332) are heavy-tailed; linear pred-vs-true scatter compresses small values and can mislead. | `:40-50` | Low | Design recommendation for the (unwritten) implementation. |

### `config/config.py`
| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| Mode resolution works for exactly `dev`/`production`; an invalid `mode` string silently merges **no** profile and pops `dev`, leaving flat keys absent → late `KeyError` (e.g. `hidden_dim`) far from the cause. | `config/config.py:19-30` | Medium | Violates "fail loudly." Validate `mode ∈ {dev,production}` at load. |
| Missing/misnamed sections fail silently: `cfg.get(section)` + `if sec is None: continue` skips a typo'd section (`preprocesing`) and never merges its profile. `mode` itself defaults silently via `.get("mode","dev")`. | `:21,25-27` | Medium | No required-field validation; surfaces downstream. |
| `domain` is validated against `DOMAINS` (loud `ValueError`); transformer reads `cfg["model"][...]` with `[]` (loud `KeyError`). | `:11-12`; `transformer.py:42-47` | — (positive) | Those paths fail loudly, which is good. |

---

## 5. Governance & documentation findings (CLAUDE.md + project_management/)

| Finding | Location | Severity | Reasoning |
|---|---|---|---|
| SSOT DOMAINS table contradicts reality on all three domains (rangeland missing; amazon "Not Started/not written"; arctic "Complete/fully implemented"). Diary repeats "fully implemented." | `current_project_status.md:27-29,54-58` | High | The table is the declared authority for domain stage; as-is it misleads anyone trusting the SSOT. Update to reflect: all three at "Preprocessing/spec" stage with empty scripts + stubbed shared metrics/plots. |
| CLAUDE.md frames the whole project as "predicts the next time step's output (next-token prediction)," but every spec + the transformer implement same-step emulation. | CLAUDE.md "Project" § | Critical | Headline ambiguity (Exec #1); reconcile the doc to the design (or change the design if forecasting is truly intended). |
| `log_experiments.md` declares itself the global logging SSOT but only specifies Arctic. | `log_experiments.md:9-11,57-130` | High | Amazon/Rangeland artifacts (parquet, per-station/site metrics, time-series plots) are undefined; "not listed ⇒ not logged" would forbid them. |
| `log_experiments.md` vs `arctic_description.md` disagree on the `metrics.csv` schema (case, lat/lon, `variable` vs `target`). | `log_experiments.md:82-97`; `arctic_description.md:191` | Medium | Two authorities, one file. |
| LR config divergence + "mutate `learning_rate` in config" pattern conflicts with the SSOT principle that the yaml is the authoritative, stable config. | `arctic_description.md:151`; `arctic_domain.yaml:92` | Medium | Adopt `initial_lr`+`optimized_lr` everywhere. |
| The LR-finder package (`pytorch-lr-finder`) is required by all three training specs but is not noted in `requirements.txt`/`environment_spec.md`. | `arctic_description.md:151`; `amazon_description.md:135`; `rangeland_description.md:176` | Low | Add the dependency, or make the LR-finder step optional. |
| CLAUDE.md Layout lists `metrics.py`/`plots.py` as shared modules; both are `NotImplementedError` stubs. Combined with the "Complete" claim, the docs overstate readiness. | CLAUDE.md "Layout" §; `shared/metrics.py`, `shared/plots.py` | Low/Medium | Reconcile stage claims with code reality. |
| Placeholder docstring convention drifts: most files use a module docstring + `§` spec pointer, but `arctic_domain/02_train.py` and `run_arctic.py` use `#` comments; `02_train.py` embeds premature design ("ArcticDataset sliding-window class…") and `run_arctic.py` references an undefined "dev or prod mode." | `domains/arctic_domain/02_train.py:1-4`; `run_arctic.py:1-4` | Low | Standardize on the docstring+pointer convention; drop premature/undefined notes. |
| Minor staleness: diary CURRENT dated 2026-06-09 ("working on project_management") vs today 2026-06-17; CLAUDE.md pointer marks all three domains "[Current]" while SSOT marks only amazon active. | `current_project_status.md:35-48`; CLAUDE.md "Current Stage" § | Low | CLAUDE.md is a non-authoritative pointer per the SSOT table, so this is low-risk, but worth syncing. |

---

## 6. Open questions for you

1. **Forecast vs emulate (design intent).** Should the model predict the *next* step
   (Xₜ→Yₜ₊₁, true forecasting, requiring a target shift) or the *current* step
   (Xₜ→Yₜ, emulation/nowcast, as currently specced)? The three specs + transformer all do the
   latter; CLAUDE.md says the former. For Arctic/Rangeland emulation, same-step is correct; for
   Amazon "forecasting" it likely is not. Which is the true goal per domain? (Drives Exec #1, #2.)

2. **Temporal evaluation.** Is spatial-only CV (no future holdout) the intended evaluation, or do
   you also want a held-out future period to measure extrapolation-in-time skill — especially for
   Arctic's projected SSP period and Amazon's "forecast"? (Drives Exec #3.)

3. **Climatology features for val/test units.** Is the global-mean substitution intentional
   leakage-avoidance, or should each unit use its own predictor-derived climatology (my
   recommendation)? If intentional, what's the rationale, given these are inputs, not targets?
   (Drives Exec #4.)

4. **Degenerate-metric contract.** For NSE/KGE/PBIAS on constant or all-zero series (Amazon fire,
   Rangeland pools), should the functions return `NaN`, skip the unit, or raise? This needs to be
   fixed before `metrics.py` is written. (Drives Exec #7.)

5. **Rangeland pool modeling.** Given pools need multi-year context but `seq_len=6`, do you want a
   longer window for Rangeland, a separate treatment for pools vs fluxes, or to accept that pools
   may be poorly predicted and document it? (Drives Exec #8.)

6. **Single logging spec.** Should `log_experiments.md` be generalized to a domain-agnostic schema
   (it currently forbids anything not Arctic-shaped), and should `metrics.csv` column
   names/case be unified across domains for shared report/plot code? (Drives Exec #6.)
