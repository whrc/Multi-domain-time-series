# Multi-Domain Time-Series Prediction — Executive Summary

## Executive Summary

Multi-domain machine learning models — models trained jointly across multiple related datasets
rather than one domain at a time — can offer substantial practical advantages: a single shared
model is cheaper to develop and maintain than a separate model per domain, it can improve
prediction skill in data-scarce settings by transferring learned representations from
data-richer domains, and, where those representations genuinely generalize, it can produce more
robust predictions than a narrowly-trained model. Despite this promise, multi-domain approaches
remain relatively underexplored in Earth system science, where machine-learned emulators and
predictive models are still, for the most part, built and evaluated one domain at a time.

This study tests that promise directly, using three environmental domains chosen to span a
realistic range of data availability and target type. The Arctic domain is data-heavy: it
emulates the Terrestrial Ecosystem Model (TEM) using decades of monthly gridded climate, soil,
and vegetation inputs across hundreds of circumpolar grid tiles, predicting gross primary
productivity (GPP) and ecosystem respiration (RECO). The Amazon domain is built entirely from
direct observation rather than a process model: monthly climate and land-use drivers at
watershed level are used to predict river discharge and wildfire activity — active fire count
and burned area — at real gauge and monitoring stations. The Rangeland domain emulates a second
process model, RangeSTAR, at a much smaller set of AmeriFlux/NEON tower sites, predicting four
carbon-flux targets: GPP, RECO, maintenance respiration (Rm), and growth respiration (Rg). All
three domains share one modeling convention throughout: a causal, same-step transformer that
consumes a sequence of monthly inputs and predicts the target at that same final step — an
emulator, not a forecaster — evaluated by spatial generalization to sites, pixels, or stations
withheld entirely from training.

Following a strict two-stage design, a dedicated model was first built, tuned, and evaluated
independently for each domain, establishing a fair, well-optimized baseline before any
cross-domain sharing was attempted. This individual-optimization stage mattered in practice: a
systematic hyperparameter search later found that Rangeland's original dedicated model had been
meaningfully under-sized relative to what its data could support, while Amazon's showed no
accuracy sensitivity to architecture size at all across a wide range tested — a useful negative
result in its own right, and a reminder that a domain's "individual baseline" is not a fixed
target but something that itself has to be gotten right before any cross-domain comparison can
be considered fair. Only once each domain's dedicated model was genuinely optimized was a single
shared multi-domain model developed — one transformer trunk, fed by per-domain input
projections and read out by per-domain prediction heads, trained jointly across all three
domains in a pretraining stage and then fine-tuned per domain — and its out-of-sample skill
compared directly against each domain's dedicated baseline on the same held-out sites.

Our results show a clear, if uneven, benefit from sharing. The data-scarce Amazon domain gained
the most: discharge prediction skill (NSE) rose from 0.37 for the dedicated model to 0.76 for
the shared, fine-tuned model; active fire count rose from 0.31 to 0.71; and burned area — the
noisiest of the three targets, with a dedicated-model skill that swings well below zero at some
stations — rose from essentially no skill (0.01) to a moderate and substantially more consistent
0.52. Arctic, the data-richest domain, also improved, from 0.87 to 0.94 for GPP and from a
highly variable 0.47 to a more consistent 0.68 for RECO, though by a smaller margin, consistent
with a domain that already has ample data to learn from on its own. Rangeland's picture turned
out to be more complicated, for reasons tied directly back to the individual-optimization
finding above, discussed further below.

Further to understand and perform deeper diagnostics of these experiments, we performed an
ablation study in which the shared trunk was retrained on every subset of domain pairings —
Arctic with each smaller domain alone, and the two smaller domains together without Arctic — and
each subset's skill compared against the full three-domain model. The results suggested that two
distinct mechanisms are at work on different Amazon targets, not one. Fire and burned-area
prediction improved almost entirely from pairing specifically with the large, data-rich Arctic
domain; pairing with Rangeland alone recovered only a small fraction of the full gain. Discharge,
in contrast, improved substantially from pairing with *either* smaller domain alone, with
Rangeland-only pretraining already recovering most of the benefit seen with all three domains
together — evidence of a more general, shared temporal representation that does not require a
large anchor domain specifically.

Furthermore, to understand how the multi-domain model was actually improving — or in some cases
not improving — upon the individual domain model, we looked into a decomposition of skill
metrics, focusing on each of the components that make up the Kling-Gupta Efficiency metric:
correlation, variability ratio, and bias ratio. We found that the improvement in Amazon is
attributed primarily to the variability and bias components, not correlation: the dedicated
model's timing correlation with observations was already reasonably strong, but its predictions
were badly under-dispersed relative to the real record and systematically biased; the shared
model corrected both of those specifically, while timing correlation improved only modestly. In
other words, sharing across domains did not chiefly teach the model *when* fire or discharge
events happen so much as *how large and how often* they really are — which had been the
dedicated model's weaker point to begin with.

Rangeland complicates a simple "multi-domain always helps" story, and the reason traces directly
back to the individual-optimization finding described above. Once Rangeland's dedicated model
was properly resized following the hyperparameter search, its accuracy matched or exceeded the
shared model on most targets — a reversal of the original comparison, which had been made
against an under-tuned dedicated baseline. This suggests that Rangeland's originally observed
"multi-domain benefit" was, to a significant degree, a capacity artifact of an unfair baseline
rather than genuine cross-domain transfer — a finding that tempers, without overturning, the
case made by Arctic and Amazon. Together, these results suggest that multi-domain sharing
delivers a real and mechanistically explainable advantage specifically where a domain is
genuinely data-scarce and its dedicated baseline is already well-tuned, rather than a universal
improvement that appears regardless of how carefully that baseline was built.

This study provides an end-to-end, mechanistically grounded look at multi-domain time-series
modeling in the Earth system context — combining a physically simulated data-rich domain, a
physically simulated data-scarce domain, and a purely observational domain within one shared
architecture — and shows concretely where, why, and through which specific mechanism
cross-domain sharing pays off. We hope this work encourages broader adoption, and more rigorous
evaluation, of multi-domain approaches in Earth system modeling, where the potential gains for
data-scarce regions and processes remain largely untapped.

## Abstract

Multi-domain machine learning models — trained jointly across related datasets rather than one
at a time — can offer substantial advantages in Earth system modeling, including more efficient
model development, improved prediction in data-scarce settings, and better generalization, yet
remain relatively underexplored in this domain. We test this potential across three domains
spanning a realistic range of data availability: a data-rich Arctic domain emulating the
Terrestrial Ecosystem Model across circumpolar grid tiles (gross primary productivity and
ecosystem respiration), an Amazon domain built entirely from direct observation of river
discharge and wildfire activity, and a Rangeland domain emulating the RangeSTAR process model at
tower sites (gross primary productivity, respiration, and its maintenance and growth
components). Dedicated, individually optimized models were developed for each domain first, then
compared out-of-sample against a single shared model — one transformer trunk trained jointly
across all three domains and fine-tuned per domain. The shared model substantially improved
skill for the data-scarce Amazon domain (discharge NSE 0.37→0.76; active fire count 0.31→0.71;
burned area 0.01→0.52) and moderately improved data-rich Arctic. An ablation study decomposing
the shared model by domain subset showed two distinct mechanisms behind Amazon's gains: fire and
burned-area skill depend specifically on pairing with the large Arctic domain, while discharge
improves from pairing with either smaller domain alone. A complementary decomposition of the
Kling-Gupta Efficiency metric showed Amazon's gains are driven primarily by corrected prediction
variability and bias, not timing correlation. A parallel hyperparameter search revealed
Rangeland's original dedicated model had been under-sized; once corrected, it matched or
exceeded the shared model on most targets, indicating its original multi-domain benefit was
substantially a baseline-capacity artifact rather than genuine cross-domain transfer. Together,
these results show that multi-domain sharing delivers a genuine, mechanistically explainable
advantage for data-scarce domains with well-tuned baselines, offering a template for wider,
more rigorously evaluated adoption of multi-domain modeling in Earth system science.
