# [Title — placeholder]

## Abstract

<!-- draft last, about 200-250 words -->

## 1. Introduction

Climate change is already reshaping ecosystems in ways that are measurable and consequential. Wildfires in the Amazon basin are becoming more frequent and severe (cite), permafrost across the Arctic is thawing faster than earlier assessments expected (cite), and rangelands that support grazing and store carbon are shifting in productivity and composition (cite). These changes affect water supplies, carbon budgets, biodiversity, and the livelihoods of the communities that depend on these landscapes. Anticipating how these systems will respond to future climate conditions is essential for adaptation planning, resource management, and policy, and doing so requires models that can translate climate and land-use information into reliable predictions of the variables decision-makers actually care about, such as river discharge, fire risk, and carbon exchange.

Process-based simulation models have long been the standard tool for this task. Models such as the Terrestrial Ecosystem Model (TEM) and RangeSTAR encode detailed physical and biological mechanisms and can be trusted to extrapolate beyond the observed record, but running them at scale is computationally expensive (cite) and often requires expert tuning for each new location or scenario. Machine learning offers a faster, more flexible alternative that learns direct relationships between available inputs and target variables. Recently, large foundation models trained on broad environmental datasets have been proposed as a way to build one model that transfers across many domains and locations (cite). In practice, these models are usually trained on public data that only loosely resembles the specific ecosystems, instruments, and variables that matter for a given real-world application, so their out-of-the-box performance in any one domain tends to be mediocre (cite). For decision-relevant predictions in a specific domain, a model built and tuned for that domain still tends to outperform a generic one, which is why dedicated single-domain models remain the common approach across environmental science.

At the same time, building a separate model for every domain has real costs. Two of the three domains studied here have comparatively little training data, which limits how well a model trained only on that domain's own data can generalize to new locations. A model that draws on multiple domains at once has the potential to share statistical strength across them (cite), which can improve generalization in the data-scarce domains if what one domain's model learns about seasonal dynamics or predictor relationships transfers usefully to another. A single shared model is also simpler to maintain, update, and deploy than a separate model per domain. Our aim in this paper is not simply to build efficient emulators of each process model, since that could be achieved more directly by running the process models themselves under many forcing scenarios. Instead, our central question is whether a single model architecture, trained across domains, can match or improve on dedicated single-domain models, and whether what the model learns from a data-rich domain can be usefully transferred to data-scarce ones.

Sharing a model across domains is not free of risk. The inputs, targets, spatial units, and underlying data sources differ substantially across domains, which can make it hard for one architecture to fit all of them well. Training on multiple domains at once can also produce negative transfer, where information from one domain actively hurts performance on another instead of helping it (cite). A domain with much more training data can dominate a shared model's learned representations at the expense of smaller domains, and a shared model can be harder to interpret than a set of specialized ones. Two-stage training, where a model is first pretrained across all domains and then fine-tuned separately for each one, is a common strategy for managing these issues (cite), and it is the approach we adopt here.

The three domains studied here were chosen partly because they differ from each other in ways that are typical of real-world environmental data, not in spite of it. In the Arctic domain, our targets are gross primary productivity and ecosystem respiration, and they come from TEM, a process model, so here the model is trained to emulate a physical simulation rather than a direct observation. In the Rangeland domain, our targets again include gross primary productivity and ecosystem respiration, among others, and they come from RangeSTAR in the same way. In the Amazon domain, our targets, river discharge and wildfire activity, come from direct station and satellite observation rather than from a process model. The three domains also differ in temporal coverage. Arctic data span both a historical period and multiple future climate scenarios extending to 2100, while Amazon and Rangeland data cover the historical period only. This mix of simulated and observed targets, and of historical-only and historical-plus-scenario coverage, is deliberate. A real deployment of a multi-domain model has to work with data collected for different purposes, at different times, using different methods, and testing whether a unified approach can handle that kind of heterogeneity is itself part of what this study evaluates, not a limitation to work around.

It is also worth being precise about what kind of prediction task this is. The model takes a sequence of predictor values from an early time step up to the current step and predicts the target variable at that same current step. It does not take today's conditions and predict several months or years ahead. In that sense this is a same-step, or nowcasting, model rather than a forecasting model (cite). Forecasting has obvious value, but our interest here is longer-term projection. Once trained, the same model can be driven with projected future climate and land-use forcing data, month by month, to trace out how discharge, fire activity, carbon exchange, or permafrost depth might evolve under a given future scenario. This kind of projection is directly useful for long-term risk assessment and planning under future climate conditions, which is the setting we are ultimately targeting.

This paper makes four contributions. First, we develop dedicated causal transformer emulators for three environmental domains that differ in their inputs, targets, spatial structure, and underlying data sources. Second, we develop a shared multi-domain model, trained with a pretraining and fine-tuning approach, that draws on all three domains at once, including transfer from the data-rich Arctic domain to the comparatively data-scarce Amazon and Rangeland domains. Third, we apply a consistent spatial-generalization evaluation across all domains, testing every model on locations it never saw during training rather than on future time periods. Fourth, we report where a shared multi-domain approach helps, where it does not, and what that implies for applying this kind of model to other environmental domains with similarly heterogeneous data. Section 2 describes the study areas and data. Section 3 describes the modeling approach and evaluation protocol. Sections 4 and 5 present and discuss results. Section 6 concludes.

## 2. Study area and Data

This study covers three environmental domains that differ in geography, spatial structure, and the origin of their target data. The three datasets are also unbalanced in scale. The Arctic domain draws on 263 grid tiles at approximately 4 km resolution, and individual tiles vary considerably in size, with the largest containing close to 9,800 land pixels, so the total number of Arctic pixels is far larger than the site or station counts in the other two domains. Each Arctic pixel in turn spans more than a century of data. The Rangeland and Amazon domains include far fewer spatial units, 59 sites and 98 stations respectively, so the Arctic is by far the largest and most data-rich of the three.

All three domains are ultimately prepared at a common monthly time step, so the same model architecture and evaluation protocol apply across them, even though the underlying data do not all start out at that resolution. Amazon's records are already monthly. Rangeland's process-model output is recorded at roughly 5-day intervals and is aggregated to monthly during preprocessing. Arctic's climate inputs are already monthly.

The domains are not entirely unrelated, however. Arctic and Rangeland targets both come from process models rather than direct measurement, and both include gross primary productivity and ecosystem respiration among their targets, so the two domains describe a similar underlying carbon exchange process even though they represent different ecosystems. The Amazon domain differs in kind rather than degree. Its targets, river discharge and wildfire activity, are directly observed rather than modeled, and describe hydrology and fire rather than carbon exchange. Despite these differences, the three domains share some of the same input variables. Precipitation and temperature, for example, are climate drivers common to all three domains, even though each domain also has additional predictors specific to its own ecosystem and process model. Table 1 summarizes the three datasets, and the subsections below describe each one in more detail.

### 2.1 Arctic

The Arctic domain covers the circumpolar Arctic region, organized into 263 grid tiles at approximately 4 km resolution. Model inputs are monthly climate variables (air temperature, precipitation, net radiation, and vapor pressure), atmospheric CO2 (recorded yearly and interpolated to monthly values), and static soil, vegetation, and fire-related covariates. The target variables come from the Terrestrial Ecosystem Model (TEM) and include gross primary productivity and ecosystem respiration, both at monthly resolution. Data cover a historical period together with two future climate scenarios, SSP1-2.6 (1901 to 2100) and SSP5-8.5 (2025 to 2100 only). The train, validation, and test split is made at the pixel level and stratified by grid, so every grid contributes pixels to all three splits and held-out pixels are genuinely unseen locations rather than unseen time periods.

### 2.2 Rangeland

The Rangeland domain covers 59 monitoring sites in the AmeriFlux and NEON networks, grouped into four plant functional types, grass (39 sites), desert scrub (7), sagebrush (7), and grass-tree (6), with records spanning 2002 to 2024. Model inputs combine a satellite-derived vegetation index (EVI2) with gridded meteorological drivers, including temperature, soil temperature, and precipitation, from the NLDAS and Daymet products. The target variables come from the RangeSTAR process model and include gross primary productivity, ecosystem respiration, maintenance respiration, and growth respiration, four targets in total. The train, validation, and test split is made at the site level and stratified by plant functional type, so every type is represented in all three splits, with 35, 11, and 8 sites respectively. RangeSTAR's underlying output is recorded at approximately 5-day intervals and is aggregated to monthly for consistency with the other two domains.

### 2.3 Amazon

The Amazon domain covers 98 river gauge stations across the Amazon basin, with monthly records from 2000 to 2024. Model inputs are precipitation, maximum and minimum temperature, evapotranspiration, vapor pressure deficit, and drainage area, plus each station's own long-term climatological averages. Unlike the Arctic and Rangeland domains, the three target variables here, river discharge, active fire count, and burned area, come directly from gauge and satellite observations rather than from a process model. The train, validation, and test split is made at the station level, with 59, 20, and 19 stations respectively.

**Table 1. Dataset summary.**

| Domain | Data source | Spatial unit | Units (train / val / test) | Temporal coverage | Dynamic inputs | Static inputs | Target variables |
|---|---|---|---|---|---|---|---|
| Arctic | TEM process model | Grid pixel (~4 km) | pixel-level, all 263 grids represented in each split | 1901-2100 (SSP1-2.6), 2025-2100 (SSP5-8.5) | air temperature, precipitation, net radiation, vapor pressure, CO2 | soil texture, drainage, fire return interval, topography, vegetation type | gross primary productivity, ecosystem respiration |
| Rangeland | RangeSTAR process model | Monitoring site | 35 / 11 / 8 (59 total) | 2002-2024 | vegetation index (EVI2), air temperature (mean, max, min), soil temperature, soil moisture (2 depths), vapor pressure deficit, incoming shortwave radiation, precipitation | clay content, plant functional type | gross primary productivity, ecosystem respiration, maintenance respiration, growth respiration |
| Amazon | Direct observation | River gauge station | 59 / 20 / 19 (98 total) | 2000-2024 | precipitation, maximum temperature, minimum temperature, evapotranspiration, vapor pressure deficit, drainage area | station long-term climatological means (precipitation, max/min temperature) | river discharge, active fire count, burned area |

## 3. Methods

### 3.1 The Prediction Task and Shared Model Architecture

Every model in this study performs the same underlying task, adapted to each domain's own inputs and targets. Given a sequence of predictor values covering months 1 through T, the model predicts the target variables at month T, using only information available up to and including that month. This is a causal, same-step prediction task rather than a multi-step forecast, and it applies identically whether the sequence describes an Arctic grid pixel, a Rangeland monitoring site, or an Amazon gauge station.

All three domains, and the shared multi-domain model described in Section 3.5, use the same underlying architecture, a causal transformer implemented once (`shared/transformer.py`) and reused across every experiment in this study. The model has four stages. First, a linear input projection maps each time step's raw predictor vector, of dimension equal to that domain's number of input features, up to a shared hidden dimension. Second, a fixed sinusoidal positional encoding is added to this projected sequence so the model can distinguish earlier from later time steps, since a transformer's attention mechanism has no inherent notion of order on its own. Third, the sequence passes through a stack of transformer encoder layers, each combining multi-head self-attention with a feedforward sublayer, GELU activation, and residual connections. A causal mask restricts every time step to attend only to itself and earlier time steps, so a prediction at month T can never depend on information from month T+1 or later, matching the same-step task defined above. Fourth, a linear output head maps the encoder's hidden representation at each time step down to that domain's number of target variables. Amazon's three targets (discharge, active fire count, burned area) are all non-negative by definition, so its output head uses a softplus activation instead of an unconstrained linear output, which is smooth and differentiable everywhere and does not permanently zero out a unit the way a ReLU can if a prediction goes negative early in training.

In shape notation, the model maps an input tensor of size (batch, seq_len, num_features) to an output tensor of size (batch, seq_len, num_targets), with num_features and num_targets set independently for each domain. Table 2, in the next subsection, gives the exact feature and target counts used for each domain.

### 3.2 Data Preparation and Preprocessing

Each domain's raw records are converted into a common tensor format before reaching the model. The dynamic (time-varying) and static (time-invariant) predictors listed in Table 1 are combined into a single per-time-step feature vector, with static predictors simply repeated at every time step. Two domains add derived features beyond their raw measurements. Rangeland and Amazon both add a cyclical encoding of calendar month, the sine and cosine of month-of-year, so the model can represent the repeating annual cycle without treating December and January as numerically distant. Amazon and Rangeland also add per-unit climatological means, computed separately for each station or site from its own historical predictor records and never from another unit's data, which would leak information across the spatial split described in Section 2, giving the model a stable sense of each location's typical conditions alongside the time-varying signal.

A shared sliding-window procedure (`shared/dataset.py`) then turns each unit's full time series into fixed-length training examples. A window of `seq_len` consecutive months is slid over each unit's record, and the model is trained to predict the target at the final month of every window from the predictor values across the whole window. All three domains use a sequence length of 12 months. During training, windows are drawn at a domain-specific stride to control how densely a long time series is sampled, Amazon and Rangeland are small enough to train at stride 1, using every possible window, while Arctic uses a much coarser stride because its far larger pixel count would otherwise produce more training windows than is practical to fit in a single run, a choice examined in detail in Section 3.4. During evaluation, stride is always 1 for every domain, so a prediction is generated for every possible month in the held-out record.

Every predictor and target is standardized to zero mean and unit variance using a scaler fit on the training split only, then applied unchanged to validation and test data, so no information from held-out units influences the scaler. Missing target values, which occur naturally in observed data, about 6% of Amazon's discharge records, for instance, are excluded from the loss at the position level rather than dropped or imputed, using the masked mean squared error described in Section 3.3. Missing predictor values arise only in the Arctic domain, as sparse gaps in a few feature columns such as fire-related covariates on land pixels, and are filled with zero after standardization, which corresponds to that feature's own mean, so an imputed value contributes no signal rather than a distorting one. Amazon's and Rangeland's predictors are fully observed, with no missing values and so no imputation needed there.

**Table 2. Model input and output dimensions per domain.**

| Domain | Input features | Target variables | Sequence length | Training stride |
|---|---|---|---|---|
| Arctic | number of static covariate channels plus 5 dynamic channels (4 climate variables and CO2), fixed at preprocessing time | 2 | 12 months | coarse, tuned in Section 3.4 |
| Rangeland | 22 | 4 | 12 months | 1 |
| Amazon | 14 | 3 | 12 months | 1 |

### 3.3 Training the Individual Domain Models

All three individual domain models are trained with the same optimization recipe, implemented once (`shared/training.py`) and reused unchanged across domains. The loss is a masked mean squared error, computed only over time steps where a target observation actually exists, so missing values never contribute a gradient. Parameters are updated with AdamW, using a learning rate found separately for each domain (below) and a weight decay of 1e-4. The learning rate follows a linear warmup over the first several epochs, growing from a small starting value to the found learning rate, followed by cosine decay to zero over the remaining epochs, a warmup that avoids destabilizing the randomly initialized transformer with a large step early in training, while the cosine decay lets the model settle into a stable solution as training progresses. Gradients are clipped to a maximum norm of 1.0 before every optimizer step, a standard safeguard against the loss spikes an unusually difficult batch or a high post-warmup learning rate can otherwise cause in transformers.

The learning rate itself is chosen with a learning-rate range test rather than fixed by hand. Before the real training run, the optimizer's learning rate is swept upward across roughly 100 mini-batches, from a very small starting value toward 1.0, while recording the training loss at each step, and the learning rate at the point of steepest loss decrease is taken as the working learning rate for that domain's full training run. This is run once per domain, and once more for the shared multi-domain model in Section 3.5, rather than searched by hand, which keeps the choice reproducible and removes a manual tuning step that would otherwise need to be repeated every time the data or model size changes.

Training runs for up to 100 epochs but stops early if it stops improving. Validation loss is checked every 1 to 2 epochs depending on domain, and the checkpoint with the lowest validation loss so far is saved after every improving check. Training stops once a set number of checks in a row, the early stopping patience, 5 to 12 evaluations depending on domain, pass without a new best validation loss, and the saved checkpoint from the best-performing check is the one carried forward to evaluation. The reported model for each domain is therefore never the last epoch trained, but the epoch that generalized best to held-out validation data during training.

Table 3 lists the architecture and training hyperparameters used for each domain's production run. Model size is scaled roughly to each domain's data volume, Arctic's model is substantially larger than Amazon's or Rangeland's, reflecting its much larger pixel count, while Rangeland, the smallest dataset, also uses the highest dropout rate to limit overfitting. Hyperparameters were set from data volume and standard transformer scaling heuristics rather than a full grid search, given the number of domains and experiments already involved in this study, with the learning rate as the one exception that is tuned automatically for every run as described above. All models are implemented in PyTorch and trained on a single NVIDIA A100 40GB GPU.

**Table 3. Architecture and training hyperparameters by domain (production configuration).**

| Domain | Hidden dim | Layers | Attention heads | Feedforward dim | Dropout | Batch size | Warmup epochs | Early stopping patience |
|---|---|---|---|---|---|---|---|---|
| Arctic | 256 | 6 | 8 | 1024 | 0.15 | 2048 | 10 | 5 (checked every 2 epochs) |
| Rangeland | 64 | 3 | 4 | 256 | 0.30 | 64 | 5 | 12 |
| Amazon | 128 | 3 | 4 | 512 | 0.20 | 256 | 10 | 12 |

Each domain also has its own form of data imbalance, handled explicitly rather than left to chance. Rangeland's four plant functional type groups are highly unequal in size (39, 7, 7, and 6 sites), so the train, validation, and test split is stratified within each group rather than drawn from the pooled site list, guaranteeing every group is represented in every split even though a naive random split could otherwise leave a small group entirely out of validation or test. Amazon's discharge target is missing for about 6% of station-months, and rather than drop or impute these positions, they are simply excluded from the loss, the same masked mean squared error used for all missing-target handling in this study.

Because the transformer's weights are randomly initialized and mini-batches are shuffled independently on every run, a single training run carries some irreducible run-to-run variance. The final production result for each domain, reported in Section 4, is the mean over 3 to 5 independently seeded training runs of that domain's final configuration, rather than a single run, so the reported metrics reflect the model's typical behavior rather than one particular random draw.

*Figure 1: [caption placeholder — per-domain workflow, from raw inputs through preprocessing and windowing to the transformer and the output prediction]*

### 3.4 Arctic Data Density and Dataset Size Experiments

Arctic's dataset is different in kind from Amazon's and Rangeland's. Its 263 grids together contain far more pixels than the other two domains have spatial units, and running every pixel through the windowing procedure at stride 1 would produce more training windows than is practical to fit into a single training run. Amazon and Rangeland, by contrast, are small enough that no such subsampling is needed. Building a training set for Arctic therefore requires an explicit choice about how to subsample the full pixel-by-time-window space at a fixed training budget, a choice the other two domains do not face, and it turned out to matter a great deal for how well the model learned.

An early comparison trained the same model on datasets built at different levels of pixel sampling density, controlling for the total number of training windows. Sampling more pixels less densely each, a coarser stride per pixel spread across more of the 263 grids, consistently outperformed sampling fewer pixels more densely, even when the total number of training windows was held fixed. The clearest evidence came from an ablation that shrunk the model and changed where in each sequence the loss was scored, changes that helped considerably on a sparse, low-pixel-diversity dataset but made results worse once pixel diversity was already high, showing that the sampling density of the training set, not the model's size or the loss's scoring position, was the dominant lever for this domain.

A second refinement addressed a subtler issue with how windows are sampled. Without adjustment, every pixel's training windows start at the same fixed set of calendar months, so the model repeatedly sees the same seasonal alignment across space. Introducing a small, per-pixel deterministic offset before windowing, so different pixels' windows start at different points in the calendar year, improved validation loss and most target metrics over the unstaggered version at the same sampling density, at no extra preprocessing cost.

With a sampling density and a windowing strategy fixed, a further comparison scaled the training set size itself, from an initial 50,000-window budget up to 500,000 windows at the same density and staggering settings, holding the validation and test pixel populations fixed throughout so the comparison isolated training-set size alone. Every target improved with more training data, with no sign of saturating at 500,000 windows. The Arctic model used in this study was trained at a 500,000-window budget with a coarse per-pixel sampling stride tuned through this process, restricted to the two flux targets described in Section 2 rather than the full four-target set explored during this investigation.

### 3.5 Building a Shared Multi-Domain Model

The multi-domain model reuses the same causal transformer backbone as the individual domain models, but wraps it with domain-specific input and output layers so a single shared encoder can serve all three domains at once. Each domain first passes its own feature vector through its own linear projection, mapping that domain's native number of input features up to a common embedding dimension shared by all three domains. This common-dimension sequence then passes through one shared transformer encoder, the same causal self-attention architecture described in Section 3.1, whose weights are shared across all three domains rather than duplicated per domain. Finally, each domain's encoded sequence passes through its own small multi-layer perceptron head, mapping the shared embedding dimension back down to that domain's own number of target variables.

In shape notation, a domain's input of size (batch, seq_len, num_features_domain) is projected to (batch, seq_len, common_dim), passed through the shared encoder unchanged in shape, and mapped by that domain's head to (batch, seq_len, num_targets_domain). Only the projection and head layers are domain-specific, the transformer encoder in between is the one component this architecture is designed to share.

Training happens in two stages. In the pretraining stage, all three domains are trained jointly. At every optimizer step, one batch is drawn from each domain, a forward and backward pass is run for each, and the three domains' losses are averaged before the shared weights are updated, so every update reflects information from all three domains simultaneously rather than one at a time. Because Arctic's dataset is far larger than Amazon's or Rangeland's, an epoch is defined by Arctic's batch count, and the two smaller domains are cycled back to the start of their data as many times as needed to keep pace, so every domain contributes the same number of batches to every update step regardless of how much data it actually has. This equal per-step mixing is the main mechanism this study uses to prevent the shared representation from being dominated purely by whichever domain happens to have the most data, whether it is sufficient to fully offset Arctic's much larger volume is assessed empirically in Section 4, not assumed here.

In the fine-tuning stage, the shared transformer encoder and the three domain-specific input projections are frozen at their pretrained values, and only each domain's output head is further trained, independently, on that domain's own training data. Because the frozen encoder and projections carry no trainable parameters at this stage, fine-tuning only has to learn a small, domain-specific mapping from the shared representation to that domain's targets, a much smaller optimization problem than training the whole model from scratch. This produces one fine-tuned checkpoint per domain, each consisting of the shared pretrained backbone plus that domain's own fine-tuned head, which Section 3.6 compares against the dedicated single-domain models from Section 3.3.

The shared model's hyperparameters, embedding dimension, number of layers, attention heads, are set at least as large as the standalone Arctic model's own hyperparameters, since the shared encoder has to represent all three domains at once rather than one. Pretraining and fine-tuning each follow the same learning-rate-finder and cosine-schedule procedure described in Section 3.3, run once for the shared pretraining stage and once per domain for fine-tuning.

*Figure 2: [caption placeholder — multi-domain pretraining and fine-tuning workflow]*

### 3.6 Evaluation Metrics and Comparison Design

Every model in this study, individual and multi-domain alike, is evaluated the same way, on spatial units held out entirely from training. A test pixel, site, or station is never seen during training or validation, and its predictions are scored across its full available time range rather than a held-out time period, so the evaluation measures how well a model generalizes to a genuinely new location rather than how well it extrapolates forward in time at a location it already knows.

Four metrics are reported for every target at every held-out unit. Root mean squared error (RMSE) measures the typical size of a prediction error in the target's own units, with 0 meaning a perfect match and larger values meaning worse, it has no upper bound and depends on the target's scale, so it is most useful for comparing models on the same target rather than across different targets. Nash-Sutcliffe efficiency (NSE) compares a model's error to the error of simply predicting the observed mean at every time step, a value of 1 is a perfect prediction, 0 means the model does no better than predicting the mean, and negative values, which are common in this study for the harder targets, mean the model does worse than that naive baseline. Kling-Gupta efficiency (KGE) is a related measure that separately accounts for correlation, variability, and bias between predictions and observations, like NSE it reaches a maximum of 1 at a perfect prediction and has no strict lower bound, but it does not share NSE's tendency to penalize variability errors more heavily than correlation errors, so the two metrics can disagree about which of two imperfect models is better. Percent bias (PBIAS) reports the average prediction error as a percentage of the total observed value, with 0% meaning no systematic bias, positive values meaning the model systematically overpredicts, and negative values meaning it systematically underpredicts.

The core comparison in this study is between three versions of each domain's model, the dedicated single-domain model from Section 3.3, the jointly pretrained multi-domain model before any domain-specific fine-tuning, and the multi-domain model after fine-tuning, all evaluated on the same held-out test units with the same four metrics. Comparing the pretrained-only version against the fine-tuned version isolates what fine-tuning itself adds on top of the shared representation. Comparing both multi-domain versions against the dedicated single-domain model addresses this study's central question, whether sharing a transformer across domains helps, hurts, or makes no difference relative to training a separate model for each domain. Because Arctic is far more data-rich than Amazon or Rangeland, this comparison also functions as a test of transfer learning, whether the shared representation learned largely from Arctic's abundant data measurably improves Amazon's and Rangeland's own results, assessed by whether their multi-domain metrics exceed their single-domain ones. Good generalization, in this study, means held-out metrics close to training-unit metrics and clearly better than a naive mean-prediction baseline, NSE and KGE noticeably above 0, poor generalization means held-out performance far weaker than training performance, or below that naive baseline altogether, which the results in Section 4 show happening for some targets and not others.

## 4. Results

*(Skeleton below — bullets describe what each subsection will report once the corresponding figures exist; full prose and real numbers to follow.)*

- Every result below follows the spatial-generalization protocol from Section 3.6, held-out units never seen in training, scored across their full available time range.
- Section order follows the logic of the modeling pipeline rather than the order things were introduced. The Arctic density and dataset-size sweep (4.1) comes first because it determines the configuration the Arctic individual model actually uses, so 4.2's Arctic numbers are the *output* of 4.1's choice, not an unexplained given.
- Four metrics (RMSE, NSE, KGE, PBIAS, defined in Section 3.6) are available for every target and unit; only a subset appears in the main-text figures below, with the rest held in Table 4 and supplementary material, to keep each figure readable.

### 4.1 Arctic Sampling Density and Dataset Size Results

- Report the capped-stride sweep result and the winning sampling density, and why pixel diversity mattered more than model size or where in the sequence the loss was scored.
- State, in text only, that staggered windowing outperformed vanilla windowing at the winning density, no dedicated figure for this comparison.
- Report the data-size scaling result (50K, 500K, and 2M windows if available) at the winning density and staggering settings, and whether it saturates.
- State the final recipe this sweep justified, the configuration whose results appear as Arctic's numbers in Section 4.2.

*Figure 3: [caption placeholder — Arctic sampling density and dataset-size sweep, two panels. Left: validation NSE and RMSE for GPP and ecosystem respiration across capped sampling stride 50-500. Right: validation NSE and RMSE vs. training-set size at the best stride and staggered windowing.]*

### 4.2 Individual Domain Model Results

- Report each domain's held-out test RMSE, NSE, KGE, and PBIAS by target (exact values in Table 4's Individual column).
- Flag which targets and domains clear the "good generalization" bar from Section 3.6 and which fall short of it.
- Break Rangeland's results down by plant functional type group, flagging the small groups (desert scrub, sagebrush, grass-tree) as high-variance given how few held-out sites they contribute.
- Break Arctic's results down by SSP scenario and by historical vs. projected period.
- These numbers are the reference point Section 4.4 measures the multi-domain versions against.

*Figure 4: [caption placeholder — individual domain model results, one row per domain (Arctic, Rangeland, Amazon), three metric columns per row (RMSE, NSE or KGE, PBIAS). Arctic's row further split by SSP scenario and historical/projected period; Rangeland's row further split by plant functional type group; Amazon's row at the domain level.]*

### 4.3 Multi-Domain Pretraining and Fine-Tuning Results

- Report whether Stage 1 joint pretraining converged smoothly for all three domains together, or whether one domain's loss lagged or dominated.
- Report Stage 1 (pretrained-only) test metrics per domain per target (Table 4's Pretrained column), the shared representation's performance before any domain-specific adaptation.
- Report how quickly each domain's head adapted during Stage 2 fine-tuning.
- Report Stage 2 (fine-tuned) test metrics per domain per target (Table 4's Fine-tuned column), the result carried into Section 4.4.

*Figure 5: [caption placeholder — multi-domain training loss curves, two panels. Left: Stage 1 pretraining loss (train and validation), per domain and overall mean, vs. epoch. Right: Stage 2 fine-tuning loss (train and validation), one line per domain, vs. epoch.]*

### 4.4 Comparing Individual and Multi-Domain Models

- Present the central three-way comparison, Individual vs. Pretrained vs. Fine-tuned, per domain per target, on identical held-out units.
- Assess transfer learning for Amazon and Rangeland specifically, whether fine-tuned performance exceeds their own individual model, meaning the Arctic-informed shared representation helped.
- Assess whether Arctic's own fine-tuned performance holds up despite the smaller domains diluting its share of pretraining.
- Classify each domain and target as positive transfer, negative transfer, or no meaningful difference, tying back to the negative-transfer risk raised in the Introduction.
- Kept deliberately simpler than Figure 4, domain-level only, no SSP or plant-functional-type breakdown, and a single metric, so this figure stays readable as a clean three-way comparison while the fine-grained detail lives in Section 4.2.

*Figure 6: [caption placeholder — individual, pretrained, and fine-tuned model comparison, one row per domain (Arctic, Rangeland, Amazon), domain-level only. Within each row, held-out test RMSE grouped by target, three boxes per target (Individual, Pretrained, Fine-tuned). NSE-or-KGE and PBIAS versions of this figure go in supplementary material, not the main text.]*

**Table 4. [caption placeholder — median RMSE, NSE, KGE, and PBIAS per domain, target, and model stage (Individual / Pretrained / Fine-tuned) across held-out test units.]**

## 5. Discussion

*(to be drafted)*

## 6. Conclusion

*(to be drafted)*

## Data & Code Availability

*(to be drafted)*

## Acknowledgments

*(to be drafted)*

## References
