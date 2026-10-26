# Figure 2 design spec: methodology sketches

Implemented in `make_figure2_schematics.py` → `fig2a_individual_domain_sketch` and
`fig2b_multidomain_two_stage_sketch` (PNG in `figures/`, SVG in `figures/svg/`). Historical
design record; not kept in sync with the pipeline.

## Architecture
- **2a, individual:** one `TransformerModel` per domain (`shared/transformer.py`), raw features in,
  no shared weights. Loss: masked MSE in standardized space.
- **2b, multi-domain** (`domains/multi_domain/model.py`): per-domain `Linear` projection
  (`nFeatures_d → D`) → shared transformer → per-domain MLP head (`D → nTargets_d`).
  - Stage 1 (pretrain): everything trainable; one batch per domain per step; loss = mean of the
    3 domains' standardized masked MSEs; best checkpoint by mean val loss.
  - Stage 2 (fine-tune): projections + transformer frozen; heads trained one domain at a time,
    each with its own optimizer and early stopping.

## Visual language
- Domain colors match `DOMAIN_COLOR`: Arctic `#CC79A7`, Amazon `#D55E00`, Rangeland `#0072B2`.
- Shared transformer: neutral gray.
- Frozen: same color, lightened. No extra "frozen" hue.
- Short straight arrows, no crossings, fonts matching the other figures.

## 2a layout
- Three independent columns (Arctic | Amazon | Rangeland), each in its domain color.
- Input `(t=1…T, n_features)` → Transformer → prediction at `T` (last step only; same-step
  emulator, not a forecast).

## 2b layout
- (a) Stage 1 | (b) Stage 2, linked by a dotted "Best checkpoint transferred" arrow.
- Each panel: 3 inputs → 3 projections (`ℝ^D`) → 1 shared transformer → 3 heads → outputs.
- Stage 2: projections + transformer grayed inside a "Freeze" box; heads stay full color with
  ①②③ order badges.
- What changes from 2a: the new projection boxes, and the transformer becomes one gray box.
