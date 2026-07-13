# Figure 2 design spec: model methodology sketches

Design spec for two hand/matplotlib-built schematic figures (not data-driven — see
`make_remaining_figures.py`'s module docstring). **Figure 2a**: the individual per-domain model.
**Figure 2b**: the unified two-stage multi-domain model. They are a deliberate before/after
pair — 2a establishes the baseline with zero parameter sharing, 2b shows exactly what becomes
shared and how training proceeds in two stages.

Architecture facts below are confirmed directly from `domains/multi_domain/model.py` and
`domains/multi_domain/02_train.py`, not guessed.

## Architecture reference

**Individual (per-domain) models — Figure 2a's subject:** each domain trains a fully separate
`TransformerModel` (`shared/transformer.py`) — raw input features go directly into that
domain's own transformer (no separate projection layer; the transformer's own internal
input/output projections are sized to that domain's native feature/target dimensions). No
weights shared across domains; each domain can even use a different architecture size. Loss:
per-domain masked-MSE between prediction and observation, both in that domain's own
standardized (z-scored) space.

**`MultiDomainModel` — Figure 2b's subject** (`domains/multi_domain/model.py`): per-domain
`nn.Linear` projection (`nFeatures_d → common_dim`, i.e. every domain is projected into the
*same* shared dimension D) → one shared transformer encoder (`common_dim → common_dim`) →
per-domain MLP head (`common_dim → nTargets_d`). This projection step is the key new element
vs. 2a — it's what makes sharing one transformer across three domains with different native
feature counts possible at all.

- **Stage 1 (pretrain):** every parameter trainable — projections, shared transformer, all 3
  heads updated jointly every step. Mixed-domain batching: each optimizer step consumes one
  batch from *each* domain; loss = mean of the 3 domains' masked-MSE losses, each computed in
  that domain's own standardized/z-scored space — this standardization is what makes averaging
  losses across domains with wildly different native units (GPP vs. discharge vs. biomass)
  mathematically meaningful. Best checkpoint saved by mean val loss.
- **Stage 2 (finetune):** transformer + projections frozen (loaded from Stage 1's best
  checkpoint); only each domain's own head trains, and domains are finetuned sequentially, one
  at a time (own optimizer, own early-stopping per domain) — not simultaneously like Stage 1.

## Shared visual language (defined once, used by both 2a and 2b)

**Color — minimal, purposeful, reuses the project's existing Okabe-Ito palette exactly (no new
hues introduced):**

- Arctic → `PALETTE[6]` reddish purple `#CC79A7`
- Amazon → `PALETTE[5]` vermilion `#D55E00`
- Rangeland → `PALETTE[4]` blue `#0072B2`

(identical to `DOMAIN_COLOR` in `make_remaining_figures.py`, for continuity with Figures 4-6.)

- **Shared/neutral component** (only exists in 2b — the shared transformer): a neutral slate
  gray, not a PALETTE hue at all, since it isn't a domain — its entire visual point is *not*
  being one of the three domain colors.
- **No separate color for "frozen."** Reuses each element's own color at reduced
  saturation/lightness instead of introducing a 4th "frozen" hue: full-saturation domain color
  = trainable right now; a lightened/desaturated version of the *same* color = frozen. The
  shared transformer (already neutral gray) gets a darker gray when trainable, a light
  hatched/washed-out gray when frozen. This is the entire color vocabulary: 3 domain hues + 1
  neutral, each with a trainable/frozen lightness variant — nothing added on a whim.

**Box/line style:** boxes sized to text with small uniform padding (no oversized placeholders);
short straight arrows, no crossing lines; one shared legend (domain colors + trainable/frozen
lightness key) placed once, not repeated per panel; consistent font sizes with the rest of the
figure set (`_style()`'s 8-9pt).

## Figure 2a: individual per-domain models

**Layout:** 3 fully independent parallel vertical stacks, side by side — Arctic | Amazon |
Rangeland — no shared box anywhere, no convergence point. The point of this figure is to show
*zero* sharing, setting up 2b's contrast.

```
┌── Arctic ──────┐   ┌── Amazon ──────┐   ┌── Rangeland ───┐
│Input:          │   │Input:          │   │Input:          │
│(t=1...T,       │   │(t=1...T,       │   │(t=1...T,       │
│ nFeatures_d)   │   │ nFeatures_d)   │   │ nFeatures_d)   │
│      │         │   │      │         │   │      │         │
│ [Transformer]  │   │ [Transformer]  │   │ [Transformer]  │  <- each domain-colored (own
│      │         │   │      │         │   │      │         │     weights, possibly own
│  (sequence out,│   │  (sequence out,│   │  (sequence out,│     architecture size)
│   last step    │   │   last step    │   │   last step    │
│   selected) ↓  │   │   selected) ↓  │   │   selected) ↓  │
│ Output @ T:    │   │ Output @ T:    │   │ Output @ T:    │
│ GPP, RECO      │   │ discharge,     │   │ GPP,RECO,      │
│                │   │ fire, burn     │   │ Rm,Rg          │
└────────────────┘   └────────────────┘   └────────────────┘
Loss = masked-MSE(pred, obs) in standardized space, independently per domain.
```

Each column entirely in its own domain color (input box, transformer box, output labels) —
since nothing is shared, there's no reason to break from domain coloring here. This sets up 2b's
transformer box turning gray as *the* visual signal that sharing was introduced. The "sequence
out → last step selected" annotation makes the causal same-step-emulator framing explicit: the
transformer processes a window of steps `1...T` but only the final position `T`'s prediction is
read out and scored — not a forecast, not sequence-to-sequence.

## Figure 2b: unified two-stage model

**Layout:** two side-by-side sub-panels, (a) Stage 1 and (b) Stage 2, mirrored vertical flow,
connected by one labeled dashed arrow (checkpoint transfer).

```
┌──────────── (a) Stage 1: Joint Pretraining ─────────────┐  ┌──────────── (b) Stage 2: Per-Domain Fine-tuning ─────────────┐
│ Input: (t=1...T, nFeatures_d) per domain                 │  │ Input: (t=1...T, nFeatures_d) per domain                    │
│  [Arctic in] [Amazon in] [Rangeland in]                   │  │  [Arctic in] [Amazon in] [Rangeland in]                     │
│       │           │            │                          │  │       │           │            │                         │
│  [Proj→D]     [Proj→D]     [Proj→D]      (trainable,      │  │  [Proj→D]     [Proj→D]     [Proj→D]   (FROZEN -- light/    │
│       │           │            │          full domain      │  │       │           │            │       desaturated domain │
│       └───────────┼────────────┘          color)           │  │       └───────────┼────────────┘       color)            │
│           [Shared Transformer]            (trainable,      │  │           [Shared Transformer]         (FROZEN -- light   │
│                   │                        dark gray)       │  │                   │                    gray)             │
│       ┌───────────┼────────────┐                          │  │       ┌───────────┼────────────┐                         │
│  [Head: Arctic][Head: Amazon][Head: Rangeland]  (trainable, │  │  [Head: Arctic][Head: Amazon][Head: Rangeland] (trainable,│
│       │           │            │            full color)    │  │       │           │            │        ①②③ sequential)  │
│  (sequence out, last step selected) ↓                      │  │  (sequence out, last step selected) ↓                     │
│ Output @ T: GPP,RECO | discharge,fire,burn | GPP,RECO,Rm,Rg │  │ Output @ T: GPP,RECO | discharge,fire,burn | GPP,RECO,Rm,Rg│
│                                                              │  │                                                          │
│ All domains projected to common dimension D (common_dim)   │  │ Each head trained independently, sequentially (①→②→③);    │
│ Loss = mean(masked-MSE, 3 domains), each in that domain's  │  │ own optimizer + own early-stopping criterion per domain    │
│ own standardized/z-scored space; joint update every step   │  │                                                          │
│ (mixed-domain batching)                                     │  │                                                          │
└──────────────────────────────────────────────────────────┘  └──────────────────────────────────────────────────────────┘
                            └──────────── "best checkpoint" (transformer + projections) ────────────┘
                                       (dashed connector, Stage 1's transformer box -> Stage 2's)
```

**Element-by-element (delta from 2a):**

- Input/output framing matches 2a exactly (`t=1...T` x `nFeatures_d` in, prediction at `T` out)
  — the causal same-step-emulator framing is identical, only what's shared changes between 2a
  and 2b.
- New projection boxes appear (didn't exist in 2a) — explicitly annotated "→ ℝ^D (common
  dimension)" so the reader sees why one transformer can now serve all three domains.
- Transformer box changes from "3 separate domain-colored boxes" (2a) to "1 neutral gray box"
  (2b) — this color change *is* the figure's core visual argument.
- Stage 1 → Stage 2: projections and transformer flip from full-color/dark-gray (trainable) to
  desaturated/light-gray (frozen); heads stay full-color in both (always trainable), gaining a
  small sequential-order badge in Stage 2 only.

## Implementation (next step, not this deliverable)

Build both as a matplotlib script (`matplotlib.patches` boxes/arrows), exporting SVG (vector,
editable in Illustrator/Inkscape afterward) alongside the usual 300dpi PNG, consistent with
every other figure in `make_remaining_figures.py`.
