"""
Makes all final publication figures for the main manuscript. All figures use flux-only
results, averaged across the 5 final-run seeds (see shared/seed_aggregation.py) -- not any
single seed, and per-seed std is not plotted (deferred, per project decision). Figures should
be in a standardized format: consistent font sizes, line widths, and a colorblind-friendly
scheme (reuses shared/plots.py's Okabe-Ito PALETTE); compact, tight layouts with minimal
whitespace; saved to ./figures/ at 300 dpi.

Figures 1 and 2 are methodological and are not produced by this script.

Figure 1 (not yet designed): all three domains' study sites on a map, colored by train/val/test
  split -- Arctic grids, Amazon gauging stations, Rangeland sites. To be revisited.

Figure 2 (2a/2b): model methodology sketches -- schematics, not data-driven, so built by a
  separate script, figures/scripts/make_figure2_schematics.py (design spec at
  figures/scripts/figure2_methodology_sketch_spec.md), not this one.

Figure 3: Arctic sampling density and dataset-size sweep, two panels, GPP/RECO averaged across
  SSP scenarios. Left: validation RMSE across capped sampling stride 50-500. Right: validation
  RMSE vs. training-set size at the best stride (400) and staggered windowing.

Figure 4: Individual domain model results, one row per domain (Arctic, Rangeland, Amazon),
  three metric columns per row (RMSE, NSE, PBIAS). Arctic's row further split by SSP scenario
  and historical/projected period; Rangeland's and Amazon's rows at the domain level.

Figure 5: Per-seed training-curve figures, two files -- fig5a = training loss, fig5b =
  validation loss (kept separate, not combined into one busy panel) -- each with two rows:
  (a) individual domain models' own 5-seed curves, (b) multi-domain model's pretrain-then-
  fine-tune 5-seed curves per domain, with a star marking each seed's own pretrain-to-
  fine-tune transition (seeds early-stop pretraining at different epochs, so there's no
  single shared transition point to draw as a vertical divider).

Figure 6: Individual, pretrained, and fine-tuned model comparison, one file per metric (fig6a =
  RMSE, fig6b = NSE, fig6c = PBIAS), each with one row per domain (Arctic, Amazon, Rangeland),
  domain-level only. Within each row, held-out test values grouped by target, three boxes per
  target (Individual, Pretrained, Fine-tuned).
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# figures/scripts/make_remaining_figures.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))
from shared.evaluate import scenario_period_label  # noqa: E402
from shared.plots import PALETTE, draw_metric_boxplot_panel  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
DPI = 300

# Final figures use the 5-seed average (seedavg) results, not any single seed -- see
# project_management/current_project_status.md and shared/seed_aggregation.py. Per-seed
# variance (std) is deliberately not plotted yet; _load_seedavg drops it.
ARCTIC_MODELS_DIR = REPO_ROOT / "outputs/arctic_domain/models"
ARCTIC_FLUXONLY_TEST = REPO_ROOT / "outputs/arctic_domain/evaluation/500K_s400_fluxonly_seedavg/metrics_test_seedavg.csv"
RANGELAND_FLUXONLY_TEST = REPO_ROOT / "outputs/rangeland_domain/evaluation_fluxonly_seedavg/metrics_test_seedavg.csv"
AMAZON_TEST = REPO_ROOT / "outputs/amazon_domain/evaluation_seedavg/metrics_test_seedavg.csv"
MD_EVAL_DIR = REPO_ROOT / "outputs/multi_domain/evaluation"
MD_PRETRAINED_SEEDAVG = MD_EVAL_DIR / "pretrained_fluxonly_seedavg"
MD_FINETUNED_SEEDAVG = MD_EVAL_DIR / "finetuned_fluxonly_seedavg"
SEEDS = [1, 2, 3, 4, 5]


def _load_seedavg(path: Path) -> pd.DataFrame:
    """Load a seedavg metrics CSV, renaming '{metric}_mean' columns back to the plain metric
    name so downstream code (written for single-seed metrics_test.csv) works unchanged.
    Per-seed std/n_seeds columns are dropped -- not plotted yet (see module-level note)."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: c[:-len("_mean")] for c in df.columns if c.endswith("_mean")})
    drop_cols = [c for c in df.columns if c.endswith("_std")] + ["n_seeds"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


STRIDES = [100, 150, 200, 250, 300, 350, 400, 500]
SSP_LABELS = {"ssp1_2_6_mri_esm2_0": "SSP1-2.6", "ssp5_8_5_mri_esm2_0": "SSP5-8.5"}
FLUX_TARGETS = ["GPP", "RECO"]
METRICS_3COL = ["RMSE", "NSE", "PBIAS"]
DOMAINS = ["arctic", "amazon", "rangeland"]
RANGELAND_DAY_TO_MONTH = 30  # display-only rescale of Rangeland's per-day RMSE to per-month,
                             # so it's magnitude-comparable to Arctic's native per-month
                             # fluxes in Figures 4 and 6 -- see figure4/figure6's own comments

AMAZON_TARGET_LABELS = {
    "active_fire_count": "Active fire\ncount",
    "burned_area": "Burned area",
    "discharge": "Discharge",
}
ARCTIC_GROUP_LABELS = {
    "historical": "Historical",
    "projected_ssp126": "Projected SSP1-2.6",
    "projected_ssp585": "Projected SSP5-8.5",
}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.7,
        "boxplot.boxprops.linewidth": 0.8,
        "boxplot.whiskerprops.linewidth": 0.8,
        "boxplot.capprops.linewidth": 0.8,
        "boxplot.medianprops.linewidth": 1.0,
    })


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _add_grid(ax: plt.Axes) -> None:
    """Sparse, low-alpha dotted gridlines on an axis's existing (major) ticks."""
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def _horizontal_xticks(ax: plt.Axes) -> None:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")


def _normalize_individual(domain: str, metric: str) -> pd.DataFrame:
    """Load an individual domain's flux-only test-set metric, normalized to a plain
    {target, metric} frame (drops Arctic's degenerate rows / period column, Rangeland's
    '_predicted' target-name suffix — neither matches multi-domain's own conventions)."""
    if domain == "arctic":
        df = _load_seedavg(ARCTIC_FLUXONLY_TEST)
        df = df[~df["obs_degenerate"]]
        return df[["target", metric]]
    if domain == "rangeland":
        df = _load_seedavg(RANGELAND_FLUXONLY_TEST)
        df = df.assign(target=df["target"].str.replace("_predicted", "", regex=False))
        return df[["target", metric]]
    df = _load_seedavg(AMAZON_TEST)  # amazon: no flux-only variant, no normalization needed
    return df[["target", metric]]


def figure3_arctic_sweep() -> None:
    """(a) val RMSE for GPP/RECO across capped stride 50-500 (50K windows), averaged across
    SSP scenarios. (b) val RMSE vs. train-set size (50K vs. 500K) at the winning stride=400,
    staggered windowing, also averaged across SSP scenarios."""
    sweep_rows = []
    for s in STRIDES:
        df = pd.read_csv(ARCTIC_MODELS_DIR / f"val_metrics_50K_s{s}.csv")
        df = df[df["target"].isin(FLUX_TARGETS)].copy()
        df["stride"] = s
        sweep_rows.append(df)
    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep_avg = sweep.groupby(["target", "stride"], as_index=False)["RMSE"].mean()

    size = pd.concat([
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_10K_s400_fluxonly.csv").assign(x=10_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_50K_s400.csv").assign(x=50_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_250K_s400_fluxonly.csv").assign(x=250_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_500K_s400.csv").assign(x=500_000),
    ], ignore_index=True)
    size = size[size["target"].isin(FLUX_TARGETS)]
    size_avg = size.groupby(["target", "x"], as_index=False)["RMSE"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    for ti, target in enumerate(FLUX_TARGETS):
        sub = sweep_avg[sweep_avg["target"] == target].sort_values("stride")
        ax1.plot(sub["stride"], sub["RMSE"], marker="o", color=PALETTE[ti], label=target)
    ax1.set_xlabel("Sampling stride")
    ax1.set_ylabel("Validation RMSE")
    ax1.set_title("(a) Sampling-density sweep (50K windows)")
    ax1.legend(frameon=False)
    _add_grid(ax1)

    for ti, target in enumerate(FLUX_TARGETS):
        sub = size_avg[size_avg["target"] == target].sort_values("x")
        ax2.plot(sub["x"], sub["RMSE"], marker="o", color=PALETTE[ti], label=target)
    ax2.set_xscale("log")
    ax2.minorticks_off()
    ax2.set_xticks([10_000, 50_000, 250_000, 500_000])
    ax2.set_xticklabels(["10K", "50K", "250K", "500K"])
    ax2.set_xlabel("Training-set windows at stride=400")
    ax2.set_title("(b) Dataset-size scale-up")
    ax2.legend(frameon=False)
    _add_grid(ax2)

    fig.tight_layout()
    _save(fig, "fig3_arctic_sampling_sweep.png")


ROW_LETTERS_4 = ["(a)", "(b)", "(c)"]

# One domain->color mapping shared across figures. PALETTE[0:3] are reserved for grouped
# categories (Arctic's historical/SSP1-2.6/SSP5-8.5 in Figure 4, Individual/Pretrained/
# Fine-tuned in Figure 6), so domain identity colors start at PALETTE[4].
DOMAIN_COLOR = {"arctic": PALETTE[6], "amazon": PALETTE[5], "rangeland": PALETTE[4]}

# Rangeland and Amazon are single ungrouped boxes (PALETTE[0] by default); recolor them so no
# domain repeats Arctic's group colors (PALETTE[0:3], one per historical/projected sub-group).
DOMAIN_BOX_COLOR = {"Rangeland": DOMAIN_COLOR["rangeland"], "Amazon": DOMAIN_COLOR["amazon"]}


def figure4_individual_domain_results() -> None:
    """3 rows (Arctic, Amazon, Rangeland) x 3 metric columns (RMSE, NSE, PBIAS)."""
    arctic = _load_seedavg(ARCTIC_FLUXONLY_TEST)
    arctic = arctic[~arctic["obs_degenerate"]].copy()
    arctic["group"] = [scenario_period_label(s, p) for s, p in zip(arctic["ssp"], arctic["period"])]

    rangeland = _load_seedavg(RANGELAND_FLUXONLY_TEST)
    rangeland["target"] = rangeland["target"].str.replace("_predicted", "", regex=False)
    # Rangeland fluxes are stored per-day; rescale RMSE to per-month (x30) for this figure
    # only, so its magnitude is visually comparable to Arctic's native per-month fluxes
    # instead of looking artificially tiny next to them. Display-only -- the underlying
    # per-day data/outputs are untouched. NSE/PBIAS are scale-invariant (unaffected by a
    # constant multiplier on both obs and pred), so only RMSE needs it.
    rangeland["RMSE"] = rangeland["RMSE"] * RANGELAND_DAY_TO_MONTH

    amazon = _load_seedavg(AMAZON_TEST)
    amazon["target"] = amazon["target"].map(AMAZON_TARGET_LABELS)

    rows = [
        ("Arctic", arctic, "group"),
        ("Amazon", amazon, None),
        ("Rangeland", rangeland, None),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.0))
    arctic_patch_handles: list = []
    for ri, (domain_name, df, group_col) in enumerate(rows):
        for ci, metric in enumerate(METRICS_3COL):
            ax = axes[ri, ci]
            draw_metric_boxplot_panel(ax, df, metric, group_col=group_col, box_width_frac=0.6)
            if ci == 0:
                ax.set_ylabel(f"{ROW_LETTERS_4[ri]} {domain_name}", fontsize=8, fontweight="bold")
            ax.set_title(metric if ri == 0 else "")
            _add_grid(ax)
            _horizontal_xticks(ax)

            if domain_name in DOMAIN_BOX_COLOR:
                for p in ax.patches:
                    p.set_facecolor(DOMAIN_BOX_COLOR[domain_name])

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if group_col and ci == 0:
                # boxplot legend handles aren't tracked by get_legend_handles_labels(); rebuild
                # from the patches draw_metric_boxplot_panel labeled via bp["boxes"][0]. Kept for
                # a figure-level legend below the Arctic row instead of embedding it in this axes.
                arctic_patch_handles = [p for p in ax.patches
                                         if p.get_label() and not p.get_label().startswith("_")]

    fig.tight_layout()
    if arctic_patch_handles:
        labels = [ARCTIC_GROUP_LABELS.get(p.get_label(), p.get_label()) for p in arctic_patch_handles]
        fig.subplots_adjust(hspace=0.55)
        row0_bottom = axes[0, 0].get_position().y0
        row1_top = axes[1, 0].get_position().y1
        fig.legend(arctic_patch_handles, labels, loc="center",
                   bbox_to_anchor=(0.5, (row0_bottom + row1_top) / 2),
                   ncol=3, frameon=True, fancybox=False, fontsize=6)
    _save(fig, "fig4_individual_domain_results.png")


# Per-seed history.csv location for each standalone individual-domain model -- mirrors each
# domain's own naming convention (Arctic: labeled subfolder; Rangeland/Amazon: suffixed
# sibling dir), same as ARCTIC_FLUXONLY_TEST/RANGELAND_FLUXONLY_TEST/AMAZON_TEST above.
INDIVIDUAL_HISTORY_PATH = {
    "arctic": lambda s: REPO_ROOT / f"outputs/arctic_domain/evaluation/500K_s400_fluxonly_seed{s}/history.csv",
    "rangeland": lambda s: REPO_ROOT / f"outputs/rangeland_domain/evaluation_fluxonly_seed{s}/history.csv",
    "amazon": lambda s: REPO_ROOT / f"outputs/amazon_domain/evaluation_seed{s}/history.csv",
}


def _plot_training_curves(loss_kind: str) -> plt.Figure:
    """Two panels, one line per seed per domain in each (15 lines/panel; loss_kind selects
    'train' or 'val' -- the two are kept in separate figures rather than one busy panel):
    (a) standalone individual-domain models, each seed's own single training run.
    (b) multi-domain model: Stage 1 (joint pretraining) directly followed by that same
        seed's Stage 2 (per-domain fine-tuning), each seed's own pretrain segment ending
        exactly where its own history.csv ends (early stopping fires at a different epoch
        per seed, so there's no single shared pretrain/fine-tune boundary -- a star marks
        each seed's own transition instead of a shared vertical divider, which would
        misrepresent every seed but the one it happened to line up with)."""
    loss_col = f"{loss_kind}_loss"  # individual + finetune history.csv column name

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [0.8, 1.2]})

    for d in DOMAINS:
        for i, s in enumerate(SEEDS):
            hist = pd.read_csv(INDIVIDUAL_HISTORY_PATH[d](s)).dropna(subset=[loss_col])
            # dropna is required, not cosmetic: eval_every_n_epochs=2 for Arctic means
            # val_loss is NaN on every other row, so with the NaN rows left in, no two
            # consecutive points are ever both valid and matplotlib silently draws nothing.
            ax1.plot(hist["epoch"], hist[loss_col], color=DOMAIN_COLOR[d], alpha=0.55,
                     linewidth=1.0, label=d.capitalize() if i == 0 else None)

    for d in DOMAINS:
        for s in SEEDS:
            pretrain = pd.read_csv(MD_EVAL_DIR / f"pretrained_fluxonly_seed{s}" / "history.csv")
            stage1_end = pretrain["epoch"].max()  # last logged epoch = this seed's own stop point
            ft = pd.read_csv(MD_EVAL_DIR / f"finetuned_fluxonly_seed{s}" / d / "history.csv")
            x = pd.concat([pretrain["epoch"], stage1_end + ft["epoch"]])
            y = pd.concat([pretrain[f"{loss_kind}_{d}"], ft[loss_col]])
            ax2.plot(x, y, color=DOMAIN_COLOR[d], alpha=0.55, linewidth=1.0)
            ax2.plot(stage1_end, pretrain[f"{loss_kind}_{d}"].iloc[-1], marker="*", markersize=7,
                     color=DOMAIN_COLOR[d], markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    # Fixed relative positions (not tied to any seed's actual epoch, since seeds stop at
    # different epochs -- see docstring) -- purely descriptive labels, not a boundary line.
    ax2.text(0.25, 0.93, "Stage 1: Joint pretraining", transform=ax2.transAxes, ha="center",
             va="top", fontsize=6.5, fontweight="bold")
    ax2.text(0.75, 0.93, "Stage 2: Per-domain fine-tuning", transform=ax2.transAxes, ha="center",
             va="top", fontsize=6.5, fontweight="bold")

    loss_label = "Training" if loss_kind == "train" else "Validation"
    ax1.set_title("(a) Individual domain models", loc="left", fontsize=8, fontweight="bold")
    ax2.set_title("(b) Multi-domain model", loc="left", fontsize=8, fontweight="bold")
    ax1.set_ylabel(f"{loss_label} MSE Loss")
    ax2.set_ylabel(f"{loss_label} MSE Loss")
    ax2.set_xlabel("Epoch")
    _add_grid(ax1)
    _add_grid(ax2)

    handles, labels = ax1.get_legend_handles_labels()
    star_handle = plt.Line2D([], [], marker="*", markersize=7, color="grey",
                             markeredgecolor="black", markeredgewidth=0.4, linestyle="None")
    # In panel (a)'s own empty upper-right space, not a separate figure-level legend below --
    # panel (a)'s curves flatten out well before the right edge, leaving room.
    ax1.legend(handles + [star_handle], labels + ["Pretraining stopped"], loc="upper right",
              frameon=True, fancybox=False, fontsize=6)
    fig.tight_layout()
    return fig


def figure5_training_curves() -> None:
    _save(_plot_training_curves("train"), "fig5a_multidomain_training_curves_train.png")
    _save(_plot_training_curves("val"), "fig5b_multidomain_training_curves_val.png")


MODEL_ORDER = ["Individual", "Pretrained", "Fine-tuned"]


FIG6_METRICS = {"RMSE": "a", "NSE": "b", "PBIAS": "c"}
ROW_LETTERS_6 = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}


def _domain_combined(domain: str, metric: str) -> pd.DataFrame:
    """Individual/Pretrained/Fine-tuned rows for one domain, one metric, as a single
    {target, metric, model} frame -- shared by figure6_model_comparison() below and by
    make_figure6a2.py's per-target variant, so both figures rescale Rangeland the same way.
    Amazon's target names are left as raw snake_case here (not remapped to a display label):
    the two figures want different label formatting (this one's narrow panels need the
    wrapped "Active fire\\ncount"; make_figure6a2.py's wider panels use a single-line
    version) -- each caller applies its own AMAZON_TARGET_LABELS after calling this."""
    individual = _normalize_individual(domain, metric).assign(model="Individual")
    pretrained = _load_seedavg(MD_PRETRAINED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Pretrained")
    finetuned = _load_seedavg(MD_FINETUNED_SEEDAVG / domain / f"{domain}_metrics_seedavg.csv")[
        ["target", metric]].assign(model="Fine-tuned")
    combined = pd.concat([individual, pretrained, finetuned], ignore_index=True)
    if domain == "rangeland" and metric == "RMSE":
        # Same display-only per-day -> per-month rescale as figure4_individual_domain_results
        # (see RANGELAND_DAY_TO_MONTH) -- applied once here so it covers all three model
        # sources (individual/pretrained/finetuned) consistently.
        combined[metric] = combined[metric] * RANGELAND_DAY_TO_MONTH
    # draw_metric_boxplot_panel groups via `sorted(unique())`; an ordered Categorical makes
    # that read Individual -> Pretrained -> Fine-tuned instead of alphabetical.
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    return combined


def figure6_model_comparison() -> None:
    """One figure per metric (fig6a=RMSE, fig6b=NSE, fig6c=PBIAS), each with 3 rows (Arctic,
    Amazon, Rangeland), a single boxplot per row grouped by target, 3 boxes per target
    (Individual / Pretrained / Fine-tuned) — domain-level only, no PFT/SSP sub-grouping."""
    for metric, suffix in FIG6_METRICS.items():
        fig, axes = plt.subplots(3, 1, figsize=(5.0, 5.5))

        for ax, domain in zip(axes, DOMAINS):
            combined = _domain_combined(domain, metric)
            if domain == "amazon":
                combined["target"] = combined["target"].map(AMAZON_TARGET_LABELS)
            draw_metric_boxplot_panel(ax, combined, metric, group_col="model",
                                       box_width_frac=0.75, group_span=0.7)
            ax.set_title(f"{ROW_LETTERS_6[domain]} {domain.capitalize()}")
            ax.set_ylabel(metric)
            _add_grid(ax)
            _horizontal_xticks(ax)

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        patch_handles = [p for p in axes[0].patches if p.get_label() and not p.get_label().startswith("_")]
        if patch_handles:
            fig.legend(patch_handles, [p.get_label() for p in patch_handles],
                       loc="lower center", ncol=3, frameon=True, fancybox=False,
                       bbox_to_anchor=(0.5, 0.0))

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        _save(fig, f"fig6{suffix}_individual_pretrained_finetuned_comparison_{metric.lower()}.png")


def main() -> None:
    _style()
    figure3_arctic_sweep()
    figure4_individual_domain_results()
    figure5_training_curves()
    figure6_model_comparison()


if __name__ == "__main__":
    main()
