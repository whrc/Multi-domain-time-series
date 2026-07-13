"""
Makes all final publication figures for the main manuscript. All figures use flux-only
results. Figures should be in a standardized format: consistent font sizes, line widths, and
a colorblind-friendly scheme (reuses shared/plots.py's Okabe-Ito PALETTE); compact, tight
layouts with minimal whitespace; saved to ./figures/ at 300 dpi.

Figures 1 and 2 are methodological and are not produced by this script.

Figure 3: Arctic sampling density and dataset-size sweep, two panels, GPP/RECO averaged across
  SSP scenarios. Left: validation RMSE across capped sampling stride 50-500. Right: validation
  RMSE vs. training-set size at the best stride (400) and staggered windowing.

Figure 4: Individual domain model results, one row per domain (Arctic, Rangeland, Amazon),
  three metric columns per row (RMSE, NSE, PBIAS). Arctic's row further split by SSP scenario
  and historical/projected period; Rangeland's and Amazon's rows at the domain level.

Figure 5: Multi-domain training loss curves, two panels. Left: Stage 1 pretraining loss (train
  and validation), per domain and overall mean, vs. epoch. Right: Stage 2 fine-tuning loss
  (train and validation), one line per domain, vs. epoch.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.evaluate import scenario_period_label  # noqa: E402
from shared.plots import PALETTE, draw_metric_boxplot_panel  # noqa: E402

FIGURES_DIR = Path("figures")
DPI = 300

ARCTIC_MODELS_DIR = Path("outputs/arctic_domain/models")
ARCTIC_FLUXONLY_TEST = Path("outputs/arctic_domain/evaluation/500K_s400_fluxonly/metrics_test.csv")
RANGELAND_FLUXONLY_TEST = Path("outputs/rangeland_domain/evaluation_fluxonly/metrics_test.csv")
AMAZON_TEST = Path("outputs/amazon_domain/evaluation/metrics_test.csv")
MD_EVAL_DIR = Path("outputs/multi_domain/evaluation")

STRIDES = [100, 150, 200, 250, 300, 350, 400, 500]
SSP_LABELS = {"ssp1_2_6_mri_esm2_0": "SSP1-2.6", "ssp5_8_5_mri_esm2_0": "SSP5-8.5"}
FLUX_TARGETS = ["GPP", "RECO"]
METRICS_3COL = ["RMSE", "NSE", "PBIAS"]
DOMAINS = ["arctic", "amazon", "rangeland"]

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
        df = pd.read_csv(ARCTIC_FLUXONLY_TEST)
        df = df[~df["obs_degenerate"]]
        return df[["target", metric]]
    if domain == "rangeland":
        df = pd.read_csv(RANGELAND_FLUXONLY_TEST)
        df = df.assign(target=df["target"].str.replace("_predicted", "", regex=False))
        return df[["target", metric]]
    df = pd.read_csv(AMAZON_TEST)  # amazon: no flux-only variant, no normalization needed
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
    ax2.set_xticks([50_000, 250_000, 500_000])
    ax2.set_xticklabels(["50K", "250K", "500K"])
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
    """3 rows (Arctic, Rangeland, Amazon) x 3 metric columns (RMSE, NSE, PBIAS)."""
    arctic = pd.read_csv(ARCTIC_FLUXONLY_TEST)
    arctic = arctic[~arctic["obs_degenerate"]].copy()
    arctic["group"] = [scenario_period_label(s, p) for s, p in zip(arctic["ssp"], arctic["period"])]

    rangeland = pd.read_csv(RANGELAND_FLUXONLY_TEST)
    rangeland["target"] = rangeland["target"].str.replace("_predicted", "", regex=False)

    amazon = pd.read_csv(AMAZON_TEST)
    amazon["target"] = amazon["target"].map(AMAZON_TARGET_LABELS)

    rows = [
        ("Arctic", arctic, "group"),
        ("Rangeland", rangeland, None),
        ("Amazon", amazon, None),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.0))
    for ri, (domain_name, df, group_col) in enumerate(rows):
        for ci, metric in enumerate(METRICS_3COL):
            ax = axes[ri, ci]
            draw_metric_boxplot_panel(ax, df, metric, group_col=group_col)
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
                # from the patches draw_metric_boxplot_panel labeled via bp["boxes"][0], embedded
                # inside the RMSE panel instead of outside the axes.
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
                if patch_handles:
                    labels = [ARCTIC_GROUP_LABELS.get(p.get_label(), p.get_label()) for p in patch_handles]
                    leg = ax.legend(patch_handles, labels, loc="upper right", fontsize=6,
                                     frameon=True, framealpha=0.7)
                    leg.set_zorder(10)

    fig.tight_layout()
    _save(fig, "fig4_individual_domain_results.png")


def figure5_training_curves() -> None:
    """Single panel: Stage 1 (joint pretraining) loss curves per domain, followed by Stage 2
    (per-domain fine-tuning) curves picking up from the pretrain checkpoint each domain's
    fine-tuning actually started from (the best, not necessarily last-plotted, pretrain epoch —
    see domains/multi_domain/02_train.py's checkpoint-on-improvement logic)."""
    pretrain = pd.read_csv(MD_EVAL_DIR / "pretrained_fluxonly" / "history.csv")
    # .round(4) in history.csv means several trailing epochs can tie at the same displayed
    # minimum; take the last of those ties so the divider lines up with where the plotted
    # curve actually flattens, not the first epoch that happened to round the same way.
    best_val_mean = pretrain["val_mean"].min()
    stage1_end = pretrain.loc[pretrain["val_mean"] == best_val_mean, "epoch"].max()

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    max_x = stage1_end
    for d in DOMAINS:
        hist = pd.read_csv(MD_EVAL_DIR / "finetuned_fluxonly" / d / "history.csv")
        ft_x = stage1_end + hist["epoch"]
        x_all = pd.concat([pretrain["epoch"], ft_x])
        train_all = pd.concat([pretrain[f"train_{d}"], hist["train_loss"]])
        val_all = pd.concat([pretrain[f"val_{d}"], hist["val_loss"]])
        ax.plot(x_all, train_all, color=DOMAIN_COLOR[d], linestyle="-", label=f"{d.capitalize()} train")
        ax.plot(x_all, val_all, color=DOMAIN_COLOR[d], linestyle="--", label=f"{d.capitalize()} val")
        max_x = max(max_x, x_all.max())

    ax.axvline(stage1_end, color="black", linewidth=1.2, linestyle=":")
    # Centered within each stage's own epoch range, well clear of the divider on both sides.
    # Bold title line + plain detail line as two stacked text objects -- fontweight="bold"
    # applies per-Text-object, so a single call can't mix weights within itself.
    label_x1 = stage1_end / 2
    label_x2 = (stage1_end + max_x) / 2
    for x, title, detail in [
        (label_x1, "Stage 1: Joint pretraining", "(shared architecture for all domains)"),
        (label_x2, "Stage 2: Per-domain fine-tuning", "(MLP head-only, frozen backbone)"),
    ]:
        ax.text(x, 0.93, title, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, fontweight="bold")
        ax.text(x, 0.86, detail, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    _add_grid(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, fancybox=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=6)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, "fig5_multidomain_training_curves.png")


MODEL_ORDER = ["Individual", "Pretrained", "Fine-tuned"]


FIG6_METRICS = {"RMSE": "a", "NSE": "b", "PBIAS": "c"}
ROW_LETTERS_6 = {"arctic": "(a)", "amazon": "(b)", "rangeland": "(c)"}


def figure6_model_comparison() -> None:
    """One figure per metric (fig6a=RMSE, fig6b=NSE, fig6c=PBIAS), each with 3 rows (Arctic,
    Amazon, Rangeland), a single boxplot per row grouped by target, 3 boxes per target
    (Individual / Pretrained / Fine-tuned) — domain-level only, no PFT/SSP sub-grouping."""
    for metric, suffix in FIG6_METRICS.items():
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.0))

        for ax, domain in zip(axes, DOMAINS):
            individual = _normalize_individual(domain, metric).assign(model="Individual")
            pretrained = pd.read_csv(MD_EVAL_DIR / "pretrained_fluxonly" / domain / f"{domain}_metrics.csv")[
                ["target", metric]].assign(model="Pretrained")
            finetuned = pd.read_csv(MD_EVAL_DIR / "finetuned_fluxonly" / domain / f"{domain}_metrics.csv")[
                ["target", metric]].assign(model="Fine-tuned")
            combined = pd.concat([individual, pretrained, finetuned], ignore_index=True)
            if domain == "amazon":
                combined["target"] = combined["target"].map(AMAZON_TARGET_LABELS)
            # draw_metric_boxplot_panel groups via `sorted(unique())`; an ordered Categorical makes
            # that read Individual -> Pretrained -> Fine-tuned instead of alphabetical.
            combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
            draw_metric_boxplot_panel(ax, combined, metric, group_col="model")
            ax.set_title(f"{ROW_LETTERS_6[domain]} {domain.capitalize()}")
            ax.set_ylabel(metric)
            _add_grid(ax)
            _horizontal_xticks(ax)
            if domain == "amazon" and metric == "RMSE":
                # Individual Amazon's RMSE is ~1000x Pretrained/Fine-tuned's (see key_findings_log
                # MD-prod0712) — log scale so all three are visible, floor cuts off the
                # meaninglessly-small tail of the range.
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-2)

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        patch_handles = [p for p in axes[0].patches if p.get_label() and not p.get_label().startswith("_")]
        if patch_handles:
            fig.legend(patch_handles, [p.get_label() for p in patch_handles],
                       loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))

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
