"""
Makes all final publication figures for the main manuscript. All figures use flux-only
results. Figures should be in a standardized format: consistent font sizes, line widths, and
a colorblind-friendly scheme (reuses shared/plots.py's Okabe-Ito PALETTE); compact, tight
layouts with minimal whitespace; saved to ./figures/ at 300 dpi.

Figures 1 and 2 are methodological and are not produced by this script.

Figure 3: Arctic sampling density and dataset-size sweep, two panels.
  Left: validation RMSE for GPP and RECO across capped sampling stride 50-500.
  Right: validation RMSE vs. training-set size at the best stride (400) and staggered
  windowing.

Figure 4: Individual domain model results, one row per domain (Arctic, Rangeland, Amazon),
  three metric columns per row (RMSE, NSE, PBIAS). Arctic's row further split by SSP scenario
  and historical/projected period; Rangeland's row further split by plant functional type
  group; Amazon's row at the domain level.

Figure 5: Multi-domain training loss curves, two panels. Left: Stage 1 pretraining loss (train
  and validation), per domain and overall mean, vs. epoch. Right: Stage 2 fine-tuning loss
  (train and validation), one line per domain, vs. epoch.

Figure 6: Individual, pretrained, and fine-tuned model comparison, one row per domain (Arctic,
  Rangeland, Amazon), domain-level only. Within each row, held-out test RMSE grouped by
  target, three boxes per target (Individual, Pretrained, Fine-tuned).
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

STRIDES = [50, 100, 150, 200, 250, 300, 350, 400, 500]
SSP_LABELS = {"ssp1_2_6_mri_esm2_0": "SSP1-2.6", "ssp5_8_5_mri_esm2_0": "SSP5-8.5"}
FLUX_TARGETS = ["GPP", "RECO"]
METRICS_3COL = ["RMSE", "NSE", "PBIAS"]
DOMAINS = ["arctic", "amazon", "rangeland"]


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


def _target_ssp_color(ti: int, si: int) -> str:
    """Fixed color per (target, ssp) pair, independent of any runtime row ordering."""
    return PALETTE[(ti * len(SSP_LABELS) + si) % len(PALETTE)]


def _normalize_individual_rmse(domain: str) -> pd.DataFrame:
    """Load an individual domain's flux-only test-set RMSE, normalized to a plain
    {target, RMSE} frame (drops Arctic's degenerate rows / period column, Rangeland's
    '_predicted' target-name suffix — neither matches multi-domain's own conventions)."""
    if domain == "arctic":
        df = pd.read_csv(ARCTIC_FLUXONLY_TEST)
        df = df[~df["obs_degenerate"]]
        return df[["target", "RMSE"]]
    if domain == "rangeland":
        df = pd.read_csv(RANGELAND_FLUXONLY_TEST)
        df = df.assign(target=df["target"].str.replace("_predicted", "", regex=False))
        return df[["target", "RMSE"]]
    df = pd.read_csv(AMAZON_TEST)  # amazon: no flux-only variant, no normalization needed
    return df[["target", "RMSE"]]


def figure3_arctic_sweep() -> None:
    """(a) val RMSE for GPP/RECO across capped stride 50-500 (50K windows). (b) val RMSE vs.
    train-set size (50K vs. 500K) at the winning stride=400, staggered windowing."""
    sweep_rows = []
    for s in STRIDES:
        df = pd.read_csv(ARCTIC_MODELS_DIR / f"val_metrics_50K_s{s}.csv")
        df = df[df["target"].isin(FLUX_TARGETS)].copy()
        df["stride"] = s
        sweep_rows.append(df)
    sweep = pd.concat(sweep_rows, ignore_index=True)

    size = pd.concat([
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_50K_s400.csv").assign(x=50_000),
        pd.read_csv(ARCTIC_MODELS_DIR / "val_metrics_500K_s400.csv").assign(x=500_000),
    ], ignore_index=True)
    size = size[size["target"].isin(FLUX_TARGETS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for ti, target in enumerate(FLUX_TARGETS):
        for si, (ssp, ssp_label) in enumerate(SSP_LABELS.items()):
            sub = sweep[(sweep["target"] == target) & (sweep["ssp"] == ssp)].sort_values("stride")
            ax1.plot(sub["stride"], sub["RMSE"], marker="o", color=_target_ssp_color(ti, si),
                     label=f"{target} ({ssp_label})")
    ax1.set_xlabel("Capped sampling stride")
    ax1.set_ylabel("Validation RMSE")
    ax1.set_title("(a) Sampling-density sweep (50K windows)")
    ax1.legend(frameon=False)

    for ti, target in enumerate(FLUX_TARGETS):
        for si, (ssp, ssp_label) in enumerate(SSP_LABELS.items()):
            sub = size[(size["target"] == target) & (size["ssp"] == ssp)].sort_values("x")
            ax2.plot(sub["x"], sub["RMSE"], marker="o", color=_target_ssp_color(ti, si),
                     label=f"{target} ({ssp_label})")
    ax2.set_xscale("log")
    ax2.minorticks_off()
    ax2.set_xticks([50_000, 500_000])
    ax2.set_xticklabels(["50K", "500K"])
    ax2.set_xlabel("Training-set size (windows; stride=400, staggered)")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("(b) Dataset-size scale-up")
    ax2.legend(frameon=False)

    fig.tight_layout()
    _save(fig, "fig3_arctic_sampling_sweep.png")


def figure4_individual_domain_results() -> None:
    """3 rows (Arctic, Rangeland, Amazon) x 3 metric columns (RMSE, NSE, PBIAS)."""
    arctic = pd.read_csv(ARCTIC_FLUXONLY_TEST)
    arctic = arctic[~arctic["obs_degenerate"]].copy()
    arctic["group"] = [scenario_period_label(s, p) for s, p in zip(arctic["ssp"], arctic["period"])]

    rangeland = pd.read_csv(RANGELAND_FLUXONLY_TEST)
    rangeland["target"] = rangeland["target"].str.replace("_predicted", "", regex=False)

    amazon = pd.read_csv(AMAZON_TEST)

    rows = [
        ("Arctic", arctic, "group"),
        ("Rangeland", rangeland, "pft"),
        ("Amazon", amazon, None),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.5))
    for ri, (domain_name, df, group_col) in enumerate(rows):
        for ci, metric in enumerate(METRICS_3COL):
            ax = axes[ri, ci]
            draw_metric_boxplot_panel(ax, df, metric, group_col=group_col)
            if ci == 0:
                ax.set_ylabel(f"{domain_name}\n{metric}", fontsize=8, fontweight="bold")
            ax.set_title(metric if ri == 0 else "")
            # draw_metric_boxplot_panel puts a legend on every panel; keep just one per row
            # (rightmost column, moved outside the axes) instead of repeating/overlapping data.
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            if group_col and ci == len(METRICS_3COL) - 1:
                # boxplot legend handles aren't tracked by get_legend_handles_labels(); rebuild
                # from the patches draw_metric_boxplot_panel labeled via bp["boxes"][0].
                patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
                if patch_handles:
                    ax.legend(patch_handles, [p.get_label() for p in patch_handles],
                             loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6, title=group_col)

    fig.tight_layout()
    _save(fig, "fig4_individual_domain_results.png")


def figure5_training_curves() -> None:
    """(a) Stage 1 pretraining loss, train+val per domain + overall val mean, vs. epoch.
    (b) Stage 2 fine-tuning loss, train+val per domain, vs. epoch (x-axis = epoch within each
    domain's own sequential fine-tuning run)."""
    pretrain = pd.read_csv(MD_EVAL_DIR / "pretrained_fluxonly" / "history.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

    for i, d in enumerate(DOMAINS):
        ax1.plot(pretrain["epoch"], pretrain[f"train_{d}"], color=PALETTE[i], linestyle="-",
                 label=f"{d} train")
        ax1.plot(pretrain["epoch"], pretrain[f"val_{d}"], color=PALETTE[i], linestyle="--",
                 label=f"{d} val")
    ax1.plot(pretrain["epoch"], pretrain["val_mean"], color="black", linestyle=":", label="val mean")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (masked MSE, normalized units)")
    ax1.set_title("(a) Stage 1 — joint pretraining")
    ax1.legend(fontsize=6, ncol=2, frameon=False)

    for i, d in enumerate(DOMAINS):
        hist = pd.read_csv(MD_EVAL_DIR / "finetuned_fluxonly" / d / "history.csv")
        ax2.plot(hist["epoch"], hist["train_loss"], color=PALETTE[i], linestyle="-", label=f"{d} train")
        ax2.plot(hist["epoch"], hist["val_loss"], color=PALETTE[i], linestyle="--", label=f"{d} val")
    ax2.set_xlabel("Epoch (within each domain's own fine-tuning run)")
    ax2.set_ylabel("Loss (masked MSE, normalized units)")
    ax2.set_title("(b) Stage 2 — per-domain fine-tuning")
    ax2.legend(fontsize=6, ncol=2, frameon=False)

    fig.tight_layout()
    _save(fig, "fig5_multidomain_training_curves.png")


MODEL_ORDER = ["Individual", "Pretrained", "Fine-tuned"]


def figure6_model_comparison() -> None:
    """3 rows (Arctic, Rangeland, Amazon), each a single RMSE boxplot grouped by target, 3
    boxes per target (Individual / Pretrained / Fine-tuned) — domain-level only, no
    PFT/SSP sub-grouping."""
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5))

    for ri, (ax, domain) in enumerate(zip(axes, DOMAINS)):
        individual = _normalize_individual_rmse(domain).assign(model="Individual")
        pretrained = pd.read_csv(MD_EVAL_DIR / "pretrained_fluxonly" / domain / f"{domain}_metrics.csv")[
            ["target", "RMSE"]].assign(model="Pretrained")
        finetuned = pd.read_csv(MD_EVAL_DIR / "finetuned_fluxonly" / domain / f"{domain}_metrics.csv")[
            ["target", "RMSE"]].assign(model="Fine-tuned")
        combined = pd.concat([individual, pretrained, finetuned], ignore_index=True)
        # draw_metric_boxplot_panel groups via `sorted(unique())`; an ordered Categorical makes
        # that read Individual -> Pretrained -> Fine-tuned instead of alphabetical.
        combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
        draw_metric_boxplot_panel(ax, combined, "RMSE", group_col="model")
        ax.set_title(domain.capitalize())
        ax.set_ylabel("RMSE")
        if domain == "amazon":
            # Individual Amazon's RMSE is ~1000x Pretrained/Fine-tuned's (see key_findings_log
            # MD-prod0712) — log scale so all three are actually visible on one axis.
            ax.set_yscale("log")

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if ri == 0:
            patch_handles = [p for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
            if patch_handles:
                ax.legend(patch_handles, [p.get_label() for p in patch_handles],
                         loc="upper left", bbox_to_anchor=(1.0, 1), fontsize=7, title="model")

    fig.tight_layout()
    _save(fig, "fig6_individual_pretrained_finetuned_comparison.png")


def main() -> None:
    _style()
    figure3_arctic_sweep()
    figure4_individual_domain_results()
    figure5_training_curves()
    figure6_model_comparison()


if __name__ == "__main__":
    main()
