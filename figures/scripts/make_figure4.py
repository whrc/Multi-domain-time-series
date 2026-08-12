"""
Figure 4: Individual domain model results, one row per domain (Arctic, Rangeland, Amazon),
three metric columns per row (RMSE, NSE, PBIAS). Arctic's row further split by SSP scenario
and historical/projected period; Rangeland's and Amazon's rows at the domain level.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.evaluate import scenario_period_label  # noqa: E402
from shared.plots import draw_metric_boxplot_panel  # noqa: E402
from _common import (  # noqa: E402
    AMAZON_TARGET_LABELS, AMAZON_TEST, ARCTIC_FLUXONLY_TEST, DOMAIN_COLOR,
    RANGELAND_DAY_TO_MONTH, RANGELAND_FLUXONLY_TEST,
    _add_grid, _horizontal_xticks, _load_seedavg, _save, _style,
)

METRICS_3COL = ["RMSE", "NSE", "PBIAS"]
ROW_LETTERS_4 = ["(a)", "(b)", "(c)"]
ARCTIC_GROUP_LABELS = {
    "historical": "Historical",
    "projected_ssp126": "Projected SSP1-2.6",
    "projected_ssp585": "Projected SSP5-8.5",
}
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


def main() -> None:
    _style()
    figure4_individual_domain_results()


if __name__ == "__main__":
    main()
