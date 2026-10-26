"""
Figure 2a/2b methodology schematics (individual vs. two-stage multi-domain model), saved as
PNG + SVG. Design spec: figure2_methodology_sketch_spec.md.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.use("Agg")

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))
from shared.plots import PALETTE  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
SVG_DIR = FIGURES_DIR / "svg"
DPI = 300
# Layout coordinates are pixels on a 200-dpi canvas, y down; fonts/line widths are in points.
CANVAS_DPI = 200

ARCTIC = PALETTE[6]      # reddish purple -- matches DOMAIN_COLOR in _common.py
AMAZON = PALETTE[5]      # vermilion
RANGELAND = PALETTE[4]   # blue
SHARED_DARK = "#4D4D4D"  # neutral gray for the shared transformer
FREEZE_BG = "#E5E5E5"
FREEZE_EDGE = "#999999"
FREEZE_LABEL = "#B87272"
FROZEN_TEXT = "#6E6E6E"
ANNOTATION_GRAY = "#808080"

DOMAINS = [
    ("Arctic", ARCTIC, "GPP, RECO"),
    ("Amazon", AMAZON, "discharge,\nfire count, burned area"),
    ("Rangeland", RANGELAND, "GPP, RECO,\nRm, Rg"),
]


def _blend(color: str, toward: str, amount: float) -> tuple[float, float, float]:
    """Linear RGB blend of `color` toward `toward` by `amount` (0-1)."""
    r, g, b = mcolors.to_rgb(color)
    tr, tg, tb = mcolors.to_rgb(toward)
    return (r + (tr - r) * amount, g + (tg - g) * amount, b + (tb - b) * amount)


def _canvas(width_px: float, height_px: float) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(width_px / CANVAS_DPI, height_px / CANVAS_DPI))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.axis("off")
    return fig, ax


def _box(ax, cx: float, cy: float, w: float, h: float, text: str, facecolor,
         textcolor: str = "black", fontsize: float = 10, fontweight: str = "normal") -> None:
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=8",
        facecolor=facecolor, edgecolor="none", zorder=2, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor, zorder=3, linespacing=1.4)


def _arrow(ax, xy1, xy2, color: str = "black", lw: float = 1.4,
           mutation_scale: float = 10) -> None:
    arrow = FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", color=color, linewidth=lw, mutation_scale=mutation_scale,
        shrinkA=0, shrinkB=0, zorder=3, clip_on=False,
    )
    ax.add_patch(arrow)


def _badge(ax, cx: float, cy: float, number: str, radius: float = 20) -> None:
    circ = Circle((cx, cy), radius, facecolor="white", edgecolor=ANNOTATION_GRAY, linewidth=1.6,
                  zorder=4, clip_on=False)
    ax.add_patch(circ)
    ax.text(cx, cy, number, ha="center", va="center", fontsize=11, fontweight="bold",
            color=ANNOTATION_GRAY, zorder=5)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    SVG_DIR.mkdir(exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    svg_path = SVG_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")
    plt.close(fig)


def figure2a_individual_domain_sketch() -> None:
    panel_left, panel_w, panel_gap = 33, 492, 94
    panel_top, panel_bottom = 68, 665
    tf_w, tf_h = 313, 154

    fig, ax = _canvas(1738, 740)
    for i, (name, color, targets) in enumerate(DOMAINS):
        left = panel_left + i * (panel_w + panel_gap)
        cx = left + panel_w / 2
        ax.add_patch(FancyBboxPatch(
            (left, panel_top), panel_w, panel_bottom - panel_top,
            boxstyle="round,pad=0,rounding_size=20", facecolor=_blend(color, "white", 0.84),
            edgecolor=color, linewidth=3, zorder=1, clip_on=False,
        ))
        ax.text(cx, 36, name, ha="center", va="center", fontsize=13.5, fontweight="bold",
                color=color)
        ax.text(cx, 139, "Input\n$(t = 1 \\ldots T,\\ n_{features})$", ha="center",
                va="center", fontsize=10.5, linespacing=1.35)
        _arrow(ax, (cx, 195), (cx, 265), lw=2.5, mutation_scale=20)
        _box(ax, cx, 354, tf_w, tf_h, "Transformer", facecolor=color, textcolor="white",
             fontsize=10.5, fontweight="bold")
        _arrow(ax, (cx, 445), (cx, 515), lw=2.5, mutation_scale=20)
        ax.text(cx, 538, f"Prediction @ $T$ (last step):\n{targets}", ha="center", va="top",
                fontsize=10.5, linespacing=1.35)
        ax.text(cx, 704, "Loss = Mean of per-target MSEs", ha="center", va="center",
                fontsize=10.5)

    _save(fig, "fig2a_individual_domain_sketch")


# Figure 2b panel geometry, shared by both Stage panels (x relative to the panel's offset)
COL_W, COL_PITCH, FIRST_COL_X = 230, 305.5, 143
PROJ_H, TF_W, TF_H, HEAD_H = 98, 482, 98, 90
Y_TITLE, Y_INPUT, Y_PROJ, Y_TF, Y_HEAD, Y_OUT, Y_LOSS = 38, 108, 224, 376, 521, 635, 708
FREEZE_TOP, FREEZE_BOTTOM, FREEZE_PAD = 147, 437, 35
# x offsets from column center of the side columns' diagonal arrows into/out of the transformer.
DIAG_NEAR, DIAG_FAR = 82, 105


def _draw_stage_panel(ax, x_offset: float, title: str, frozen: bool, loss_text: str) -> None:
    """One stage panel. `frozen` greys projections + transformer in a "Freeze" box and numbers
    the heads in fine-tuning order."""
    col_x = [x_offset + FIRST_COL_X + i * COL_PITCH for i in range(3)]
    cx_panel = col_x[1]
    feed_color = FROZEN_TEXT if frozen else "black"

    ax.text(cx_panel, Y_TITLE, title, ha="center", va="center", fontsize=13,
            fontweight="bold")
    if frozen:
        freeze_left = col_x[0] - COL_W / 2 - FREEZE_PAD
        freeze_right = col_x[2] + COL_W / 2 + FREEZE_PAD
        ax.add_patch(Rectangle(
            (freeze_left, FREEZE_TOP), freeze_right - freeze_left, FREEZE_BOTTOM - FREEZE_TOP,
            facecolor=FREEZE_BG, edgecolor=FREEZE_EDGE, linewidth=1.8, zorder=0,
        ))
        ax.text(freeze_right - 12, FREEZE_BOTTOM - 20, "Freeze", ha="right", va="center",
                fontsize=10.7, fontweight="bold", color=FREEZE_LABEL)

    for i, (cx, (name, color, targets)) in enumerate(zip(col_x, DOMAINS)):
        toward_center = (cx_panel > cx) - (cx_panel < cx)
        ax.text(cx, Y_INPUT, f"{name} Input", ha="center", va="center", fontsize=8.6)
        _arrow(ax, (cx, 125), (cx, 170), color=feed_color)
        _box(ax, cx, Y_PROJ, COL_W, PROJ_H, "Linear Projection\n$(\\mathbb{R}^D)$",
             facecolor=_blend(color, FREEZE_BG, 0.45) if frozen else color,
             textcolor=feed_color, fontsize=8.7)
        if toward_center:
            _arrow(ax, (cx + toward_center * DIAG_NEAR, 293),
                   (cx + toward_center * DIAG_FAR, 316), color=feed_color)
            _arrow(ax, (cx + toward_center * DIAG_FAR, 443),
                   (cx + toward_center * DIAG_NEAR, 466))
        else:
            _arrow(ax, (cx, 290), (cx, 318), color=feed_color)
            _arrow(ax, (cx, 428), (cx, 461))
        _box(ax, cx, Y_HEAD, COL_W, HEAD_H, f"{name}\nMLP Head", facecolor=color,
             textcolor="white", fontsize=9.2, fontweight="bold")
        if frozen:
            _badge(ax, cx - COL_W / 2 - 10, Y_HEAD, str(i + 1))
        _arrow(ax, (cx, 578), (cx, 600))
        ax.text(cx, Y_OUT, targets, ha="center", va="center", fontsize=8.8, linespacing=1.4)

    _box(ax, cx_panel, Y_TF, TF_W, TF_H, "Shared\nTransformer",
         facecolor=_blend(SHARED_DARK, FREEZE_BG, 0.45) if frozen else SHARED_DARK,
         textcolor="#E6E6E6" if frozen else "white", fontsize=10.3, fontweight="bold")
    ax.text(cx_panel, Y_LOSS, loss_text, ha="center", va="center", fontsize=9.1,
            color=ANNOTATION_GRAY)


def figure2b_multidomain_sketch() -> None:
    stage2_offset = 1059
    fig, ax = _canvas(2008, 749)

    _draw_stage_panel(ax, 0, "(a) Stage 1: Joint Pretraining", frozen=False,
                      loss_text="Loss = Mean of per-target MSEs across all domains")
    _draw_stage_panel(ax, stage2_offset, "(b) Stage 2: Per-Domain Fine-tuning", frozen=True,
                      loss_text="Loss = Mean of per-target MSEs for each domain")

    # Dotted shaft + separate solid head: a dotted FancyArrowPatch would dot the head too.
    x_start = FIRST_COL_X + COL_PITCH + TF_W / 2 + 38
    x_end = stage2_offset + FIRST_COL_X - COL_W / 2 - FREEZE_PAD - 25
    head_len = 20
    ax.plot([x_start, x_end - head_len], [Y_TF, Y_TF], linestyle=(0, (1.5, 2.5)), color="black",
            linewidth=1.6, solid_capstyle="butt")
    _arrow(ax, (x_end - head_len, Y_TF), (x_end, Y_TF), lw=1.6, mutation_scale=13)
    x_mid = (x_start + x_end) / 2
    ax.text(x_mid, Y_TF - 7, "Best checkpoint", ha="center", va="bottom", fontsize=8.9,
            color=ANNOTATION_GRAY)
    ax.text(x_mid, Y_TF + 7, "transferred", ha="center", va="top", fontsize=8.9,
            color=ANNOTATION_GRAY)

    _save(fig, "fig2b_multidomain_two_stage_sketch")


def main() -> None:
    figure2a_individual_domain_sketch()
    figure2b_multidomain_sketch()


if __name__ == "__main__":
    main()
