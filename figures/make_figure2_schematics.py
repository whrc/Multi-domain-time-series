"""
Builds Figure 2a (individual per-domain model) and Figure 2b (unified two-stage multi-domain
model) -- hand-designed methodology schematics, not data-driven, so kept separate from
run_figures_main.py. Design spec: figures/figure2_methodology_sketch_spec.md.

Saves both PNG (300dpi, matches every other figure) and SVG (vector, editable in
Illustrator/Inkscape) for each figure.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.plots import PALETTE  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent
DPI = 300

ARCTIC = PALETTE[6]      # reddish purple -- matches DOMAIN_COLOR in run_figures_main.py
AMAZON = PALETTE[5]      # vermilion
RANGELAND = PALETTE[4]   # blue
SHARED_DARK = "#4D4D4D"  # neutral slate gray -- not a domain, not a PALETTE hue

DOMAINS = [
    ("Arctic", ARCTIC, "GPP, RECO"),
    ("Amazon", AMAZON, "discharge,\nfire count, burned area"),
    ("Rangeland", RANGELAND, "GPP, RECO,\nRm, Rg"),
]


def _style() -> None:
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.linewidth": 0,
    })


def _lighten(color: str, amount: float = 0.65) -> tuple[float, float, float]:
    """Blend a color toward white -- used for the "frozen" state instead of a new hue."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _box(ax, cx: float, cy: float, w: float, h: float, text: str, facecolor,
          edgecolor: str = "black", linestyle: str = "-", fontsize: float = 7.0,
          fontweight: str = "normal", textcolor: str = "black", linewidth: float = 1.0) -> None:
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=facecolor, edgecolor=edgecolor, linestyle=linestyle,
        linewidth=linewidth, zorder=2, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, color=textcolor, zorder=3, linespacing=1.35)


def _arrow(ax, xy1, xy2, color: str = "black", lw: float = 1.0, linestyle: str = "-",
           mutation_scale: float = 7.0) -> None:
    arrow = FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", color=color, linewidth=lw, linestyle=linestyle,
        mutation_scale=mutation_scale, shrinkA=0, shrinkB=0, zorder=1, clip_on=False,
    )
    ax.add_patch(arrow)


def _badge(ax, cx: float, cy: float, number: str, radius: float = 0.14) -> None:
    """A large, clearly-distinguished circular badge (e.g. sequential-order ①②③ markers) --
    used instead of small inline text so it reads at a glance, not on close inspection."""
    circ = Circle((cx, cy), radius, facecolor="white", edgecolor="#333333", linewidth=1.1,
                  zorder=4, clip_on=False)
    ax.add_patch(circ)
    ax.text(cx, cy, number, ha="center", va="center", fontsize=9.5, fontweight="bold",
             color="#333333", zorder=5)


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def figure2a_individual_domain_sketch() -> None:
    """3 fully independent per-domain models, side by side -- zero parameter sharing."""
    col_w, gap = 2.0, 0.55
    col_x = [col_w / 2 + i * (col_w + gap) for i in range(3)]
    width = 3 * col_w + 2 * gap

    fig, ax = plt.subplots(figsize=(width, 2.9))
    ax.set_xlim(0, width)
    ax.set_ylim(-0.15, 2.9)
    ax.axis("off")

    input_h, tf_h, out_h = 0.75, 0.5, 0.75
    y_label, y_input, y_tf, y_out = 2.7, 2.25, 1.55, 0.85

    for cx, (name, color, targets) in zip(col_x, DOMAINS):
        ax.text(cx, y_label, name, ha="center", va="center", fontsize=9, fontweight="bold",
                 color=color)
        _box(ax, cx, y_input, col_w, input_h, "Input\n$(t = 1 \\ldots T,\\ n_{features})$",
             facecolor=_lighten(color, 0.82), edgecolor=color, linewidth=1.2)
        _arrow(ax, (cx, y_input - input_h / 2), (cx, y_tf + tf_h / 2), color=color)
        _box(ax, cx, y_tf, col_w, tf_h, "Transformer\nEncoder", facecolor=color,
             edgecolor=color, fontweight="bold", textcolor="white")
        _arrow(ax, (cx, y_tf - tf_h / 2), (cx, y_out + out_h / 2), color=color)
        _box(ax, cx, y_out, col_w, out_h, f"Prediction @ $T$ (last step):\n{targets}",
             facecolor=_lighten(color, 0.82), edgecolor=color, linewidth=1.2)

    ax.text(width / 2, 0.15,
             "Loss = masked-MSE(predictions, observations) in standardized space,\n"
             "independently per domain",
             ha="center", va="center", fontsize=7, style="italic", linespacing=1.4)

    fig.tight_layout(pad=0.3)
    _save(fig, "fig2a_individual_domain_sketch")


def _fan_x(cx: float, w: float, n: int) -> list[float]:
    """n evenly-spread x-positions across the middle 55% of a box of width w centered at cx --
    landing points for converging/diverging arrows so they fan out cleanly, not stack."""
    if n == 1:
        return [cx]
    span = w * 0.55
    return [cx - span / 2 + i * span / (n - 1) for i in range(n)]


def _draw_stage_panel(ax, x0: float, col_w: float, col_gap: float, title: str,
                        frozen_backbone: bool, sequential_heads: bool, caption: list[str]) -> tuple:
    """Draws one full Stage panel (3 inputs -> 3 projections -> 1 shared transformer -> 3 heads
    -> 3 outputs) at horizontal offset x0. Returns (panel_center_x, transformer_y, head_y) for
    the inter-panel checkpoint connectors drawn by the caller."""
    panel_w = 3 * col_w + 2 * col_gap
    col_x = [x0 + col_w / 2 + i * (col_w + col_gap) for i in range(3)]
    cx_panel = x0 + panel_w / 2

    y_title, y_input, y_proj, y_tf, y_head, y_out = 3.35, 2.95, 2.3, 1.6, 0.9, 0.3
    input_h, proj_h, tf_h, head_h, out_h = 0.35, 0.55, 0.55, 0.5, 0.5
    tf_w = col_w * 1.7

    ax.text(cx_panel, y_title, title, ha="center", va="center", fontsize=8.5, fontweight="bold")

    for cx, (name, color, _targets) in zip(col_x, DOMAINS):
        # Input (always a light, trainable-looking swatch -- inputs are just data, not weights)
        _box(ax, cx, y_input, col_w, input_h, f"{name} Input", facecolor=_lighten(color, 0.85),
             edgecolor=color, linewidth=0.9, fontsize=5.6)
        _arrow(ax, (cx, y_input - input_h / 2), (cx, y_proj + proj_h / 2), color=color, lw=0.8,
               mutation_scale=5.5)
        # Projection -- frozen in Stage 2
        if frozen_backbone:
            pc, pe, pls = _lighten(color, 0.82), _lighten(color, 0.35), "--"
        else:
            pc, pe, pls = color, color, "-"
        _box(ax, cx, y_proj, col_w, proj_h, "Linear Projection\n$(\\mathbb{R}^D)$", facecolor=pc,
             edgecolor=pe, linestyle=pls, fontsize=5.8, textcolor=("#888888" if frozen_backbone else "black"))

    # Converging arrows: projection -> shared transformer
    tf_edge, tf_fill, tf_ls = (SHARED_DARK, SHARED_DARK, "-") if not frozen_backbone else \
        (_lighten(SHARED_DARK, 0.35), _lighten(SHARED_DARK, 0.75), "--")
    for cx, land_x in zip(col_x, _fan_x(cx_panel, tf_w, 3)):
        _arrow(ax, (cx, y_proj - proj_h / 2), (land_x, y_tf + tf_h / 2), color="#999999", lw=0.7,
               mutation_scale=5)
    _box(ax, cx_panel, y_tf, tf_w, tf_h, "Shared\nTransformer", facecolor=tf_fill,
         edgecolor=tf_edge, linestyle=tf_ls, fontweight="bold",
         textcolor=("white" if not frozen_backbone else "#999999"), fontsize=6.8)

    # Diverging arrows: shared transformer -> heads
    for cx, land_x in zip(col_x, _fan_x(cx_panel, tf_w, 3)):
        _arrow(ax, (land_x, y_tf - tf_h / 2), (cx, y_head + head_h / 2), color="#999999", lw=0.7,
               mutation_scale=5)

    for i, (cx, (name, color, targets)) in enumerate(zip(col_x, DOMAINS)):
        _box(ax, cx, y_head, col_w, head_h, f"{name}\nMLP Head", facecolor=color, edgecolor=color,
             fontweight="bold", textcolor="white", fontsize=6.0)
        if sequential_heads:
            _badge(ax, cx + col_w / 2, y_head + head_h / 2, str(i + 1))
        _arrow(ax, (cx, y_head - head_h / 2), (cx, y_out + out_h / 2), color=color, lw=0.8,
               mutation_scale=5.5)
        _box(ax, cx, y_out, col_w, out_h, targets, facecolor=_lighten(color, 0.85),
             edgecolor=color, linewidth=0.9, fontsize=5.8)

    for i, line in enumerate(caption):
        ax.text(cx_panel, -0.05 - 0.24 * i, line, ha="center", va="center", fontsize=5.8,
                 style="italic", color="#333333", linespacing=1.3)

    return (cx_panel, y_tf, y_head)


def figure2b_multidomain_sketch() -> None:
    """Two side-by-side panels: (a) Stage 1 joint pretraining (everything trainable), (b) Stage
    2 per-domain fine-tuning (shared transformer + projections frozen, heads trained
    sequentially) -- connected by a dashed 'best checkpoint' arrow between the two panels'
    shared-transformer boxes."""
    col_w, col_gap, inter_gap = 1.35, 0.18, 1.1
    panel_w = 3 * col_w + 2 * col_gap
    width = 2 * panel_w + inter_gap

    fig, ax = plt.subplots(figsize=(width, 3.55))
    ax.set_xlim(0, width)
    ax.set_ylim(-0.85, 3.6)
    ax.axis("off")

    p1 = _draw_stage_panel(
        ax, 0.0, col_w, col_gap, "(a) Stage 1: Joint Pretraining",
        frozen_backbone=False, sequential_heads=False,
        caption=["All domains projected to common dimension $D$.",
                 "Loss = mean(masked-MSE, 3 domains), each in its own",
                 "standardized space; joint update every step",
                 "(mixed-domain batching)."],
    )
    p2 = _draw_stage_panel(
        ax, panel_w + inter_gap, col_w, col_gap, "(b) Stage 2: Per-Domain Fine-tuning",
        frozen_backbone=True, sequential_heads=True,
        caption=["Full Stage 1 checkpoint loaded (transformer,",
                 "projections, all heads); transformer + projections",
                 "then frozen. Each head continues training from its",
                 "Stage 1 value, independently and sequentially (1→2→3)."],
    )
    cx1, y_tf1, y_head1 = p1
    cx2, y_tf2, y_head2 = p2
    tf_half = col_w * 1.7 / 2

    # Full model (transformer + projections + every head) is loaded from Stage 1's best
    # checkpoint -- shown as two parallel connectors (transformer row, head row), not just one
    # arrow that would visually imply only the transformer transfers. Only the top connector
    # carries a label (short, fits the roomier proj-transformer gap); the bottom one is left
    # unlabeled -- same dashed style reads as "same relationship" once the first is labeled, and
    # there isn't vertical room for a second label without crowding the head boxes just below.
    _arrow(ax, (cx1 + tf_half, y_tf1), (cx2 - tf_half, y_tf2),
           color="#555555", lw=1.1, linestyle="--", mutation_scale=8)
    _arrow(ax, (cx1 + tf_half, y_head1), (cx2 - tf_half, y_head2),
           color="#555555", lw=1.1, linestyle="--", mutation_scale=8)
    ax.text((cx1 + cx2) / 2, y_tf1 + 0.22, "best checkpoint\ntransferred",
             ha="center", va="center", fontsize=6, style="italic", color="#555555", linespacing=1.2)

    fig.tight_layout(pad=0.3)
    _save(fig, "fig2b_multidomain_two_stage_sketch")


def main() -> None:
    _style()
    figure2a_individual_domain_sketch()
    figure2b_multidomain_sketch()


if __name__ == "__main__":
    main()
