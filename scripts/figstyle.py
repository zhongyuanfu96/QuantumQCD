"""Shared figure style for the paper figures (npj Quantum Information).

Rules implemented here (from the journal's technical check):
- no figure titles, no "Figure N" text and no legend/caption text inside the image
  (axis labels, tick labels and plot keys are fine);
- multi-panel figures carry bold lowercase panel letters a, b, c, ... inside the
  image, placed above the top-left corner of each panel; single-panel figures carry none;
- one image file per figure: FigN.png (330 dpi raster) and FigN.pdf (vector), same content.

Usage:
    from figstyle import new_figure, label_panels, save_figure, COLORS
    fig, axes = new_figure(ncols=2, width='double')      # returns fig and flat list of axes
    ... plot ...
    label_panels(fig, axes)                                # only for multi-panel figures
    save_figure(fig, 15)                                  # writes figures/Fig15.png and Fig15.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# widths in inches: match the \includegraphics widths used in the manuscript
WIDTHS = {"single": 3.5, "medium": 4.5, "double": 6.5}
COLORS = {"blue": "C0", "orange": "C1", "green": "C2", "red": "C3", "purple": "C4"}
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,      # titles must not be used; kept equal for safety
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "grid.color": "0.8",
    "axes.axisbelow": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,       # embed TrueType so text stays editable
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


def new_figure(nrows=1, ncols=1, width="single", height=None, sharex=False, sharey=False, **kw):
    """Create a figure of the given width (inches or key of WIDTHS); returns (fig, flat axes list)."""
    w = WIDTHS.get(width, width)
    if height is None:
        height = (w / ncols) * 0.72 * nrows + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height), sharex=sharex, sharey=sharey, **kw)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    return fig, axes


def label_panels(fig, axes, letters="abcdefgh", dx=-0.02, dy=0.02, size=10):
    """Bold lowercase letters above the top-left corner of each panel (row-major order)."""
    fig.canvas.draw()
    for ax, letter in zip(axes, letters):
        bbox = ax.get_position()
        fig.text(bbox.x0 + dx, bbox.y1 + dy, letter, fontsize=size, fontweight="bold",
                 ha="right", va="bottom")


def assert_no_titles(fig):
    for ax in fig.axes:
        if ax.get_title():
            raise RuntimeError(f"panel title present: {ax.get_title()!r}; titles are not allowed")
    if fig._suptitle is not None:
        raise RuntimeError("figure suptitle present; not allowed")


def save_figure(fig, number, out_dir=FIG_DIR, tight=True):
    """Write figures/Fig<number>.png (300 dpi) and .pdf; returns the png path."""
    assert_no_titles(fig)
    os.makedirs(out_dir, exist_ok=True)
    kw = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    png = os.path.join(out_dir, f"Fig{number}.png")
    # 330 dpi so that the tight-cropped canvas still exceeds 300 dpi at the width
    # the manuscript places it (the crop trims a few percent of the nominal width)
    fig.savefig(png, dpi=330, **kw)
    fig.savefig(os.path.join(out_dir, f"Fig{number}.pdf"), **kw)
    return png
