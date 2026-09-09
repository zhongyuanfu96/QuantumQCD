#!/usr/bin/env python3
"""Paper Figure 3: KL divergence attained by the three measurements (N = 8).

Regenerates the published figure ``known_barplot_KL.png`` (source: notebook
``optimization_known/adaptive_gradient/plots_combined.ipynb``, cell 9).

The figure is plot-only: six arrays of per-state-pair KL divergences, each of
shape (500, 1), are reduced to their mean and population standard deviation
(``np.std``, ddof=0).  Bars show the means, the dashed twin-axis curves show the
relative standard deviation (RSD = std / mean).

Naming trap preserved from the notebook: the folder ``plot_N8_100_pairs``
actually holds 500 pairs and is the **Wishart** ensemble, while
``adap_grad_jeremy/plot_N8_500_pairs`` is the **unitary-product** ensemble.

Usage
-----
    python3 scripts/fig03_kl_barplot.py [--out-dir DIR] [--plot-only]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# quantum_qcd modules are on the path for consistency with the other figure
# scripts; this figure needs no simulation routine from them.
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from figstyle import new_figure, save_figure  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data")
# Wishart ensemble (folder name says "100_pairs" but it holds 500 pairs).
DIR_W = os.path.join(DATA_DIR, "plot_N8_100_pairs")
# Unitary-product ensemble.
DIR_U = os.path.join(DATA_DIR, "adap_grad_jeremy", "plot_N8_500_pairs")

N = 8
CATEGORIES = ["Maximum-KL", "Helstrom", "Standard"]
BAR_W = 0.35


def load_stats():
    """Return (means_W, stds_W, means_U, stds_U), each of length 3."""
    order = ["vonNeumann", "helstrom", "standard"]
    arrays_W = [np.load(os.path.join(DIR_W, f"N{N}_{k}_KL.npy")) for k in order]
    arrays_U = [np.load(os.path.join(DIR_U, f"N{N}_{k}_KL.npy")) for k in order]
    means_W = np.array([a.mean() for a in arrays_W])
    stds_W = np.array([a.std() for a in arrays_W])          # ddof = 0
    means_U = np.array([a.mean() for a in arrays_U])
    stds_U = np.array([a.std() for a in arrays_U])          # ddof = 0
    return means_W, stds_W, means_U, stds_U


def make_figure(means_W, stds_W, means_U, stds_U):
    x = np.arange(len(CATEGORIES))                # 0, 1, 2

    fig, axes = new_figure(width="single")   # figstyle default height for a single panel
    ax = axes[0]

    # ---------- bar plot of KL means ----------
    ax.bar(x - BAR_W / 2, means_W, width=BAR_W, label=r"Mean (Wishart)", alpha=0.9)
    ax.bar(x + BAR_W / 2, means_U, width=BAR_W, label=r"Mean (Unitary-product)", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim([0, 1.5])
    ax.set_ylabel("KL Divergence")
    ax.grid(False)
    ax.grid(True, axis="y", linestyle="--")       # y-only grid, as in the notebook

    # ---------- relative-standard-deviation curves on a twin axis ----------
    ax2 = ax.twinx()
    ax2.grid(False)                               # twinx must not double the grid
    relstd_W = stds_W / means_W
    relstd_U = stds_U / means_U
    ax2.plot(x, relstd_W, marker="o", linestyle="--", color="tab:blue",
             markersize=4, label=r"RSD (Wishart)")
    ax2.plot(x, relstd_U, marker="s", linestyle="--", color="tab:orange",
             markersize=4, label=r"RSD (Unitary-product)")
    ax2.set_ylabel("Relative Standard Deviation")
    ax2.set_ylim(0, 0.75)

    # ---------- combined legend (order as in the notebook) ----------
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right",
               frameon=False, fontsize=6, handlelength=1.6, handletextpad=0.5,
               labelspacing=0.25, borderaxespad=0.3)

    return fig, ax, ax2, relstd_W, relstd_U


def main():
    p = argparse.ArgumentParser(description="Regenerate paper Figure 3.")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "figures"),
                   help="directory for Fig3.png / Fig3.pdf")
    p.add_argument("--plot-only", action="store_true",
                   help="no effect: this figure never simulates, it only reads "
                        "the stored KL arrays from data/")
    args = p.parse_args()

    means_W, stds_W, means_U, stds_U = load_stats()
    fig, ax, ax2, relstd_W, relstd_U = make_figure(means_W, stds_W, means_U, stds_U)

    print("category      mean_W   std_W    RSD_W    mean_U   std_U    RSD_U")
    for i, c in enumerate(CATEGORIES):
        print(f"{c:<12}  {means_W[i]:.6f} {stds_W[i]:.6f} {relstd_W[i]:.6f}  "
              f"{means_U[i]:.6f} {stds_U[i]:.6f} {relstd_U[i]:.6f}")
    print("left  y-limits:", ax.get_ylim(), " right y-limits:", ax2.get_ylim())

    png = save_figure(fig, 3, out_dir=args.out_dir)
    print("wrote", png)
    print("wrote", os.path.splitext(png)[0] + ".pdf")


if __name__ == "__main__":
    main()
