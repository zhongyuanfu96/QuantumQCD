#!/usr/bin/env python3
"""Figure 1 - convergence of projected gradient ascent (Alg. 1) for full-rank states.

Reproduces the published figure ``rho_sigma_set.png`` (notebook
``optimization_known/adaptive_gradient/plots_combined.ipynb``, cell 5).

The figure is pure plotting: four pre-computed (3, 800) arrays of the KL divergence
between the pre- and post-change outcome distributions along the 800 PGA iterations,
for three independently sampled full-rank state pairs with N = 8.

  a  Wishart ensemble,          U^(0) = I_N        data/plot_alg_convergence/N8_U_I.npy
  b  Wishart ensemble,          U^(0) = Lambda     data/plot_alg_convergence/N8_U_Lambda.npy
  c  unitary-product ensemble,  U^(0) = I_N        data/plot_3pair_state/N8_U_I.npy
  d  unitary-product ensemble,  U^(0) = Lambda     data/plot_3pair_state/N8_U_Lambda.npy

No simulation and no RNG are involved, so ``--plot-only`` is accepted but changes
nothing: every run reads the same four .npy files from ``data/``.

Usage:
    PYTHONPATH=src python3 scripts/fig01_pga_convergence_full_rank.py
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))   # quantum_qcd package (not needed here: plot only)
sys.path.insert(0, HERE)                        # figstyle

from figstyle import new_figure, label_panels, save_figure  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")

# (sub-folder, file) for each panel, in row-major order a, b, c, d.
PANEL_FILES = [
    ("plot_alg_convergence", "N8_U_I.npy"),       # a  Wishart,         U0 = I_N
    ("plot_alg_convergence", "N8_U_Lambda.npy"),  # b  Wishart,         U0 = Lambda
    ("plot_3pair_state", "N8_U_I.npy"),           # c  unitary-product, U0 = I_N
    ("plot_3pair_state", "N8_U_Lambda.npy"),      # d  unitary-product, U0 = Lambda
]
N_SETS = 3          # three state pairs per panel -> C0 blue, C1 orange, C2 green
YLIM_ROW = [(0, 1.8), (0, 0.45)]   # top row (Wishart), bottom row (unitary-product)
YTICK_ROW = [0.2, 0.05]            # y-tick spacing of the published figure


def load_curves():
    """Load the four (3, 800) KL-divergence traces in panel order a, b, c, d."""
    curves = []
    for folder, name in PANEL_FILES:
        path = os.path.join(DATA_DIR, folder, name)
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] < N_SETS:
            raise RuntimeError(f"{path}: unexpected shape {arr.shape}")
        curves.append(arr)
    return curves


def make_figure(curves):
    # 2 x 2, shared x and per-row shared y, exactly as the source cell.
    fig, axes = new_figure(nrows=2, ncols=2, width="double", height=4.2,
                           sharex=True, sharey="row")
    for panel, (ax, arr) in enumerate(zip(axes, curves)):
        for set_num in range(N_SETS):
            # no explicit colour -> default cycle C0 blue, C1 orange, C2 green; solid, no marker
            ax.plot(arr[set_num, :])
        ax.set_ylim(*YLIM_ROW[panel // 2])
        ax.yaxis.set_major_locator(MultipleLocator(YTICK_ROW[panel // 2]))
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.grid(True, linestyle="--")
    # figure-level axis labels (the source cell used fontsize=12 on a 12x7 in canvas;
    # here the figstyle label size is used on the 6.5 in canvas)
    fig.supxlabel("Iterations")
    fig.supylabel("KL Divergence")
    # no titles and no suptitle (journal rule); the panel information is in the caption
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # extra vertical room between the rows so the panel letters c, d do not collide
    # with the tick labels of the top row
    fig.subplots_adjust(hspace=0.30)
    label_panels(fig, axes, dy=0.012)
    return fig, axes


def report(curves):
    """Print the plotted values at a few iteration indices for the self-check."""
    idx = [0, 100, 200, 400, 799]
    names = ["a Wishart U0=I", "b Wishart U0=Lambda",
             "c unitary-product U0=I", "d unitary-product U0=Lambda"]
    print(f"iterations checked: {idx}")
    for name, arr in zip(names, curves):
        print(f"  {name}: shape {arr.shape}")
        for set_num in range(N_SETS):
            vals = ", ".join(f"{arr[set_num, i]:.4f}" for i in idx)
            print(f"    pair {set_num + 1} (C{set_num}): {vals}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"),
                    help="directory for Fig1.png / Fig1.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help="accepted for interface consistency; this figure never simulates")
    args = ap.parse_args()

    curves = load_curves()
    report(curves)
    fig, _ = make_figure(curves)
    png = save_figure(fig, 1, out_dir=args.out_dir)
    print("wrote", png)
    print("wrote", os.path.splitext(png)[0] + ".pdf")


if __name__ == "__main__":
    main()
