#!/usr/bin/env python3
"""Paper Figure 9 -- effect of the estimation window on detection performance
with an unknown post-change state.

ADD versus FAP for 500 unitary-product state pairs (N = 8), for the
window-based CUSUM-like test with three estimators of the post-change
distribution q_hat_t:

    infinite window,  T0 = 256                      -> C0 blue
    finite window,    T1 = 100, T0 = 256            -> C1 orange
    exponential window, eps = 0.99                  -> C2 green

Panel a: maximum-sensitivity ("vN", rho-eigenbasis) measurement.
Panel b: standard (computational-basis) measurement.

Source: notebooks_raw/Quantum_Anomaly_Detection/optimization_unknown/
        plots_unknown.ipynb, cell 29 (pure plotting cell -- everything is
        loaded from the saved sweeps in window_u/; no simulation is needed).

The FAP/ADD sweeps themselves were produced by
optimization_unknown/Sensitivity_FAP_ADD_full_rank_2.ipynb (500 state pairs
x 100 thresholds x 500 Monte-Carlo runs, many CPU-hours); they are shipped as
.npy files under data/window_u/ and only loaded here.

Usage:
    python3 scripts/fig09_windows.py [--out-dir DIR] [--plot-only]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# the released modules live in ../src/quantum_qcd (kept on the path for
# consistency with the other figure scripts; Fig. 9 is pure post-processing
# of the saved sweeps and needs none of them)
sys.path.insert(0, os.path.join(_ROOT, "src", "quantum_qcd"))
sys.path.insert(0, _HERE)

from figstyle import new_figure, label_panels, save_figure  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data", "window_u")

# ---------------------------------------------------------------- parameters
# exactly the values set inside notebook cell 29
MAX_FAP = 5000      # x-axis upper limit and interpolation cut-off
N_POINTS = 200      # resolution of the common FAP grid
N = 8               # Hilbert-space dimension
T0 = 256            # Dirichlet-multinomial prior weight
BETA = 0.99         # exponential-window forgetting factor (eps)
T_HAT = 100         # finite-window length T1
ENSEMBLE = "Unitary-product"

# (measurement key in the file names, panel) in the notebook's column order
MEASUREMENTS = ["vN", "st"]
# method loop order fixes the default-cycle colours: inf -> C0, fnt -> C1, exp -> C2
METHODS = ["inf", "fnt", "exp"]


def method_key_and_label(method):
    """Return (file-name suffix, legend label) exactly as in cell 29."""
    if method == "fnt":
        return (f"fnt{T_HAT}",
                r"Finite, $T_1$" + f" = {T_HAT}" + r", $T_0$" + f" = {T0}")
    if method == "exp":
        return (f"exp{int(BETA * 100)}",
                f"Exponential, $\\epsilon$ = {BETA}")
    return "inf", r"Infinite, $T_0$" + f" = {T0}"


def load_curves(meas, key):
    """Load the (500, 100) FAP and ADD sweeps for one measurement/window."""
    fap = np.load(os.path.join(DATA_DIR, f"N{N}_{T0}_FAP_{meas}_{key}.npy"))
    add = np.load(os.path.join(DATA_DIR, f"N{N}_{T0}_ADD_{meas}_{key}.npy"))
    return fap, add


def interpolate_mean_std(fap, add, common_x):
    """Cell-29 post-processing.

    For every state pair, truncate the threshold sweep at the FIRST index whose
    FAP reaches MAX_FAP (inclusive), interpolate ADD onto the common FAP grid,
    then take the mean and the population standard deviation (np.std, ddof=0)
    across the 500 pairs.
    """
    n_sets = fap.shape[0]
    all_interp = np.zeros((n_sets, common_x.size))
    for i in range(n_sets):
        idx = np.where(fap[i, :] >= MAX_FAP)[0][0]
        all_interp[i, :] = np.interp(common_x, fap[i, :idx + 1], add[i, :idx + 1])
    return np.mean(all_interp, axis=0), np.std(all_interp, axis=0)


def build_figure():
    common_x = np.linspace(0, MAX_FAP, N_POINTS)

    # width='double' (6.5 in) for a two-panel figure; a little extra height so
    # the shared key below the axes and the panel letters both fit
    fig, axes = new_figure(nrows=1, ncols=2, width="double", height=2.6,
                           sharey=True)

    for col, meas in enumerate(MEASUREMENTS):
        ax = axes[col]
        for method in METHODS:
            key, label_name = method_key_and_label(method)
            fap, add = load_curves(meas, key)
            mean_add, std_add = interpolate_mean_std(fap, add, common_x)

            line, = ax.plot(common_x, mean_add, label=label_name)
            ax.fill_between(common_x,
                            mean_add - std_add,
                            mean_add + std_add,
                            color=line.get_color(), alpha=0.2)
            print(f"{ENSEMBLE:<15}, {meas}, {key:<6}: "
                  f"ADD(FAP=5000) = {mean_add[-1]:.2f} +- {std_add[-1]:.2f}")

        ax.set_xlim(0, MAX_FAP)
        ax.grid(linestyle="--")
        # no ax.set_title(...): journal rule (the notebook titled the panels
        # 'Unitary-product, Maximum-sensitivity Measurement' / '..., Standard
        # Measurement'; that information now lives in the caption)

    axes[0].set_ylim(0, 3000)          # sharey propagates to panel b

    # shared axis labels, as in the notebook (fig.supxlabel / fig.supylabel)
    supx = fig.supxlabel("FAP", fontsize=8)
    fig.supylabel("ADD", fontsize=8)

    # single shared key below the panels, as the notebook did (3 columns,
    # no frame); this is a plot key, not caption text, so it is kept
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 1])
    supx.set_y(0.055)                  # pull "FAP" up against the tick labels
    label_panels(fig, axes)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(_ROOT, "figures"),
                        help="directory for Fig9.png / Fig9.pdf")
    parser.add_argument("--plot-only", action="store_true",
                        help="no-op for this figure: Fig. 9 never simulates, "
                             "it only re-plots the shipped data/window_u sweeps")
    args = parser.parse_args()

    fig = build_figure()
    png = save_figure(fig, 9, out_dir=args.out_dir)
    print("wrote", png)
    print("wrote", os.path.splitext(png)[0] + ".pdf")


if __name__ == "__main__":
    main()
