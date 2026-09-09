#!/usr/bin/env python3
"""Figure 10: detection performance with the estimated vs. the exact post-change distribution.

Regenerates paper Fig. 10 (published as ``unknown_FAPvsADD_estimated.png``) from the
pre-computed FAP/ADD sweeps stored in ``code_release/data/window_u/``.

Source: notebooks_raw/Quantum_Anomaly_Detection/optimization_unknown/plots_unknown.ipynb,
cell 31 (self-contained plotting cell; no simulation is involved).

Panel a: CUSUM-like test using the estimated post-change distribution q_hat_t obtained with
         an exponential window (eps = 0.99).
Panel b: CUSUM test using the exact post-change distribution q.
Both panels: maximum-sensitivity measurement (C0, blue) and standard measurement (C1, orange);
solid line = sample mean of the ADD over the 500 unitary-product state pairs, shaded band =
+/- 1 standard deviation (np.std, ddof = 0).

NOTE (reproduced verbatim from the notebook): the *same* exponential-window FAP array is used
in both panels -- only the ADD file switches between ``_exp99`` and ``_exact``.  There is no
``FAP_*_exact`` file and none is needed.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the paper's shared modules (no simulation is needed for this figure, but keep the
# release layout consistent: scripts import from ../src)
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
sys.path.insert(0, HERE)

from figstyle import label_panels, new_figure, save_figure  # noqa: E402

DATA_DIR = os.path.join(HERE, os.pardir, "data", "window_u")
DEFAULT_OUT = os.path.join(HERE, os.pardir, "figures")

# ---- parameters of the notebook cell -------------------------------------------------
MAX_FAP = 5000      # FAP horizon used for the truncation and the x axis
N_POINTS = 200      # resolution of the common FAP grid used for interpolation
N = 8               # Hilbert-space dimension
T0 = 256            # pre-change sample budget used when the sweeps were produced
BETA = 0.99         # forgetting factor of the exponential window
ENSEMBLE = "Unitary-product"
DISTRIBUTIONS = ["Maximum-sensitivity Measurement", "Standard Measurement"]
Q_TYPES = [r"Estimated $\hat{\mathbf{q}}_t$", r"Exact $\mathbf{q}$"]


def interp_mean_std(fap, add, common_x, max_fap=MAX_FAP):
    """Notebook post-processing: truncate each row at the first FAP >= max_fap, interpolate
    ADD onto the common FAP grid, then take the mean and the population std over the rows."""
    n_sets = fap.shape[0]
    all_interp = np.zeros((n_sets, common_x.size))
    for i in range(n_sets):
        idx = np.where(fap[i, :] >= max_fap)[0][0]
        all_interp[i, :] = np.interp(common_x, fap[i, : idx + 1], add[i, : idx + 1])
    return np.mean(all_interp, axis=0), np.std(all_interp, axis=0)


def load_curves():
    """Return {(q_type_index, meas): (mean, std)} plus the common FAP grid."""
    common_x = np.linspace(0, MAX_FAP, N_POINTS)
    curves = {}
    for col, q_type in enumerate(Q_TYPES):
        method = f"exp{int(BETA * 100)}" if q_type == Q_TYPES[0] else "exact"
        for name in DISTRIBUTIONS:
            meas = "vN" if name == "Maximum-sensitivity Measurement" else "st"
            # the exponential-window FAP is deliberately reused for the exact-q panel
            fap = np.load(os.path.join(DATA_DIR, f"N{N}_{T0}_FAP_{meas}_exp{int(BETA * 100)}.npy"))
            add = np.load(os.path.join(DATA_DIR, f"N{N}_{T0}_ADD_{meas}_{method}.npy"))
            print(f"[fig10] col {col} ({method}), meas {meas}: FAP {fap.shape}, ADD {add.shape}")
            curves[(col, meas)] = interp_mean_std(fap, add, common_x)
    return common_x, curves


def make_figure(common_x, curves, out_dir):
    fig, axes = new_figure(nrows=1, ncols=2, width="double", height=2.9, sharey=True)

    labels = {
        "vN": rf"Maximum-sensitivity Measurement w/ an Exponential Window, $\epsilon$ = {BETA}",
        "st": rf"Standard Measurement w/ an Exponential Window, $\epsilon$ = {BETA}",
    }
    for col in range(2):
        ax = axes[col]
        for meas in ("vN", "st"):                     # vN -> C0 blue, st -> C1 orange
            mean_add, std_add = curves[(col, meas)]
            line, = ax.plot(common_x, mean_add, label=labels[meas])
            ax.fill_between(common_x, mean_add - std_add, mean_add + std_add,
                            color=line.get_color(), alpha=0.2)
        ax.set_xlim(0, MAX_FAP)                       # y limits: autoscaled, as in the notebook
        ax.grid(linestyle="--")

    handles, lab = axes[0].get_legend_handles_labels()
    # axes occupy the upper part of the canvas; the shared FAP label and the plot key
    # (below the axes, exactly as in the source cell) sit in the reserved strip below
    fig.tight_layout(rect=[0.035, 0.215, 1, 1])
    fig.supxlabel("FAP", y=0.145)
    fig.supylabel("ADD", x=0.012)
    fig.legend(handles, lab, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=1, frameon=False)
    label_panels(fig, axes)
    png = save_figure(fig, 10, out_dir=out_dir)
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="directory for Fig10.png / Fig10.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help="no-op for this figure: it is a pure plotting script that reads the "
                         "pre-computed sweeps in data/window_u/ (no simulation, no cache)")
    args = ap.parse_args()

    common_x, curves = load_curves()

    # numbers quoted in the self-check
    for col, tag in enumerate(("estimated q_hat_t", "exact q")):
        for meas in ("vN", "st"):
            mean_add, std_add = curves[(col, meas)]
            for x in (1000.0, 2500.0, 5000.0):
                j = int(np.argmin(np.abs(common_x - x)))
                print(f"[fig10] {tag:>18s} {meas}: FAP={common_x[j]:7.1f} "
                      f"mean ADD={mean_add[j]:8.3f} std={std_add[j]:8.3f}")

    png = make_figure(common_x, curves, os.path.abspath(args.out_dir))
    print(f"[fig10] wrote {png} and {png[:-4]}.pdf")


if __name__ == "__main__":
    main()
