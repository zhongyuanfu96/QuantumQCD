#!/usr/bin/env python3
"""Figure 8: sensitivity of the maximum-sensitivity and standard measurements.

Reproduces cell 16 of
notebooks_raw/Quantum_Anomaly_Detection/optimization_unknown/plots_unknown.ipynb
(published image: png_original/unknown_sens_barplot.png).

Two ensembles of 500 full-rank pre-change states with N = 8 are used:

* "Wishart"          -- data/rho_sigma/rho_8_set_500.npy
                        (generate_rho_set(8, 500, load_exist=True), cell 4)
* "Unitary-product"  -- data/adap_grad_jeremy/rho_8_set_500.npy
                        (generate_rho_set_jeremy(8, 500, load_exist=True), cell 7)

Both plotted quantities are closed form (no Monte Carlo):

* maximum-sensitivity measurement:  G = sum_i 1 / lambda_i(rho) = tr(rho^-1)
* standard (computational-basis) measurement:  G = sum_i 1 / p_i,
  p_i = tr(|i><i| rho) = Re diag(rho)

Bars show the ensemble mean of G, the dashed twin-axis lines the relative
standard deviation RSD = std(G) / mean(G) with std computed with ddof = 0
(np.std default), exactly as in the notebook.

The computation takes well under a second; the resulting six arrays are still
cached in data/derived/fig08_sensitivity.npz so that --plot-only can re-plot
without touching the state sets.

Usage:
    python3 scripts/fig08_sensitivity_barplot.py [--out-dir DIR] [--plot-only]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from quantum_qcd.quantum_utils import measurement_distribution  # noqa: E402
from figstyle import new_figure, save_figure  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
CACHE = os.path.join(DERIVED_DIR, "fig08_sensitivity.npz")

# Notebook parameters (cells 13 and 16), unchanged.
N = 8
N_SETS = 500
MEASUREMENTS = ["Maximum-sensitivity", "Standard"]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_rho_sets():
    """The two 500-state ensembles, loaded only from code_release/data/."""
    rho_w = np.load(os.path.join(DATA_DIR, "rho_sigma", f"rho_{N}_set_{N_SETS}.npy"))
    rho_u = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy", f"rho_{N}_set_{N_SETS}.npy"))
    return rho_w, rho_u


def standard_povm(n):
    """Computational-basis PVM {|i><i|}, i = 0 ... n-1, as an (n, n, n) stack."""
    I = np.eye(n)
    return np.stack([np.outer(I[:, i], I[:, i].conj()) for i in range(n)])


def max_sens_stats(rho_set):
    """Mean and std (ddof = 0) of tr(rho^-1) over a (n_sets, N, N) ensemble."""
    max_sens = np.sum(1 / np.linalg.eigvalsh(rho_set), axis=1)
    return max_sens.mean(), max_sens.std()


def max_sens_stats_standard(n, n_sets, rho_set):
    """Mean and std (ddof = 0) of sum_i 1/p_i under the standard measurement.

    measurement_distribution(Q, rho) with the computational-basis PVM returns
    Re tr(|i><i| rho), i.e. the real diagonal of rho: identical to the
    notebook's distribution_standard(n, rho) helper (cell 11).  The PVM has
    n = N elements, so no null outcome is appended.
    """
    Q = standard_povm(n)
    max_sens = np.zeros(n_sets)
    for set_num in range(n_sets):
        p = measurement_distribution(Q, rho_set[set_num, :, :])
        max_sens[set_num] = np.sum(1 / p)
    return max_sens.mean(), max_sens.std()


def compute(verbose=True):
    """Bar heights and RSD values, in the notebook's loop order."""
    rho_w, rho_u = load_rho_sets()

    means_w, stds_w = [], []
    means_u, stds_u = [], []
    for measure in MEASUREMENTS:
        # Wishart ensemble ------------------------------------------------
        if measure == "Maximum-sensitivity":
            mu, sd = max_sens_stats(rho_w)
        else:
            mu, sd = max_sens_stats_standard(N, N_SETS, rho_w)
        means_w.append(mu)
        stds_w.append(sd)
        if verbose:
            print(f"Measurement = {measure} done")
            print(f"Wishart mean, sd, sd/mean: {mu:.3f}, {sd:.3f}, {sd / mu:.3f}")

        # Unitary-product ensemble ----------------------------------------
        if measure == "Maximum-sensitivity":
            mu, sd = max_sens_stats(rho_u)
        else:
            mu, sd = max_sens_stats_standard(N, N_SETS, rho_u)
        means_u.append(mu)
        stds_u.append(sd)
        if verbose:
            print(f"Unitary-product mean, sd, sd/mean: {mu:.3f}, {sd:.3f}, {sd / mu:.3f}")

    means_w = np.array(means_w)
    means_u = np.array(means_u)
    stds_w = np.array(stds_w)
    stds_u = np.array(stds_u)
    return {
        "means_w": means_w,
        "means_u": means_u,
        "stds_w": stds_w,
        "stds_u": stds_u,
        "rel_std_w": stds_w / means_w,
        "rel_std_u": stds_u / means_u,
    }


def load_cache():
    with np.load(CACHE) as f:
        return {k: f[k] for k in f.files}


def save_cache(res):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    np.savez(CACHE, **res)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot(res, out_dir):
    """Grouped bars (mean sensitivity) plus twin-axis RSD lines, one panel."""
    fig, (ax,) = new_figure(width="single")

    x = np.arange(len(MEASUREMENTS))  # category centres
    width = 0.30                      # bar width

    # Bars: default colour cycle C0 (Wishart) and C1 (Unitary-product).
    # capsize is passed as in the notebook, but no yerr is given, so no error
    # bars are drawn.
    ax.bar(x - width / 2, res["means_w"], width, capsize=4, alpha=0.8, label="Wishart")
    ax.bar(x + width / 2, res["means_u"], width, capsize=4, alpha=0.8, label="Unitary-product")

    ax.set_xticks(x, [f"{measure}" for measure in MEASUREMENTS])
    ax.set_ylabel("Mean Sensitivity")
    ax.set_ylim(0, 4.5e3)            # linear y axis, as in the published figure
    ax.grid(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, res["rel_std_w"], marker="o", linestyle="--", color="tab:blue",
             label=r"RSD (Wishart)")
    ax2.plot(x, res["rel_std_u"], marker="s", linestyle="--", color="tab:orange",
             label=r"RSD (Unitary-product)")
    ax2.set_ylabel("RSD")
    ax2.set_ylim(0, 15)
    ax2.grid(False)

    # ---------- combine legends ----------
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)

    # Single-panel figure: no panel letter, and no title (journal rules).
    return save_figure(fig, 8, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "figures"),
                        help="directory for Fig8.png / Fig8.pdf")
    parser.add_argument("--plot-only", action="store_true",
                        help="re-plot from data/derived/fig08_sensitivity.npz")
    args = parser.parse_args()

    np.random.seed(0)  # no RNG is used here; set for uniformity across scripts

    if args.plot_only:
        if not os.path.exists(CACHE):
            raise SystemExit(f"--plot-only needs the cache {CACHE}; run without it first")
        res = load_cache()
        print(f"loaded cached statistics from {CACHE}")
    else:
        res = compute()
        save_cache(res)
        print(f"cached statistics in {CACHE}")

    print("bar heights  (Wishart)         :", np.array2string(res["means_w"], precision=3))
    print("bar heights  (Unitary-product) :", np.array2string(res["means_u"], precision=3))
    print("RSD          (Wishart)         :", np.array2string(res["rel_std_w"], precision=3))
    print("RSD          (Unitary-product) :", np.array2string(res["rel_std_u"], precision=3))

    png = plot(res, args.out_dir)
    print("wrote", png)
    print("wrote", png[:-4] + ".pdf")


if __name__ == "__main__":
    main()
