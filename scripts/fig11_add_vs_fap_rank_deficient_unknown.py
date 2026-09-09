#!/usr/bin/env python3
"""Figure 11: ADD vs FAP for rank-deficient pre-change states with an *unknown* post-change state.

Regenerates paper Fig. 11 (published as ``unknown_ADDvsFAP_R.png``).

Source cell
-----------
notebooks_raw/Quantum_Anomaly_Detection/optimization_unknown/Sensitivity_FAP_ADD_rank_def.ipynb,
cell 25 (helpers: cells 1, 4, 5, 6, 10, 11, 15; set-up cell 21).  The embedded output of
cell 25 is byte-identical (md5 b3b144479fe110390703c966874cc9b0) to the published PNG.

What is simulated
-----------------
N = 8, one state pair per rank: ``rho = rho_8_rank_R_set_500.npy[9]`` (index
``set_num_rank_def = 9``) for R in {5, 6, 7, 8}, and the single full-rank post-change state
``sigma_8.npy`` (the same sigma for every R).  The measurement is the maximum-sensitivity
("optimized von Neumann") POVM ``Q = povm_positive_eigen(rho, eps=1e-6)``: the R rank-1
projectors onto the eigenvectors of rho with eigenvalue > 1e-6.  Hence

    p_dist = tr(Q_i rho)                             (R entries)
    q_dist = tr(Q_i sigma)                           (R entries)
    q_dist = np.append(q_dist, 1 - sum(q_dist))      (R + 1 entries, only when R < N)

the appended entry being the mass of sigma outside supp(rho), i.e. the null-space outcome.
For R = 8 there is no null-space outcome and q keeps 8 entries.

Threshold grid: ``h_exp = np.linspace(0, 4.5, 100)`` for every R.  Three Monte-Carlo runs
per rank, all with a *vector* of thresholds so that one trajectory feeds every threshold
(this is what makes the published curves smooth):

  fap_vN       = cusum_fap_exp_single (2048,   p, h, forget_val = 0.99)     -> x axis, both panels
  add_vN_est   = cusum_add_exp_single (204800, p, q, h, forget_val = 0.99)  -> panel a
  add_vN_exact = cusum_add_exp_single_exact(204800, p, q, h)                -> panel b

Panel a uses the exponential-window estimate q_hat_t (initialised at [p, 0], eps = 0.99,
updated *after* the LLR and the threshold test); panel b uses the exact q.  In both, hitting
the null-space outcome stops the run immediately and credits the current time to every
threshold that has not fired yet.

The serial notebook routines (cusum_fap_exp_single / cusum_add_exp_single /
cusum_add_exp_single_exact, re-exported unchanged in quantum_qcd.cusum_nb) took ~30 min on
Colab.  This script calls their seeded, vectorised equivalents
(cusum_*_exp_single*_batch in quantum_qcd.cusum_nb), which are the same statistical objects
with the same sample sizes.

Journal changes relative to the notebook cell: the two panel titles (r'Estimated $\\hat{q}_t$'
and r'Exact $\\mathbf{q}$') are dropped and replaced by the bold panel letters a / b -- the
manuscript legend carries that information; figstyle sizes/fonts are used.  Everything else
(colours C0..C3 for R = 5, 6, 7, 8, solid lines, linear axes, xlim 0-5000, dashed grid,
shared y axis, plot key of four entries in one row below the axes, supxlabel 'FAP' /
supylabel 'ADD') follows the source cell.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
sys.path.insert(0, HERE)

from figstyle import label_panels, new_figure, save_figure  # noqa: E402
from quantum_qcd.cusum_nb import (  # noqa: E402
    cusum_add_exp_single_batch,
    cusum_add_exp_single_exact_batch,
    cusum_fap_exp_single_batch,
)
from quantum_qcd.quantum_utils import measurement_distribution, povm_positive_eigen  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, os.pardir, "data"))
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, os.pardir, "figures"))
CACHE = os.path.join(DERIVED_DIR, "fig11_add_vs_fap_unknown.npz")

# ---- parameters of the notebook cell (cells 21 and 25) --------------------------------
N = 8                     # Hilbert-space dimension
N_SETS = 500              # state sets on disk: rho_8_rank_R_set_500.npy
R_LIST = [5, 6, 7, 8]     # pre-change ranks
SET_NUM_RANK_DEF = 9      # index of the state pair used for every rank
TEST_SIZE = 2 ** 11       # 2048 Monte-Carlo runs for the FAP
ADD_MULTI = 100           # ADD uses TEST_SIZE * ADD_MULTI = 204800 runs
FORGOT_VAL = 0.99         # exponential-window forgetting factor
MAX_FAP = 5000            # x-axis limit
EPS_SUPPORT = 1e-6        # eigenvalue threshold of povm_positive_eigen
H_EXP = np.linspace(0, 4.5, 100)
SEED = 0                  # the notebook was unseeded; we seed for reproducibility


def distributions(R):
    """(p_dist, q_dist) of the maximum-sensitivity measurement for pre-change rank R."""
    rho_set = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy",
                                   f"rho_{N}_rank_{R}_set_{N_SETS}.npy"))
    sigma = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy", f"sigma_{N}.npy"))
    rho = rho_set[SET_NUM_RANK_DEF, :, :]

    Q = povm_positive_eigen(rho, eps=EPS_SUPPORT)          # (R, N, N) rank-1 projectors
    p_dist = measurement_distribution(Q, rho, append_null=False)     # R entries
    q_dist = measurement_distribution(Q, sigma, append_null=True)    # R + 1 entries if R < N
    return p_dist, q_dist


def simulate():
    """Run the three Monte-Carlo sweeps for every rank; returns the arrays the figure plots."""
    n_h = H_EXP.size
    fap = np.zeros((len(R_LIST), n_h))
    add_est = np.zeros((len(R_LIST), n_h))
    add_exact = np.zeros((len(R_LIST), n_h))
    q_last = np.full(len(R_LIST), np.nan)

    for k, R in enumerate(R_LIST):
        p_dist, q_dist = distributions(R)
        if R < N:
            q_last[k] = q_dist[-1]
            print(f"[fig11] R = {R}: p sum {p_dist.sum():.3f} {p_dist.shape}, "
                  f"q sum {q_dist.sum():.3f} {q_dist.shape}, q_null {q_dist[-1]:.3f}", flush=True)
        else:
            print(f"[fig11] R = {R}: p sum {p_dist.sum():.3f} {p_dist.shape}, "
                  f"q sum {q_dist.sum():.3f} {q_dist.shape}", flush=True)

        t0 = time.time()
        # independent RNG streams per (rank, quantity), all derived from SEED
        fap[k] = cusum_fap_exp_single_batch(TEST_SIZE, p_dist, H_EXP,
                                            forget_val=FORGOT_VAL, batch_size=TEST_SIZE,
                                            seed=np.random.default_rng([SEED, R, 0]))
        t1 = time.time()
        add_est[k] = cusum_add_exp_single_batch(TEST_SIZE * ADD_MULTI, p_dist, q_dist, H_EXP,
                                                forget_val=FORGOT_VAL, batch_size=4096,
                                                seed=np.random.default_rng([SEED, R, 1]))
        t2 = time.time()
        add_exact[k] = cusum_add_exp_single_exact_batch(TEST_SIZE * ADD_MULTI, p_dist, q_dist,
                                                        H_EXP, batch_size=4096,
                                                        seed=np.random.default_rng([SEED, R, 2]))
        t3 = time.time()
        print(f"[fig11] R = {R}: FAP {t1 - t0:6.1f}s (max {fap[k].max():8.1f}), "
              f"ADD_est {t2 - t1:5.1f}s (max {add_est[k].max():6.2f}), "
              f"ADD_exact {t3 - t2:5.1f}s (max {add_exact[k].max():6.2f})", flush=True)

    return fap, add_est, add_exact, q_last


def load_cache():
    d = np.load(CACHE)
    if not np.array_equal(d["R_list"], np.array(R_LIST)):
        raise RuntimeError(f"cache {CACHE} was produced for other ranks")
    return d["fap"], d["add_est"], d["add_exact"], d["q_last"]


def save_cache(fap, add_est, add_exact, q_last):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    np.savez(CACHE, fap=fap, add_est=add_est, add_exact=add_exact, q_last=q_last,
             h_exp=H_EXP, R_list=np.array(R_LIST), test_size=TEST_SIZE,
             add_test_size=TEST_SIZE * ADD_MULTI, forget_val=FORGOT_VAL, seed=SEED,
             set_num=SET_NUM_RANK_DEF)
    print(f"[fig11] cached simulation in {CACHE}")


def make_figure(fap, add_est, add_exact, out_dir):
    fig, axes = new_figure(nrows=1, ncols=2, width="double", height=2.9, sharey=True)

    for k, R in enumerate(R_LIST):
        # colours come from the default cycle in the order R = 5, 6, 7, 8 -> C0, C1, C2, C3,
        # exactly as the source cell produced them (both axes advance identically)
        label = r"$R_{\rho}$" + f" = {R}"
        axes[0].plot(fap[k], add_est[k], label=label)      # panel a: estimated q_hat_t
        axes[1].plot(fap[k], add_exact[k], label=label)    # panel b: exact q

    for ax in axes:
        ax.set_xlim(0, MAX_FAP)          # y limits autoscaled (sharey), as in the notebook
        ax.grid(linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    # axes occupy the upper part of the canvas; the shared FAP label and the plot key
    # (one row of four, below the axes, exactly as in the source cell) sit in the strip below
    fig.tight_layout(rect=[0.035, 0.175, 1, 1])
    fig.supxlabel("FAP", y=0.115)
    fig.supylabel("ADD", x=0.012)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=4, frameon=False)
    label_panels(fig, axes)
    return save_figure(fig, 11, out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="directory for Fig11.png / Fig11.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help=f"re-plot from the cached simulation in {CACHE} instead of simulating")
    args = ap.parse_args()

    np.random.seed(SEED)     # the batched routines take an explicit rng; this pins the rest
    if args.plot_only:
        fap, add_est, add_exact, q_last = load_cache()
        print(f"[fig11] loaded cached simulation from {CACHE}")
    else:
        t0 = time.time()
        fap, add_est, add_exact, q_last = simulate()
        print(f"[fig11] simulation took {time.time() - t0:.1f}s")
        save_cache(fap, add_est, add_exact, q_last)

    # numbers quoted in the self-check: ADD interpolated at a few FAP positions
    for x in (500.0, 1000.0, 2500.0, 5000.0):
        row_e = [np.interp(x, fap[k], add_est[k]) for k in range(len(R_LIST))]
        row_x = [np.interp(x, fap[k], add_exact[k]) for k in range(len(R_LIST))]
        print(f"[fig11] FAP = {x:6.0f}  ADD_est  " +
              "  ".join(f"R={R}: {v:6.2f}" for R, v in zip(R_LIST, row_e)))
        print(f"[fig11] FAP = {x:6.0f}  ADD_exact" +
              "  ".join(f"R={R}: {v:6.2f}" for R, v in zip(R_LIST, row_x)))

    png = make_figure(fap, add_est, add_exact, os.path.abspath(args.out_dir))
    print(f"[fig11] wrote {png} and {png[:-4]}.pdf")


if __name__ == "__main__":
    main()
