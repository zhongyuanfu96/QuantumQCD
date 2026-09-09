#!/usr/bin/env python3
"""Figure 7: conditional average detection delay (ADD) versus false alarm period
(FAP) under the maximum-KL measurement, for rank-deficient pre-change states.

For N = 8 and pre-change rank R_rho = 5, 6, 7 the measurement is optimised with
the rank-aware projected gradient ascent (Alg. 1) on the state pair
(rho, sigma) = (rho_8_rank_R_set_500[9], sigma_8) -- the same pairs as Figs. 5
and 6.  Because rho is rank deficient, the post-change state has weight
q_0 = 1 - sum(q) = 0.394 / 0.282 / 0.097 outside the support of rho; that mass
is the extra "null-space" outcome x_t = 0, which is impossible before the change
and therefore raises an alarm at every threshold at once.  The Monte Carlo then
splits the detection delay by the cause of stopping:

  * solid curves  -- runs that stopped on the null-space outcome  (x_t = 0),
  * dashed curves -- runs that stopped because the CUSUM statistic crossed the
                     largest threshold (s_t > h).

For R_rho = 5 the second event never happens in 50 000 runs (the counter keeps
its 1e-10 initial value), so the dashed blue curve is identically zero and lies
on the horizontal axis -- a deliberate artefact of the source cell that the
manuscript legend describes and that must not be "fixed".

Source: notebooks_raw/Quantum_Anomaly_Detection/optimization_known/
        adaptive_gradient/adap_grad_rank_defic_rho.ipynb,
        cell 65 (simulation) and cell 66 (plot); helper cells 1, 3, 9, 11, 13,
        15, 28, 31, 62, 64.  The optimiser called by cell 65,
        von_neumann_pgd_backtracking, was deleted from that notebook while it
        stayed live in the Colab kernel; its definition comes from the sibling
        adap_grad_rank_defic_rho_copy.ipynb cell 27 and is ported as
        quantum_qcd.pga.pgd_rank_aware.  Cell 3's cusum_fap_batch and cell 64's
        cusum_add are ported as quantum_qcd.cusum_nb.cusum_fap_batch and
        quantum_qcd.cusum_nb.cusum_add_cond_exact_batch.

Data (loaded only from code_release/data/):
    data/adap_grad_jeremy/sigma_8.npy
    data/adap_grad_jeremy/rho_8_rank_{5,6,7}_set_500.npy
    data/X/X_8_rho.npy, data/X/X_8_sigma.npy   (the seeds the states were built
        from; loaded and checked for provenance, not used in the computation,
        exactly as generate_X_single(load_exist=True) does in the notebook)

Usage:
    python3 scripts/fig07_add_vs_fap_conditional_known.py [--out-dir DIR] [--plot-only]

Runtime: about 6 minutes (dominated by the three FAP simulations, 50 000 runs
each against the 20 000-step censoring cap).  Results are cached in
data/derived/fig07_conditional_add.npz so --plot-only re-plots instantly.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from quantum_qcd import pga                      # noqa: E402
from quantum_qcd import cusum_nb                 # noqa: E402
from figstyle import new_figure, save_figure     # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "derived", "fig07_conditional_add.npz")

# --- parameters of the source cell (unchanged) -------------------------------
N = 8                    # Hilbert-space dimension
N_SETS = 500             # state-set file size (rho_8_rank_R_set_500.npy)
SET_NUM_DEFICIT = 9      # index of the state pair inside the set
R_LIST = [5, 6, 7]
MAX_ITER = 1000          # PGA iterations (deterministic; never reduced)
TOL = 0.0                # run the full MAX_ITER iterations
TEST_SIZE = 50_000       # Monte-Carlo runs, cell 65
MAX_TIME = 20_000        # censoring cap of cusum_fap_batch (cell 3 default)
N_THRESH = 100
# per-rank threshold grids (cell 65)
THRESHOLDS = {
    5: np.linspace(0, 0.7, N_THRESH),
    6: np.linspace(0, 4.5, N_THRESH),
    7: np.linspace(0, 5.5, N_THRESH),
}
# plot limits (cell 66)
XLIM = (0, 5000)
YLIM = (0, 15)

# seeds: the notebook Monte Carlo was unseeded; everything here is seeded so the
# figure is reproducible.  One stream per (rank, quantity).
SEED_BASE = 0
FAP_SEED = {R: 1000 + R for R in R_LIST}
ADD_SEED = {R: 2000 + R for R in R_LIST}


def optimise_measurement(R):
    """p and q of the maximum-KL measurement for the rank-R pre-change state."""
    sigma = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy", "sigma_8.npy"))
    rho_set = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy",
                                   f"rho_{N}_rank_{R}_set_{N_SETS}.npy"))
    rho = rho_set[SET_NUM_DEFICIT, :, :]

    th_rho = th.tensor(rho, dtype=th.cfloat)
    th_sigma = th.tensor(sigma, dtype=th.cfloat)
    eigvals, U0 = th.linalg.eigh(th_rho)          # U0 is ignored when R < N

    _Q, _hist, p, q = pga.pgd_rank_aware(th_rho, th_sigma, N, R,
                                         maxiter=MAX_ITER, tol=TOL, U0=U0)
    return np.asarray(p, dtype=float), np.asarray(q, dtype=float)


def compute():
    np.random.seed(SEED_BASE)
    th.manual_seed(SEED_BASE)

    # provenance check only: these are the matrices the states were generated
    # from (generate_X_single(N, load_exist=True) in cells 13/15/31).
    for name in ("X_8_rho.npy", "X_8_sigma.npy"):
        _ = np.load(os.path.join(DATA_DIR, "X", name))

    fap_arr = np.zeros((len(R_LIST), N_THRESH))
    add_arr_xt = np.zeros((len(R_LIST), N_THRESH))
    add_arr_st = np.zeros((len(R_LIST), N_THRESH))
    thresh_arr = np.zeros((len(R_LIST), N_THRESH))
    counters = np.zeros((len(R_LIST), 2))
    data = {}

    for R_num, R in enumerate(R_LIST):
        t0 = time.time()
        p, q = optimise_measurement(R)
        # the extra outcome carries the post-change weight outside supp(rho)
        q_add = cusum_nb.append_null_outcome(q)
        threshold = THRESHOLDS[R]
        print(f"R = {R}, q_0 = {1 - sum(q):.3f}")

        # FAP: observations from p, log-likelihood ratio built with the
        # sub-normalised q (only p is normalised inside -- notebook behaviour).
        fap = cusum_nb.cusum_fap_batch(TEST_SIZE, p, q, h_vec=threshold,
                                       max_time=MAX_TIME, seed=FAP_SEED[R])
        # conditional ADD: observations from the augmented q_add.
        add_xt, add_st, counter_xt, counter_st = cusum_nb.cusum_add_cond_exact_batch(
            TEST_SIZE, p, q_add, h_vec=threshold, seed=ADD_SEED[R],
            batch_size=TEST_SIZE)
        print(f"counter_xt = {counter_xt}, counter_st = {counter_st}"
              f"   [{time.time() - t0:.1f} s]\n")

        fap_arr[R_num, :] = fap
        add_arr_xt[R_num, :] = add_xt
        add_arr_st[R_num, :] = add_st
        thresh_arr[R_num, :] = threshold
        counters[R_num, :] = (counter_xt, counter_st)
        data[f"p_R{R}"] = p
        data[f"q_R{R}"] = q

    data.update(fap_arr=fap_arr, add_arr_xt=add_arr_xt, add_arr_st=add_arr_st,
                thresholds=thresh_arr, counters=counters,
                R_list=np.array(R_LIST))
    return data


def load_cache():
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


def make_figure(data, out_dir):
    fap_arr = data["fap_arr"]
    add_arr_xt = data["add_arr_xt"]
    add_arr_st = data["add_arr_st"]

    fig, axes = new_figure(nrows=1, ncols=1, width="medium")
    ax = axes[0]

    for R_num, R in enumerate(R_LIST):
        # solid curve: stopping through the null-space outcome x_t = 0.
        # The colour comes from the default cycle (C0 blue, C1 orange, C2 green);
        # the dashed partner reuses it explicitly, which does not advance the
        # cycle -- exactly as in cell 66.
        curve_xt = ax.plot(fap_arr[R_num, :], add_arr_xt[R_num, :],
                           label=r'$R_{\rho}$' + f' = {R}, ' + r'$x_{{t}}=0$')
        line_color = curve_xt[0].get_color()
        ax.plot(fap_arr[R_num, :], add_arr_st[R_num, :],
                label=r'$R_{\rho}$' + f' = {R}, ' + r'$s_{{t}}>h$',
                linestyle='--', color=line_color)

        # the cell's own diagnostic: ADD at the first threshold whose FAP
        # exceeds 5000 (printed only, the plotted curves are not truncated)
        over = np.where(fap_arr[R_num, :] > XLIM[1])[0]
        if over.size:
            i5 = over[0]
            print(f"R = {R}, ADD_opt_st = {add_arr_st[R_num, i5]:.3f}, "
                  f"ADD_opt_xt = {add_arr_xt[R_num, i5]:.3f}   "
                  f"(FAP = {fap_arr[R_num, i5]:.1f}, threshold index {i5})\n")

    ax.set_ylim(*YLIM)
    ax.set_xlim(*XLIM)
    ax.grid(linestyle='--')
    ax.legend(loc='upper left')
    ax.set_xlabel('FAP')
    ax.set_ylabel('ADD')
    fig.tight_layout(pad=0.3)
    return save_figure(fig, 7, out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from data/derived/fig07_conditional_add.npz")
    args = ap.parse_args()

    if args.plot_only and os.path.exists(CACHE):
        data = load_cache()
        print(f"loaded cached curves from {CACHE}")
    else:
        t0 = time.time()
        data = compute()
        print(f"simulation took {time.time() - t0:.1f} s")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
        print(f"cached curves to {CACHE}")

    png = make_figure(data, args.out_dir)
    print(f"wrote {png} and {png[:-4]}.pdf")


if __name__ == "__main__":
    main()
