#!/usr/bin/env python3
"""Figure 5 - convergence of projected gradient ascent (Alg. 1) for rank-deficient pre-change states.

Reproduces the published figure ``rho_sigma_set_R.png`` (notebook
``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho.ipynb``, cell 37;
helper cells 1, 9, 11, 13, 15, 28, 31, 35).

Objective f(V) of Alg. 1 versus iteration count over 800 iterations, for N = 8 and
one fixed pre-change state rho of rank R_rho = 5 (C0 blue), 6 (C1 orange),
7 (C2 green) and 8 (C3 red), each paired with the *same* full-rank post-change
state sigma.

The four curves are recomputed here (no saved curves exist).  The computation is
fully deterministic: the states come from disk and the iterate is initialised at
V^(0) = I_R, so there is no RNG anywhere.  ``alg1`` is the verbatim port of the
notebook's cell 35 that lives in :mod:`quantum_qcd.pga`; note that
``quantum_qcd.channel`` defines a *different* ``alg1`` with the same signature but
a four-value return, hence the qualified ``pga.alg1`` call below.

Because R_rho < N truncates the outcome distribution q (it sums to 1 - q_0 < 1),
f(V) is negative for R_rho = 5 and 6.  That is expected, not a bug; see the
manuscript text around Fig. 5.

Data (all read from ``code_release/data/``):
    adap_grad_jeremy/sigma_8.npy                  post-change state, shared by all four runs
    adap_grad_jeremy/rho_8_rank_{5,6,7,8}_set_500.npy   pre-change state sets, index 9 used
    X/X_8_rho.npy, X/X_8_sigma.npy                seeds of the state generators; loaded
                                                  exactly as the notebook does with
                                                  load_exist=True, i.e. never used

Runtime is about 1-2 minutes for all four curves on two CPU threads; the resulting
(4, 800) history is cached to ``data/derived/fig05_pga_convergence.npz`` so that
``--plot-only`` re-plots without re-running the optimisation.

Usage:
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        PYTHONPATH=src python3 scripts/fig05_pga_convergence_rank_deficient.py \
        [--out-dir DIR] [--plot-only]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch as th
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))   # quantum_qcd package
sys.path.insert(0, HERE)                        # figstyle

from quantum_qcd import pga                     # noqa: E402  (qualified: channel.alg1 differs)
from figstyle import new_figure, save_figure    # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "derived", "fig05_pga_convergence.npz")

# ---- parameters of notebook cells 31 and 37 (unchanged) -------------------
N = 8                       # Hilbert-space dimension
N_SETS = 500                # state sets on disk: rho_8_rank_R_set_500.npy
R_LIST = [5, 6, 7, 8]       # pre-change ranks -> C0, C1, C2, C3
MAX_ITER = 800              # deterministic; tol = 0 disables early stopping
SET_NUM_DEFICIT = 9         # index of the state pair used inside each 500-state set
ALG1_KW = dict(             # cell 37's call, verbatim
    alpha_init=1e-1,
    min_step=1e-5,
    min_norm=1e-6,
    armijo_alpha=1e-4,
    armijo_beta=0.9,
    maxiter=MAX_ITER,
    tol=0,
    V0="Identity",
)


def load_states():
    """Load X_rho/X_sigma (unused, as in the notebook), sigma and the four rho sets."""
    # generate_X_single(N, load_exist=True) -- returned but never used downstream,
    # because generate_sigma / generate_rho_set_rank also run with load_exist=True.
    _X_rho = np.load(os.path.join(DATA_DIR, "X", f"X_{N}_rho.npy"))
    _X_sigma = np.load(os.path.join(DATA_DIR, "X", f"X_{N}_sigma.npy"))
    # generate_sigma(X_sigma, N, load_exist=True): the SAME full-rank sigma for every R.
    sigma = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy", f"sigma_{N}.npy"))
    # generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)
    rho_sets = {}
    for R in R_LIST:
        path = os.path.join(DATA_DIR, "adap_grad_jeremy",
                            f"rho_{N}_rank_{R}_set_{N_SETS}.npy")
        arr = np.load(path)
        if arr.shape != (N_SETS, N, N):
            raise RuntimeError(f"{path}: unexpected shape {arr.shape}")
        rho_sets[R] = arr
    return sigma, rho_sets


def simulate():
    """Re-run cell 37: four alg1 histories of length MAX_ITER.  Returns (4, 800) array."""
    # No RNG is used anywhere below (states from disk, V^(0) = I_R, deterministic
    # Armijo line search).  The seeds are set only so the script is reproducible if
    # the code ever grows a random initialisation.
    np.random.seed(0)
    th.manual_seed(0)

    sigma, rho_sets = load_states()
    th_sigma = th.tensor(sigma, dtype=th.cfloat)

    hist = np.zeros((len(R_LIST), MAX_ITER))
    for k, R in enumerate(R_LIST):
        rho = rho_sets[R][SET_NUM_DEFICIT, :, :]
        th_rho = th.tensor(rho, dtype=th.cfloat)

        t0 = time.time()
        Q_opt, obj, p, q = pga.alg1(th_rho, th_sigma, N, R, **ALG1_KW)
        dt = time.time() - t0

        obj = np.asarray([float(v) for v in obj])
        if obj.size != MAX_ITER:
            raise RuntimeError(f"R = {R}: history length {obj.size} != {MAX_ITER}")
        hist[R - 5, :] = obj[-MAX_ITER:]

        # the notebook's own printed diagnostics
        print(f"R_rho = {R}  ({dt:.1f} s)")
        print(f"KL start -> end : {obj[0]:.4f} -> {obj[-1]:.4f}")
        print(f"p distribution: {p}")
        print(f"q distribution: {q}, q_0 = {1 - sum(q)}")
    return hist


def make_figure(hist):
    fig, axes = new_figure(ncols=1, width="single")
    ax = axes[0]
    for k, R in enumerate(R_LIST):
        # no explicit colour -> default cycle: R=5 C0 blue, 6 C1 orange, 7 C2 green, 8 C3 red;
        # solid, no marker, x = implicit index 0..799, exactly as the source cell
        ax.plot(hist[R - 5, :], label=r"$R_{\rho}$" + f" = {R}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("f(V)")
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True)
    ax.legend()          # default 'best' -> lower left, as in the published figure
    fig.tight_layout()   # no title / suptitle (journal rule)
    return fig, axes


def report(hist):
    """Print the plotted values at a few iteration indices for the self-check."""
    idx = [0, 1, 100, 200, 400, 600, 799]
    print(f"iterations checked: {idx}")
    print(f"history shape: {hist.shape}")
    for k, R in enumerate(R_LIST):
        vals = ", ".join(f"{hist[R - 5, i]:.4f}" for i in idx)
        print(f"  R_rho = {R} (C{k}): {vals}")
    print(f"y data range: {hist.min():.4f} .. {hist.max():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"),
                    help="directory for Fig5.png / Fig5.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help=f"re-plot from {CACHE} without re-running the optimisation")
    args = ap.parse_args()

    if args.plot_only and os.path.exists(CACHE):
        with np.load(CACHE) as z:
            hist = z["hist"]
            if list(z["R_list"]) != R_LIST:
                raise RuntimeError(f"{CACHE}: R_list {list(z['R_list'])} != {R_LIST}")
        print(f"loaded cached histories from {CACHE}")
    else:
        if args.plot_only:
            print(f"no cache at {CACHE}; simulating")
        hist = simulate()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, hist=hist, R_list=np.array(R_LIST),
                            max_iter=MAX_ITER, set_num_deficit=SET_NUM_DEFICIT, N=N)
        print(f"cached histories to {CACHE}")

    report(hist)
    fig, _ = make_figure(hist)
    png = save_figure(fig, 5, out_dir=args.out_dir)
    print("wrote", png)
    print("wrote", os.path.splitext(png)[0] + ".pdf")


if __name__ == "__main__":
    main()
