#!/usr/bin/env python3
"""Figure 14 - convergence of the rank minimization and joint alternating optimization steps.

Reproduces the published figure ``obj.png`` (notebook
``probe_known/probing_known.ipynb``, plot cell 117; helper cells 2, 22, 79, 82, 84, 97, 109).

Three panels, all computed here (the notebook saves no curves; every run is seeded
and deterministic, so the panels are reproduced exactly rather than replotted):

  a  Log-det objective of the rank minimization step versus iteration, for
     m_1 = 3 (C0 blue), 4 (C1 orange), 6 (C2 green), 12 (C3 red), 13 (C4 purple),
     2000 Adam steps each; the legend also reports the resulting rank R* of the
     pre-change channel output (2, 3, 4, 5, 6).            [cell 97]
  b  KL divergence of the joint alternating optimization for a *known*
     post-change channel versus iteration, m_1 = 13 (C0 blue) and 25 (C1 orange),
     200 outer iterations with 5 inner measurement steps.  [cell 84 on cell 79]
  c  Sensitivity sum_i 1/lambda_i(N(rho)) of the joint alternating optimization
     for an *unknown* post-change channel versus iteration, m_1 = 13 (C0 blue)
     and 25 (C1 orange), 500 iterations.                   [cell 109]

Everything runs through :mod:`quantum_qcd.channel`, the verbatim port of those
cells.  Note that ``channel.alg1`` (used internally by panels a and b) is a
different function from ``pga.alg1`` despite the identical signature, which is why
the module is imported qualified.

Panel b deliberately uses ``joint_optimization_known(..., x_line_search=True)``:
probing_known defines ``von_neumann_pgd_back_probing_alg1`` twice with the same
execution count (cells 79 and 82).  Cell 84 - the run whose output is the
published panel - executed under cell 79's backtracking probe step and plateaus
at KL = 1.85; cell 82's fixed unit step plateaus at 1.35 and is *not* this panel.
See the ``joint_optimization_known`` docstring.

Fixed parameters (unchanged from the notebook): N_a = 8, N_b = 6, Kraus seeds
11 (pre-change) and 12 (post-change); panel a th.manual_seed(10) once before the
m_1 loop, Adam lr = 0.002, eps = 1e-6, rank threshold 1e-5; panel b probe from
generate_pure_state(8, num_seed=5); panel c probe from data/X/X_8_rho.npy.

Data (all read from ``code_release/data/``):
    X/X_8_rho.npy      probe-state factor X that starts panel c (cell 109's
                       ``generate_X_single(8, load_exist=True)``)
    X/X_8_sigma.npy    returned by the same loader and discarded by the notebook;
                       kept so the data set is complete

Runtime is about 1 minute on two CPU threads; the curves are cached to
``data/derived/fig14_channel_convergence.npz`` so ``--plot-only`` re-plots
without re-running the optimisations.

Usage:
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        PYTHONPATH=src python3 scripts/fig14_channel_convergence.py \
        [--out-dir DIR] [--plot-only]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))   # quantum_qcd package
sys.path.insert(0, HERE)                        # figstyle

from quantum_qcd import channel                 # noqa: E402  (qualified: pga.alg1 differs)
from figstyle import new_figure, label_panels, save_figure   # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "derived", "fig14_channel_convergence.npz")

# ---- parameters of notebook cells 97 / 84 / 109 (unchanged) ---------------
N_A, N_B = 8, 6                     # channel input / output dimension
SEED_K, SEED_L = 11, 12             # pre- / post-change Kraus seeds
M_LIST_MIN = [3, 4, 6, 12, 13]      # panel a sweep (cell 97)
M_LIST_JOINT = [13, 25]             # panels b, c sweep (cells 84, 109)
MAX_ITER_MIN = 2000                 # Adam steps of the rank minimization
MAX_ITER_U_MIN = 1000               # alg1 steps after the rank minimization
MAX_ITER_KNOWN = 200                # outer iterations of cell 84
MAX_ITER_U_KNOWN = 5                # inner measurement steps of cell 84
MAX_ITER_UNKNOWN = 500              # iterations of cell 109


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def simulate():
    """Run the three seeded optimisations of cells 97, 84 and 109."""
    t0 = time.time()

    # ---- panel a: cell 97, one th.manual_seed(10) stream for the whole sweep
    print(f"[a] rank minimization sweep, m1 = {M_LIST_MIN}, "
          f"{MAX_ITER_MIN} Adam steps each ...")
    sweep = channel.rank_minimization_sweep(m_list=M_LIST_MIN, N_a=N_A, N_b=N_B,
                                            max_iter=MAX_ITER_MIN,
                                            seed_K=SEED_K, seed_L=SEED_L)
    hist_a = np.array([sweep[m][1] for m in M_LIST_MIN], dtype=float)
    ranks = np.array([sweep[m][2] for m in M_LIST_MIN], dtype=int)
    for m, r in zip(M_LIST_MIN, ranks):
        print(f"    m1 = {m:2d}: R* = {r}, log-det {sweep[m][1][0]:8.3f} "
              f"-> {sweep[m][1][-1]:8.3f}")
    print(f"    [{time.time() - t0:.1f} s]")

    # cell 97 continues with alg1 on the rank-reduced probe; it is not plotted
    # in Fig. 14 (it produces the p, q of Fig. 15), so it is not run here.

    # ---- panel b: cell 84 (probe step with line search, cell 79's definition)
    hist_b = []
    for m in M_LIST_JOINT:
        t1 = time.time()
        hist, p, q = channel.joint_optimization_known(
            N_a=N_A, N_b=N_B, m1=m, max_iter=MAX_ITER_KNOWN,
            max_iter_U=MAX_ITER_U_KNOWN, seed_K=SEED_K, seed_L=SEED_L)
        hist_b.append(hist)
        print(f"[b] known post-change, m1 = {m:2d}: KL {hist[0]:.4f} -> "
              f"{hist[-1]:.4f} over {len(hist)} iterations "
              f"[{time.time() - t1:.1f} s]")
        print(f"    p = {np.array2string(p, precision=6)}")
        print(f"    q = {np.array2string(q, precision=6)}")

    # ---- panel c: cell 109, probe initialised from data/X/X_8_rho.npy
    hist_c = []
    for m in M_LIST_JOINT:
        t1 = time.time()
        hist, p, q = channel.joint_optimization_unknown(
            N_a=N_A, N_b=N_B, m1=m, max_iter=MAX_ITER_UNKNOWN,
            seed_K=SEED_K, seed_L=SEED_L,
            data_dir=os.path.join(DATA_DIR, "X"))
        hist_c.append(hist)
        print(f"[c] unknown post-change, m1 = {m:2d}: sensitivity {hist[0]:.4f} "
              f"-> {hist[-1]:.4f} over {len(hist)} iterations "
              f"[{time.time() - t1:.1f} s]")

    # cell 109 can break early once min eig(N(rho)) < 1e-5; pad so the curves
    # live in one array and remember the true lengths.
    len_b = np.array([len(h) for h in hist_b], dtype=int)
    len_c = np.array([len(h) for h in hist_c], dtype=int)
    arr_b = np.full((len(hist_b), len_b.max()), np.nan)
    arr_c = np.full((len(hist_c), len_c.max()), np.nan)
    for k, h in enumerate(hist_b):
        arr_b[k, :len(h)] = h
    for k, h in enumerate(hist_c):
        arr_c[k, :len(h)] = h

    print(f"total simulation time {time.time() - t0:.1f} s")
    return hist_a, ranks, arr_b, len_b, arr_c, len_c


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def make_figure(hist_a, ranks, hist_b, len_b, hist_c, len_c):
    """Rebuild cell 117's 1x3 panels on a large canvas, without the titles."""
    # one row, three equal panels with the proportions of the published figure
    # (a 16 x 4.7 in canvas scaled to the line width): wide canvas, wide gaps,
    # fonts enlarged so that they print at about 5 pt when scaled to 7.3 in
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9, "lines.linewidth": 1.5})
    fig, axes = new_figure(nrows=1, ncols=3, width=14.0, height=4.2)

    # panel a - cell 117's first loop: colours follow the default cycle in the
    # order m1 = 3, 4, 6, 12, 13 -> C0 blue, C1 orange, C2 green, C3 red, C4 purple
    for k, m in enumerate(M_LIST_MIN):
        axes[0].plot(hist_a[k], label=rf"$m_1$ = {m}, $R^*$ = {ranks[k]}")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Log-Det")
    axes[0].legend(loc="lower right", ncol=1, handlelength=2.0, labelspacing=0.35,
                   borderpad=0.3, handletextpad=0.5, borderaxespad=0.3)

    # panels b and c - m1 = 13 -> C0 blue, 25 -> C1 orange
    for k, m in enumerate(M_LIST_JOINT):
        axes[1].plot(hist_b[k, :len_b[k]], label=rf"$m_1$ = {m}")
        axes[2].plot(hist_c[k, :len_c[k]], label=rf"$m_1$ = {m}")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("KL Divergence")
    axes[1].legend(loc="lower right", handlelength=1.3, labelspacing=0.25,
                   borderpad=0.3, handletextpad=0.5, borderaxespad=0.3)

    axes[2].set_xlabel("Iterations")
    axes[2].set_ylabel("Sensitivity")
    axes[2].legend(loc="upper left", handlelength=1.3, labelspacing=0.25,
                   borderpad=0.3, handletextpad=0.5, borderaxespad=0.3)

    for ax in axes:
        ax.grid(True, linestyle="--")     # as the notebook; linear axes throughout

    fig.tight_layout(w_pad=4.0)           # no titles / suptitle (journal rule)
    label_panels(fig, axes, size=13)
    return fig, axes


def report(hist_a, ranks, hist_b, len_b, hist_c, len_c):
    """Print the plotted values at a few x positions for the self-check."""
    idx_a = [0, 100, 250, 500, 1000, 1500, 1999]
    print(f"\npanel a  log-det, iterations {idx_a}")
    for k, m in enumerate(M_LIST_MIN):
        vals = ", ".join(f"{hist_a[k, i]:8.3f}" for i in idx_a)
        print(f"  m1 = {m:2d} (C{k}, R* = {ranks[k]}): {vals}")
    print(f"  panel a y range: {hist_a.min():.3f} .. {hist_a.max():.3f}; "
          f"x range: 0 .. {hist_a.shape[1] - 1}")

    idx_b = [0, 10, 25, 50, 75, 100, 150, 199]
    print(f"\npanel b  KL divergence, iterations {idx_b}")
    for k, m in enumerate(M_LIST_JOINT):
        vals = ", ".join(f"{hist_b[k, i]:.4f}" for i in idx_b)
        print(f"  m1 = {m:2d} (C{k}): {vals}")
    print(f"  panel b y range: {np.nanmin(hist_b):.4f} .. {np.nanmax(hist_b):.4f}; "
          f"lengths {[int(v) for v in len_b]}")

    idx_c = [0, 50, 100, 150, 200, 250, 300, 400, 499]
    print(f"\npanel c  sensitivity, iterations {idx_c}")
    for k, m in enumerate(M_LIST_JOINT):
        vals = ", ".join(f"{hist_c[k, i]:9.3f}" for i in idx_c)
        print(f"  m1 = {m:2d} (C{k}): {vals}")
    print(f"  panel c y range: {np.nanmin(hist_c):.3f} .. {np.nanmax(hist_c):.3f}; "
          f"lengths {[int(v) for v in len_c]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"),
                    help="directory for Fig14.png / Fig14.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help=f"re-plot from {CACHE} without re-running the optimisations")
    args = ap.parse_args()

    if args.plot_only and os.path.exists(CACHE):
        with np.load(CACHE) as z:
            hist_a, ranks = z["hist_a"], z["ranks"]
            hist_b, len_b = z["hist_b"], z["len_b"]
            hist_c, len_c = z["hist_c"], z["len_c"]
            if list(z["m_list_min"]) != M_LIST_MIN or list(z["m_list_joint"]) != M_LIST_JOINT:
                raise RuntimeError(f"{CACHE}: m1 lists do not match this script")
        print(f"loaded cached curves from {CACHE}")
    else:
        if args.plot_only:
            print(f"no cache at {CACHE}; simulating")
        hist_a, ranks, hist_b, len_b, hist_c, len_c = simulate()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, hist_a=hist_a, ranks=ranks,
                            hist_b=hist_b, len_b=len_b,
                            hist_c=hist_c, len_c=len_c,
                            m_list_min=np.array(M_LIST_MIN),
                            m_list_joint=np.array(M_LIST_JOINT),
                            N_a=N_A, N_b=N_B, seed_K=SEED_K, seed_L=SEED_L)
        print(f"cached curves to {CACHE}")

    report(hist_a, ranks, hist_b, len_b, hist_c, len_c)
    fig, _ = make_figure(hist_a, ranks, hist_b, len_b, hist_c, len_c)
    png = save_figure(fig, 14, out_dir=args.out_dir)
    print("\nwrote", png)
    print("wrote", os.path.splitext(png)[0] + ".pdf")


if __name__ == "__main__":
    main()
