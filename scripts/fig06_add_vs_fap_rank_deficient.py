#!/usr/bin/env python3
"""Figure 6 - detection delay versus false alarm period for rank-deficient pre-change states.

Reproduces the published figure ``known_ADDvsFAP_R.png`` (notebook
``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho.ipynb``, cell 56;
helper cells 1, 3, 4, 9, 11, 13, 15, 17, 20, 28, 31).

Average detection delay (ADD) versus false alarm period (FAP) under the
maximum-KL measurement (solid) and the Helstrom measurement (dashed), for N = 8
and pre-change rank R_rho = 5 (C0 blue), 6 (C1 orange), 7 (C2 green) and
8 (C3 red).  Each pair of curves uses the *same* state pair (rho, sigma) that
Fig. 5 uses for that rank: ``rho_8_rank_R_set_500.npy`` index 9 and the single
full-rank ``sigma_8.npy``.

Everything is computed here; no FAP/ADD arrays exist on disk for this figure.
Per rank R the script

  1. runs the support-projected rank-aware PGA (``pga.pgd_rank_aware``, the
     verbatim port of ``von_neumann_pgd_backtracking(rho, sigma, n, r, ...)``
     from the sibling notebook ``adap_grad_rank_defic_rho_copy.ipynb`` cell 27 --
     the defining cell was deleted from the notebook that draws this figure) for
     1000 iterations with tol = 0 and U0 = eigenvectors of rho, giving the
     maximum-KL outcome distributions p (sums to 1) and q (sums to 1 - q_0,
     q_0 = 0.394 / 0.282 / 0.097 / 0.000 for R = 5 / 6 / 7 / 8);
  2. sweeps a per-rank CUSUM threshold grid of 100 points -- linspace(0, 4),
     (0, 4.5), (0, 6), (0, 8) for R = 5, 6, 7, 8 -- with the batched Monte-Carlo
     routines of :mod:`quantum_qcd.cusum_nb`:
       FAP: ``cusum_fap_batch`` (cell 3).  It normalises ONLY p, so the
            log-likelihood ratio uses the sub-normalised q of a rank-deficient
            pair; normalising q would change every R < 8 curve.  Runs are
            censored at max_time = 20000.
       ADD: ``cusum_add_batch_rank`` (cell 4) for R < 8, fed with the appended
            null-space outcome ``np.append(q, 1 - sum(q))``, which fires every
            threshold at once; ``cusum_add_batch`` (cell 3) for R = 8, where the
            null outcome has zero mass;
  3. repeats the FAP/ADD sweep for the Helstrom (binary) measurement of cell 20,
     always on the threshold grid linspace(0, 8, 100), and draws it dashed in the
     colour of the solid curve of the same rank.

Monte Carlo: test_size = 2**11 = 2048 runs per curve, exactly as the notebook.
The notebook set no seed; here a single ``np.random.RandomState(0)`` is threaded
through the eight sweeps in the notebook's order, which is equivalent to putting
``np.random.seed(0)`` at the top of cell 56 (see the module notes of
``cusum_nb``), so the figure is bit-reproducible.

Data (all read from ``code_release/data/``):
    adap_grad_jeremy/sigma_8.npy                        post-change state (all four ranks)
    adap_grad_jeremy/rho_8_rank_{5,6,7,8}_set_500.npy   pre-change state sets, index 9 used
    X/X_8_rho.npy, X/X_8_sigma.npy                      generator seeds; loaded exactly as
                                                        the notebook does with
                                                        load_exist=True, i.e. never used

Runtime is about 1 minute on two CPU threads (four PGA runs ~1 s total, eight
CUSUM sweeps dominated by the four maximum-KL FAP sweeps at ~7 s each).  The
eight (100,) curve pairs are cached to
``data/derived/fig06_add_vs_fap_rank_deficient.npz`` so ``--plot-only`` re-plots
without simulating.

Usage:
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        PYTHONPATH=src python3 scripts/fig06_add_vs_fap_rank_deficient.py \
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

from quantum_qcd import cusum_nb, pga           # noqa: E402  (qualified imports:
from quantum_qcd.quantum_utils import check_complex  # noqa: E402  channel.alg1 differs)
from figstyle import new_figure, save_figure    # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "derived", "fig06_add_vs_fap_rank_deficient.npz")

# ---- parameters of notebook cells 31 and 56 (unchanged) ---------------------
N = 8                       # Hilbert-space dimension
N_SETS = 500                # state sets on disk: rho_8_rank_R_set_500.npy
R_LIST = [5, 6, 7, 8]       # pre-change ranks -> C0, C1, C2, C3
MAX_ITER = 1000             # PGA iterations (deterministic; tol = 0)
TEST_SIZE = 2 ** 11         # 2048 Monte-Carlo runs per curve
SET_NUM_DEFICIT = 9         # index of the state pair inside each 500-state set
MAX_FAP = 5000              # x limit, and the FAP at which the ADDs are printed
SEED = 0                    # not in the notebook; see the module docstring

# per-rank threshold grids of cell 56 (the Helstrom sweep always uses (0, 8))
THRESHOLDS = {
    5: np.linspace(0, 4, 100),
    6: np.linspace(0, 4.5, 100),
    7: np.linspace(0, 6, 100),
    8: np.linspace(0, 8, 100),
}
THRESHOLD_HEL = np.linspace(0, 8, 100)


def load_states():
    """Load X_rho/X_sigma (unused, as in the notebook), sigma and the four rho sets."""
    # generate_X_single(N, load_exist=True): returned by the notebook but never
    # used downstream, because generate_sigma / generate_rho_set_rank also run
    # with load_exist=True and simply read the .npy files.
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


def distribution_binary_measurement(n, rho, sigma):
    """Helstrom (binary) measurement outcome distributions.

    Verbatim transcription of cell 20 of the source notebook: the POVM is
    {Q_1, I - Q_1} with Q_1 the projector onto the positive eigenspace of
    rho - sigma.  There is no equivalent in ``quantum_qcd`` --
    ``quantum_utils.povm_positive_eigen`` builds *rank-one* projectors from the
    positive eigenvalues of a single state, which is a different POVM -- so the
    cell is reproduced here, including its use of ``np.linalg.eig`` (not
    ``eigh``) and of ``check_complex`` with tol = 1e-6 (cell 17, re-used from
    ``quantum_qcd.quantum_utils``).
    """
    X = rho - sigma
    eigenvalues, eigenvectors = np.linalg.eig(X)
    eigenvectors = np.matrix(eigenvectors)

    Q_1 = np.zeros((n, n), dtype=complex)
    for i in range(n):
        if eigenvalues[i] > 0:
            Q_1 += eigenvectors[:, i].dot(eigenvectors[:, i].conj().T)
    Q_2 = np.eye(n) - Q_1

    p_dist = np.array([np.trace(Q_1.dot(rho)), np.trace(Q_2.dot(rho))])
    q_dist = np.array([np.trace(Q_1.dot(sigma)), np.trace(Q_2.dot(sigma))])

    p_dist, q_dist = check_complex(p=p_dist, q=q_dist, tol=1e-6)

    return p_dist, q_dist, Q_1, Q_2


def simulate():
    """Re-run cell 56: eight (100,) FAP/ADD curve pairs.  Returns a dict of arrays."""
    # The optimisation below is deterministic (states from disk, U0 = eigenvectors
    # of rho for R = N and I_R for R < N, deterministic Armijo line search); the
    # seeds are set so the script stays reproducible if that ever changes.
    np.random.seed(SEED)
    th.manual_seed(SEED)
    # One legacy RandomState threaded through all eight sweeps, in the notebook's
    # order -- equivalent to np.random.seed(SEED) at the top of cell 56.
    rng = np.random.RandomState(SEED)

    sigma, rho_sets = load_states()
    th_sigma = th.tensor(sigma, dtype=th.cfloat)

    data = {}
    for R in R_LIST:
        rho = rho_sets[R][SET_NUM_DEFICIT, :, :]
        th_rho = th.tensor(rho, dtype=th.cfloat)

        eigvals, U0 = th.linalg.eigh(th_rho)

        t0 = time.time()
        # von_neumann_pgd_backtracking(th_rho, th_sigma, N, R, maxiter=1000, tol=0, U0=U0)
        U_opt, hist, p, q = pga.pgd_rank_aware(
            th_rho, th_sigma, N, R, maxiter=MAX_ITER, tol=0, U0=U0
        )
        t_pga = time.time() - t0
        print(f"R = {R}, q_0 = {1 - sum(q):.3f}   (PGA {t_pga:.1f} s, f = {hist[-1]:.4f})")

        threshold = THRESHOLDS[R]

        t0 = time.time()
        if R != N:
            # cusum_fap_batch reads only the first len(p) entries of q and does
            # NOT normalise it: the sub-normalised q is deliberate.
            fap = cusum_nb.cusum_fap_batch(TEST_SIZE, p, q, h_vec=threshold, seed=rng)
            q_add = cusum_nb.append_null_outcome(q)     # np.append(q, 1 - sum(q))
            add = cusum_nb.cusum_add_batch_rank(TEST_SIZE, p, q_add, h_vec=threshold,
                                                seed=rng)
        else:
            fap = cusum_nb.cusum_fap_batch(TEST_SIZE, p, q, h_vec=threshold, seed=rng)
            add = cusum_nb.cusum_add_batch(TEST_SIZE, p, q, h_vec=threshold, seed=rng)
        t_kl = time.time() - t0

        index_5000 = np.where(fap > MAX_FAP)[0][0]
        add_opt = add[index_5000]

        # --- Helstrom measurement on the same state pair -----------------------
        p_hel, q_hel, _, _ = distribution_binary_measurement(N, rho, sigma)
        print(f"R = {R}, p = {p_hel}, q = {q_hel}")

        t0 = time.time()
        fap_hel = cusum_nb.cusum_fap_batch(TEST_SIZE, p_hel, q_hel,
                                           h_vec=THRESHOLD_HEL, seed=rng)
        add_hel_curve = cusum_nb.cusum_add_batch(TEST_SIZE, p_hel, q_hel,
                                                 h_vec=THRESHOLD_HEL, seed=rng)
        t_hel = time.time() - t0

        index_5000 = np.where(fap_hel > MAX_FAP)[0][0]
        add_hel = add_hel_curve[index_5000]

        print(f"FAP = {MAX_FAP}, ADD_opt = {add_opt:.3f}, ADD_Hel = {add_hel:.3f}"
              f"   (CUSUM {t_kl:.1f} s + {t_hel:.1f} s)\n")

        data[f"R{R}_fap"] = fap
        data[f"R{R}_add"] = add
        data[f"R{R}_fap_hel"] = fap_hel
        data[f"R{R}_add_hel"] = add_hel_curve
        data[f"R{R}_threshold"] = threshold
        data[f"R{R}_p"] = np.asarray(p, dtype=float)
        data[f"R{R}_q"] = np.asarray(q, dtype=float)
        data[f"R{R}_p_hel"] = np.asarray(p_hel, dtype=float)
        data[f"R{R}_q_hel"] = np.asarray(q_hel, dtype=float)
    data["threshold_hel"] = THRESHOLD_HEL
    return data


def make_figure(data):
    fig, axes = new_figure(ncols=1, width="medium")
    ax = axes[0]
    for k, R in enumerate(R_LIST):
        color = f"C{k}"          # notebook: automatic cycle -> C0 C1 C2 C3, one per rank
        ax.plot(data[f"R{R}_fap"], data[f"R{R}_add"], color=color,
                label=r"$R_{\rho}$" + f" = {R}, maximum-KL")
        # dashed Helstrom curve in the colour of the solid curve of the same rank
        ax.plot(data[f"R{R}_fap_hel"], data[f"R{R}_add_hel"], color=color,
                linestyle="--", label=r"$R_{\rho}$" + f" = {R}, Helstrom")
    ax.set_xlim(0, MAX_FAP)      # linear axes, exactly as the source cell
    ax.set_ylim(0, 50)
    ax.grid(linestyle="--")
    ax.legend(loc="upper left", handlelength=1.6, labelspacing=0.25,
              borderpad=0.3, handletextpad=0.5)
    ax.set_xlabel("FAP")
    ax.set_ylabel("ADD")
    fig.tight_layout()           # no title / suptitle (journal rule)
    return fig, axes


def report(data):
    """Print the plotted values near a few FAP positions for the self-check."""
    x_check = [500, 1000, 2000, 3000, 4000, 5000]
    print(f"FAP positions checked: {x_check}")
    for k, R in enumerate(R_LIST):
        for tag, name in (("", "maximum-KL"), ("_hel", "Helstrom  ")):
            fap = data[f"R{R}_fap{tag}"]
            add = data[f"R{R}_add{tag}"]
            vals = ", ".join(f"{np.interp(x, fap, add):.3f}" for x in x_check)
            print(f"  R_rho = {R} (C{k}) {name}: ADD = {vals}"
                  f"   [FAP range {fap[0]:.1f} .. {fap[-1]:.1f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"),
                    help="directory for Fig6.png / Fig6.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help=f"re-plot from {CACHE} without re-running the simulation")
    args = ap.parse_args()

    if args.plot_only and os.path.exists(CACHE):
        with np.load(CACHE) as z:
            data = {k: z[k] for k in z.files}
        print(f"loaded cached curves from {CACHE}")
    else:
        if args.plot_only:
            print(f"no cache at {CACHE}; simulating")
        t0 = time.time()
        data = simulate()
        print(f"simulation took {time.time() - t0:.1f} s")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
        print(f"cached curves to {CACHE}")

    report(data)
    fig, _ = make_figure(data)
    png = save_figure(fig, 6, out_dir=args.out_dir)
    print("wrote", png)
    print("wrote", os.path.splitext(png)[0] + ".pdf")


if __name__ == "__main__":
    main()
