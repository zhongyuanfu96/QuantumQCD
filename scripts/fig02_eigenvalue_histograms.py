#!/usr/bin/env python3
"""Figure 2 - eigenvalue distributions of the two random-state ensembles.

Reproduces the published figure ``eigenvalue_histograms.png``: overlaid
percentage-normalised histograms of the eigenvalues of 10^4 independent
full-rank density operators with N = 8, drawn from

  * the Wishart ensemble        (blue,   C0): X ~ complex Ginibre, rho = X^H X / tr(X^H X);
  * the unitary-product ensemble (orange, C1): rho = U diag(p) U^H with U the QR factor of a
    complex Ginibre matrix and p_i = u_i / sum_j u_j, u_i ~ U(0, 1).

Source cell: notebooks_raw/Quantum_Anomaly_Detection/optimization_unknown/kl_sensitivity.ipynb,
cell 35 (num_samples = 10_000, n = 8, bins = 200, weights = 100/len, alpha = 0.6,
labels "Wishart ensemble" / "Unitary-product ensemble", legend upper right, dashed grid).
The notebook cell is unseeded; this script seeds numpy and torch with 0 so the figure is
reproducible.  Journal changes: no title, figstyle sizes/fonts, single-panel width (no panel
letters).

Usage:
    python3 scripts/fig02_eigenvalue_histograms.py               # simulate + plot
    python3 scripts/fig02_eigenvalue_histograms.py --plot-only   # re-plot from the cached npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from quantum_qcd.quantum_states import generate_mixed_state  # noqa: E402  (serial reference path)
from figstyle import new_figure, save_figure  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
CACHE = os.path.join(DERIVED_DIR, "fig02_eigenvalues.npz")
FIG_DIR = os.path.join(_ROOT, "figures")

# parameters of notebook cell 35
N = 8
NUM_SAMPLES = 10_000
BINS = 200
SEED = 0


def _ginibre(num, n):
    """Batch of complex Ginibre matrices, exactly as in the cell: randn + 1j*randn."""
    return np.random.randn(num, n, n) + 1j * np.random.randn(num, n, n)


def sample_wishart_eigs(num_samples=NUM_SAMPLES, n=N):
    """Eigenvalues of ``num_samples`` trace-normalised Wishart matrices (flat array).

    Cell 35's ``generate_pd_matrix(n)``: X = randn(n,n) + 1j*randn(n,n); X = X^H X;
    X /= tr(X); eigenvalues via ``np.linalg.eigvals(...).real``.  Vectorised over samples.
    """
    X = _ginibre(num_samples, n)
    with np.errstate(all="ignore"):  # numpy 2.0 raises spurious FP warnings in complex matmul
        A = np.conj(np.transpose(X, (0, 2, 1))) @ X
    A = A / np.trace(A, axis1=1, axis2=2)[:, None, None]
    return np.linalg.eigvals(A).real.ravel()


def sample_unitary_product_eigs(num_samples=NUM_SAMPLES, n=N, serial=False):
    """Eigenvalues of ``num_samples`` unitary-product states (flat array).

    Cell 35's ``generate_mixed_state(n)``: X = randn + 1j*randn cast to ``th.cfloat``,
    U = qr(X)[0], p = th.rand(n) normalised to sum 1, rho = U diag(p) U^H; eigenvalues via
    ``np.linalg.eigvals(rho.numpy()).real``.  ``serial=True`` uses the library port
    ``quantum_qcd.quantum_states.generate_mixed_state`` (identical construction, the Ginibre
    seed matrix drawn by torch instead of numpy) one state at a time.
    """
    if serial:
        eigs = []
        for _ in range(num_samples):
            rho = generate_mixed_state(n).numpy()
            eigs.extend(np.linalg.eigvals(rho).real)
        return np.asarray(eigs, dtype=float)
    X = th.tensor(_ginibre(num_samples, n), dtype=th.cfloat)
    U = th.linalg.qr(X)[0]
    p = th.rand(num_samples, n).to(th.cfloat)
    p = p / p.sum(dim=1, keepdim=True)
    rho = U @ th.diag_embed(p) @ U.mH
    return np.linalg.eigvals(rho.numpy()).real.ravel()


def simulate(num_samples=NUM_SAMPLES, n=N, serial=False):
    np.random.seed(SEED)
    th.manual_seed(SEED)
    t0 = time.time()
    eigs_pd = sample_wishart_eigs(num_samples, n)
    print(f"Wishart ensemble:         {eigs_pd.size} eigenvalues, "
          f"range [{eigs_pd.min():.6f}, {eigs_pd.max():.6f}]  ({time.time() - t0:.1f} s)")
    t1 = time.time()
    eigs_mixed = sample_unitary_product_eigs(num_samples, n, serial=serial)
    print(f"Unitary-product ensemble: {eigs_mixed.size} eigenvalues, "
          f"range [{eigs_mixed.min():.6f}, {eigs_mixed.max():.6f}]  ({time.time() - t1:.1f} s)")
    os.makedirs(DERIVED_DIR, exist_ok=True)
    np.savez_compressed(CACHE, eigs_pd=eigs_pd, eigs_mixed=eigs_mixed,
                        num_samples=num_samples, n=n, seed=SEED)
    print(f"cached -> {CACHE}")
    return eigs_pd, eigs_mixed


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit(f"--plot-only needs the cache {CACHE}; run without --plot-only first.")
    d = np.load(CACHE)
    print(f"loaded {CACHE} (num_samples={int(d['num_samples'])}, n={int(d['n'])})")
    return d["eigs_pd"], d["eigs_mixed"]


def plot(eigs_pd, eigs_mixed, out_dir):
    fig, axes = new_figure(width="single")
    ax = axes[0]
    # counts -> percentage of all eigenvalues of that ensemble (each histogram sums to 100 %)
    w1 = np.full_like(eigs_pd, 100 / len(eigs_pd))
    w2 = np.full_like(eigs_mixed, 100 / len(eigs_mixed))
    # each histogram bins over its own data range, as in the source cell
    ax.hist(eigs_pd, bins=BINS, weights=w1, alpha=0.6, color="C0", label="Wishart ensemble")
    ax.hist(eigs_mixed, bins=BINS, weights=w2, alpha=0.6, color="C1",
            label="Unitary-product ensemble")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Percentage of eigenvalues (%)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    png = save_figure(fig, 2, out_dir=out_dir)
    print(f"wrote {png} and {png[:-4]}.pdf")
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=FIG_DIR, help="directory for Fig2.png / Fig2.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from data/derived/fig02_eigenvalues.npz without simulating")
    ap.add_argument("--serial", action="store_true",
                    help="draw the unitary-product states one at a time with the library's "
                         "generate_mixed_state instead of the vectorised sampler")
    args = ap.parse_args()

    t0 = time.time()
    if args.plot_only:
        eigs_pd, eigs_mixed = load_cache()
    else:
        eigs_pd, eigs_mixed = simulate(serial=args.serial)
    plot(eigs_pd, eigs_mixed, args.out_dir)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
