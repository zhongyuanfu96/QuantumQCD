#!/usr/bin/env python3
r"""Figure 12: conditional detection delay under the maximum-sensitivity measurement
for rank-deficient pre-change states, with the post-change distribution unknown.

Regenerates paper Fig. 12 (published as ``unknown_ADDvsFAP_R_cond.png``).

Source
------
notebooks_raw/Quantum_Anomaly_Detection/optimization_unknown/Sensitivity_FAP_ADD_rank_def.ipynb
  * cell 32  -- the plotting/driver cell (simulation + figure)
  * cell 30  -- ``cusum_fap_exp_fast``   (re-exported unchanged by quantum_qcd.cusum_nb)
  * cell 31  -- ``cusum_add_est`` / ``cusum_add_exact`` (batched, seeded ports in cusum_nb)
  * cells 10/11/15/21 -- state generation and ``povm_positive_eigen`` (already in
    quantum_qcd.quantum_states / quantum_qcd.quantum_utils)

What is drawn
-------------
For pre-change rank R_rho = 5 (C0 blue), 6 (C1 orange) and 7 (C2 green) of the N = 8
state pair (rho = rho_set[9], sigma = sigma_8), ADD versus FAP **conditioned on the
stopping cause**:

  solid  -- the run stopped on the null-space outcome  x_t = 0   (outcome index R,
            impossible before the change);
  dashed -- the run stopped because the CUSUM-like statistic crossed the largest
            threshold, s_t > h.  Same colour as the solid curve of that rank.

Panel a: post-change pmf estimated online with an exponential window (eps = 0.99).
Panel b: exact post-change pmf q.

Both conditional averages are normalised by their own counters, which start at 1e-10
exactly as in the notebook, so a cause that never fires gives an identically-zero
curve -- this is what pins the R = 5 dashed curves to ADD = 0 in both panels.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))
sys.path.insert(0, HERE)

from figstyle import label_panels, new_figure, save_figure          # noqa: E402
from quantum_qcd.cusum_nb import (                                  # noqa: E402
    append_null_outcome,
    cusum_add_cond_est_batch,
    cusum_add_cond_exact_batch,
    cusum_fap_exp_fast_seeded,
)
from quantum_qcd.quantum_utils import povm_positive_eigen           # noqa: E402

DATA_DIR = os.path.join(HERE, os.pardir, "data")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
CACHE = os.path.join(DERIVED_DIR, "fig12_cond_add.npz")
DEFAULT_OUT = os.path.join(HERE, os.pardir, "figures")

# ---- parameters of cell 32 (unchanged) ----------------------------------------------
N = 8                      # Hilbert-space dimension
N_SETS = 500               # size of the stored rho ensemble (file name only)
R_LIST = [5, 6, 7]         # pre-change ranks
SET_NUM_DEFICIT = 9        # index of the state pair inside each rho set
FORGOT_VAL = 0.99          # exponential-window forgetting factor
TEST_SIZE = 50_000         # trials of the FAP simulation
ADD_MULTI = 100            # ADD trials = TEST_SIZE * ADD_MULTI = 5_000_000
EPS_POVM = 1e-6            # support threshold of povm_positive_eigen
FAP_BATCH = 1000           # cell 32 passes batch_size = 1000 to cusum_fap_exp_fast
ADD_BATCH = 50_000         # chunk size of the vectorised ADD ports (see deviations)
MAX_FAP = 5000             # x-axis horizon / the notebook's diagnostic point
# per-rank threshold grids -- R = 5 stops at 4.5, R = 6 and 7 at 5 (cell 32)
H_GRID = {5: np.linspace(0, 4.5, 100),
          6: np.linspace(0, 5, 100),
          7: np.linspace(0, 5, 100)}
# seeds (the notebook seeded nothing); one per (rank, routine) so a rerun is bit-exact
SEEDS = {(R, kind): 1000 * R + k
         for R in R_LIST for k, kind in enumerate(("fap", "est", "exact"))}


# -------------------------------------------------------------------------------------
# distributions
# -------------------------------------------------------------------------------------
def outcome_distributions(R):
    """p (length R) and q (length R + 1, null-space outcome last) of cell 32."""
    rho_set = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy",
                                   f"rho_{N}_rank_{R}_set_{N_SETS}.npy"))
    sigma = np.load(os.path.join(DATA_DIR, "adap_grad_jeremy", f"sigma_{N}.npy"))
    rho = rho_set[SET_NUM_DEFICIT, :, :]

    Q = povm_positive_eigen(rho, eps=EPS_POVM)      # maximum-sensitivity measurement
    p_dist = np.diagonal(Q @ rho, offset=0, axis1=-1, axis2=-2).sum(-1).real
    q_dist = np.diagonal(Q @ sigma, offset=0, axis1=-1, axis2=-2).sum(-1).real
    if R < N:
        q_dist = append_null_outcome(q_dist)        # np.append(q, 1 - sum(q))
    return p_dist, q_dist


# -------------------------------------------------------------------------------------
# simulation
# -------------------------------------------------------------------------------------
def simulate():
    """Run the three Monte-Carlo jobs of cell 32 for every rank; return a dict of arrays."""
    out = {}
    for R in R_LIST:
        p_dist, q_dist = outcome_distributions(R)
        h_exp = H_GRID[R]
        print(f"\nRank = {R}")
        print(f"p: {sum(p_dist):.3f}, {p_dist.shape}")
        print(f"q: {sum(q_dist):.3f}, {q_dist.shape}, {q_dist[-1]:.3f}")

        t0 = time.time()
        fap = cusum_fap_exp_fast_seeded(TEST_SIZE, p_dist, h_exp,
                                        forget_val=FORGOT_VAL, batch_size=FAP_BATCH,
                                        seed=SEEDS[(R, "fap")])
        print(f"  FAP   ({TEST_SIZE} trials): {time.time() - t0:.1f} s")

        t0 = time.time()
        add_est_xt, add_est_st, est_xt_count, est_st_count = cusum_add_cond_est_batch(
            TEST_SIZE * ADD_MULTI, p_dist, q_dist, h_exp, forget_val=FORGOT_VAL,
            seed=SEEDS[(R, "est")], batch_size=ADD_BATCH)
        print(f"  ADD est   ({TEST_SIZE * ADD_MULTI} trials): {time.time() - t0:.1f} s")

        t0 = time.time()
        add_exact_xt, add_exact_st, exact_xt_count, exact_st_count = (
            cusum_add_cond_exact_batch(
                TEST_SIZE * ADD_MULTI, p_dist, q_dist, h_exp,
                seed=SEEDS[(R, "exact")], batch_size=ADD_BATCH))
        print(f"  ADD exact ({TEST_SIZE * ADD_MULTI} trials): {time.time() - t0:.1f} s")

        # the notebook's diagnostics, at the first threshold whose FAP reaches 5000
        first_index = np.where(fap >= MAX_FAP)[0][0]
        print(f"Est counter: {int(est_xt_count)}, {int(est_st_count)}")
        print(f"Est: FAP, ADD_xt, ADD_st: {fap[first_index]:.2f}, "
              f"{add_est_xt[first_index]:.2f}, {add_est_st[first_index]:.2f}")
        print(f"Exact counter: {int(exact_xt_count)}, {int(exact_st_count)}")
        print(f"Exact: FAP, ADD_xt, ADD_st: {fap[first_index]:.2f}, "
              f"{add_exact_xt[first_index]:.2f}, {add_exact_st[first_index]:.2f}")

        out[f"h_{R}"] = h_exp
        out[f"fap_{R}"] = fap
        out[f"add_est_xt_{R}"] = add_est_xt
        out[f"add_est_st_{R}"] = add_est_st
        out[f"add_exact_xt_{R}"] = add_exact_xt
        out[f"add_exact_st_{R}"] = add_exact_st
        out[f"counts_{R}"] = np.array([est_xt_count, est_st_count,
                                       exact_xt_count, exact_st_count])
        out[f"p_{R}"] = p_dist
        out[f"q_{R}"] = q_dist
    return out


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit(f"--plot-only needs the cache {CACHE}; run without it first")
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


# -------------------------------------------------------------------------------------
# figure
# -------------------------------------------------------------------------------------
def make_figure(res, out_dir):
    # two panels, shared y axis, journal double width
    fig, axes = new_figure(nrows=1, ncols=2, width="double", height=3.5, sharey=True)

    for R in R_LIST:
        fap = res[f"fap_{R}"]
        # panel a: estimated q_hat_t     panel b: exact q
        for ax, tag in ((axes[0], "est"), (axes[1], "exact")):
            curve = ax.plot(fap, res[f"add_{tag}_xt_{R}"],
                            label=r'$R_{\rho}$' + f' = {R}, ' + r'$x_{{t}}=0$')
            line_color = curve[0].get_color()     # C0 / C1 / C2 in rank order
            ax.plot(fap, res[f"add_{tag}_st_{R}"],
                    label=r'$R_{\rho}$' + f' = {R}, ' + r'$s_{{t}}>h$',
                    color=line_color, linestyle='--')

    for ax in axes:
        ax.set_xlim(0, MAX_FAP)
        ax.grid(linestyle='--')
    axes[0].set_ylim(0, 30)                        # sharey -> both panels 0..30

    handles, labels = axes[0].get_legend_handles_labels()
    # axes in the upper part of the canvas; the shared FAP label and the plot key
    # (below the axes, as in the source cell) live in the reserved bottom strip
    fig.tight_layout(rect=[0.03, 0.19, 1, 1])
    fig.supxlabel('FAP', y=0.125)
    fig.supylabel('ADD', x=0.012)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False)
    label_panels(fig, axes)                        # bold a / b (replace the cell titles)
    return save_figure(fig, 12, out_dir=out_dir)


def report(res):
    """Numbers used for the self-check against the published PNG."""
    for R in R_LIST:
        fap = res[f"fap_{R}"]
        first_index = np.where(fap >= MAX_FAP)[0][0]
        print(f"\n[fig12] R = {R}: fap[0]={fap[0]:.1f} fap[-1]={fap[-1]:.1f} "
              f"first h with FAP>=5000 -> index {first_index}, FAP {fap[first_index]:.2f}")
        for tag in ("est", "exact"):
            xt = res[f"add_{tag}_xt_{R}"]
            st = res[f"add_{tag}_st_{R}"]
            vals = [np.interp(x, fap, xt) for x in (1000, 2000, 3000, 4000, 5000)]
            vals_st = [np.interp(x, fap, st) for x in (1000, 2000, 3000, 4000, 5000)]
            print(f"   {tag:5s} x_t=0 at FAP 1k..5k: "
                  + ", ".join(f"{v:.2f}" for v in vals))
            print(f"   {tag:5s} s_t>h at FAP 1k..5k: "
                  + ", ".join(f"{v:.2f}" for v in vals_st))
        c = res[f"counts_{R}"]
        print(f"   counters est ({int(c[0])}, {int(c[1])})  exact ({int(c[2])}, {int(c[3])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="directory for Fig12.png / Fig12.pdf")
    ap.add_argument("--plot-only", action="store_true",
                    help=f"re-plot from the cached simulation in {CACHE}")
    args = ap.parse_args()

    if args.plot_only:
        res = load_cache()
    else:
        t0 = time.time()
        res = simulate()
        os.makedirs(DERIVED_DIR, exist_ok=True)
        np.savez_compressed(CACHE, **res)
        print(f"\n[fig12] simulation cached in {CACHE} ({time.time() - t0:.1f} s)")

    report(res)
    png = make_figure(res, args.out_dir)
    print(f"\n[fig12] wrote {png} and {png[:-4]}.pdf")


if __name__ == "__main__":
    main()
