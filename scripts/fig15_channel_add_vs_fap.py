#!/usr/bin/env python3
"""Figure 15: average detection delay (ADD) versus false alarm period (FAP) for
quantum *channel* change detection, N_a = 8 (probe) and N_b = 6 (channel output).

Six curves, three colours:

    C0 blue    m1 = 3   solid  Max-KL   (known post-change)   \\ probe + measurement
               m1 = 3   dashed Max-sens (unknown post-change) /  from the rank
    C1 orange  m1 = 6   solid  Max-KL                         \\  minimization step
               m1 = 6   dashed Max-sens                       /   only
    C2 green   m1 = 13  solid  Max-KL   \\ rank minimization followed by the joint
               m1 = 13  dashed Max-sens /  alternating optimization step

Source: notebooks_raw/Quantum_Anomaly_Detection/probe_known/probing_known.ipynb,
plot cell 118, fed by

  stage 1 (deterministic measurement design, ~7 s, module src/quantum_qcd/channel.py)
    cell  97  known,   m1 < 13 : log-det rank minimization + alg1   -> design_known
    cell  84  known,   m1 = 13 : joint alternating optimization     -> design_known
    cell 105  unknown, m1 < 13 : same rank minimization, eigenbasis -> design_unknown
    cell 109  unknown, m1 = 13 : sensitivity maximization           -> design_unknown

  stage 2 (CUSUM Monte Carlo, ~50 s, module src/quantum_qcd/cusum_nb.py)
    cell 101  known,   m1 = 3, 6 : test_size 1000, q augmented with the null-space
              outcome 1 - sum(q); cusum_fap_batch(max_time=50_000) and
              cusum_add_batch_rank
    cell 102  known,   m1 = 13   : test_size 2000, full-rank pair;
              cusum_fap_batch(max_time=50_000) and cusum_add_batch
    cell 106  unknown, m1 = 3, 6 : test_size 2000, exponentially-weighted CUSUM
              (forgetting factor 0.99), p augmented with 1e-4 and q with
              1 - sum(q); cusum_fap_exp_batch / cusum_add_exp_batch, max_steps=50_000
    cell 112  unknown, m1 = 13   : test_size 3000, outcomes with p < 1e-5 deleted
              from both pmfs; cusum_fap_batch(max_time=100_000) and cusum_add_batch

The notebook loops also cover m1 = 4, 12 (rank minimization) and m1 = 25 (joint);
cell 118 never plots them.  They are skipped here -- channel.design_* replays the
notebook's single th.manual_seed(10) draw stream via ``skip_draws``, so m1 = 3, 6
and 13 come out bit-identical to the values they had inside the notebook's loop.

The only deliberate change to the simulation is that every Monte-Carlo call is
seeded (the notebook cells set no seed at all, so the published curves cannot be
reproduced sample-for-sample); the sample sizes, thresholds, horizons and pmfs
are the notebook's.  --test-scale multiplies every sample size by an integer
factor if a smoother reproduction is wanted; it defaults to 1, i.e. the
notebook's own 1000 / 2000 / 2000 / 3000 runs.

Usage:
    python3 scripts/fig15_channel_add_vs_fap.py [--out-dir DIR] [--plot-only]
                                                [--test-scale K]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "quantum_qcd"))
sys.path.insert(0, HERE)

import channel                                    # noqa: E402
import cusum_nb as cn                             # noqa: E402
from figstyle import new_figure, save_figure      # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "derived", "fig15_channel_add_vs_fap.npz")

# --- parameters of the source cells (unchanged) ------------------------------
N_A = 8                       # probe / channel input dimension   (cells 105, 109)
N_B = 6                       # channel output dimension          (cells 105, 109)
M_PLOT = [3, 6, 13]           # cell 118's loop
EXP_VAL = 0.99                # forgetting factor, set in cell 101, used by cell 106
P_NULL = 1e-4                 # pre-change mass given to the null outcome (cell 106)
TOL_DELETE = 1e-5             # cell 112 deletes outcomes with p < 1e-5
Q_AUGMENT_TOL = 1e-4          # cell 101's `if abs(1 - q.sum()) > 1e-4`

# threshold grids, per m1 and per branch (cells 101, 102, 106, 112)
H_KNOWN = {3: np.linspace(0, 2.7, 100),      # cell 101
           6: np.linspace(0, 4, 100),        # cell 101
           13: np.linspace(0, 5, 100)}       # cell 102
H_UNKNOWN = {3: np.linspace(0, 2.7, 100),    # cell 106
             6: np.linspace(0, 4, 100),      # cell 106
             13: np.linspace(0, 8, 100)}     # cell 112

TEST_SIZE_KNOWN = {3: 1000, 6: 1000,         # cell 101
                   13: 2000}                 # cell 102
TEST_SIZE_UNKNOWN = {3: 2000, 6: 2000,       # cell 106
                     13: 3000}               # cell 112
MAX_TIME_KNOWN = 50_000                      # cells 101, 102
MAX_STEPS_EXP = 50_000                       # cell 106
MAX_TIME_UNKNOWN_JOINT = 100_000             # cell 112

# axes of cell 118
XLIM = (0, 5000)
YLIM = (0, 12)

# master seed for the Monte-Carlo stage (the notebook cells were unseeded)
SEED = 0

# legend labels, exactly as cell 118 builds them
def _label(m1, known):
    suffix = ("Rank Min only" if m1 < N_A + N_B - 1 else "Rank Min & Joint Opt")
    return f"{'Max-KL' if known else 'Max-sens'}, $m_1$ = {m1}, {suffix}"


# -----------------------------------------------------------------------------
# stage 1: measurement design (deterministic)
# -----------------------------------------------------------------------------
def designs():
    """(p, q) pairs for the six curves; cells 84/97/105/109 via channel.py."""
    out = {}
    for m1 in M_PLOT:
        dk = channel.design_known(m1, N_a=N_A, N_b=N_B)
        du = channel.design_unknown(m1, N_a=N_A, N_b=N_B)
        out[m1] = {"known": dk, "unknown": du}
        for tag, d in (("Max-KL ", dk), ("Max-sens", du)):
            print(f"  m1 = {m1:>2}  {tag}  [{d['method']:>8}]  R* = {d['rank']}  "
                  f"sum(p) = {d['p'].sum():.6f}  sum(q) = {d['q'].sum():.6f}")
            print(f"              p = {np.array2string(d['p'], precision=6)}")
            print(f"              q = {np.array2string(d['q'], precision=6)}")
    return out


# -----------------------------------------------------------------------------
# stage 2: CUSUM Monte Carlo
# -----------------------------------------------------------------------------
def known_curves(m1, p, q, rng_fap, rng_add, scale=1):
    """Cell 101 (m1 < 13, rank-deficient pair) or cell 102 (m1 = 13, full rank)."""
    h = H_KNOWN[m1]
    n = TEST_SIZE_KNOWN[m1] * scale
    if m1 < N_A + N_B - 1:
        # cell 101: the rank-minimized measurement leaves 1 - sum(q) of the
        # post-change mass on outcomes with p_i = 0; it becomes an extra symbol.
        if abs(1 - q.sum()) <= Q_AUGMENT_TOL:
            raise RuntimeError(
                f"m1 = {m1}: sum(q) = {q.sum()!r} is within {Q_AUGMENT_TOL} of 1, "
                "so cell 101 would have re-used the previous m1's fap/add arrays "
                "(a stale-variable bug in the notebook). Not reproducible here."
            )
        q_aug = cn.append_null_outcome(q)          # np.append(q, 1 - q.sum())
        fap = cn.cusum_fap_batch(n, p, q_aug, h, max_time=MAX_TIME_KNOWN,
                                 seed=rng_fap)
        add = cn.cusum_add_batch_rank(n, p, q_aug, h, seed=rng_add)
    else:
        # cell 102: p and q both sum to 1, no null outcome
        fap = cn.cusum_fap_batch(n, p, q, h, max_time=MAX_TIME_KNOWN, seed=rng_fap)
        add = cn.cusum_add_batch(n, p, q, h, seed=rng_add)
    return fap, add


def unknown_curves(m1, p, q, rng_fap, rng_add, scale=1):
    """Cell 106 (m1 < 13, exponential window) or cell 112 (m1 = 13, exact q)."""
    h = H_UNKNOWN[m1]
    n = TEST_SIZE_UNKNOWN[m1] * scale
    if m1 < N_A + N_B - 1:
        # cell 106: p <- np.append(p, 1e-4), q <- np.append(q, 1 - q.sum()); the
        # null outcome is an ordinary symbol with a tiny pre-change mass, so the
        # exponential-window detector sees a large but finite log-likelihood ratio.
        p_aug, q_aug = cn.augment_with_null_outcome(p, q, p_null=P_NULL)
        fap = cn.cusum_fap_exp_batch(n, p_aug, h, EXP_VAL,
                                     max_steps=MAX_STEPS_EXP, seed=rng_fap)
        add = cn.cusum_add_exp_batch(n, p_aug, q_aug, h, EXP_VAL,
                                     max_steps=MAX_STEPS_EXP, seed=rng_add)
    else:
        # cell 112: drop the outcomes the sensitivity design has emptied out
        drop = np.where(p < TOL_DELETE)[0]
        p = np.delete(p, drop)
        q = np.delete(q, drop)
        if drop.size:
            print(f"  m1 = {m1}: deleted outcomes {drop.tolist()} with p < "
                  f"{TOL_DELETE:g}; sum(p) = {p.sum():.6f}, sum(q) = {q.sum():.6f}")
        fap = cn.cusum_fap_batch(n, p, q, h, max_time=MAX_TIME_UNKNOWN_JOINT,
                                 seed=rng_fap)
        add = cn.cusum_add_batch(n, p, q, h, seed=rng_add)
    return fap, add


def compute(scale=1):
    print("stage 1: measurement design (deterministic; cells 84/97/105/109)")
    t0 = time.time()
    des = designs()
    print(f"  stage 1 took {time.time() - t0:.1f} s")

    # one independent stream per Monte-Carlo call, all derived from SEED
    streams = np.random.SeedSequence(SEED).spawn(4 * len(M_PLOT))
    rngs = [np.random.default_rng(s) for s in streams]

    print("\nstage 2: CUSUM Monte Carlo (cells 101/102/106/112), "
          f"seeded from SeedSequence({SEED})")
    data = {}
    t0 = time.time()
    for i, m1 in enumerate(M_PLOT):
        for known in (True, False):
            d = des[m1]["known" if known else "unknown"]
            fn = known_curves if known else unknown_curves
            j = 4 * i + (0 if known else 2)
            t1 = time.time()
            fap, add = fn(m1, d["p"].copy(), d["q"].copy(), rngs[j], rngs[j + 1],
                          scale=scale)
            key = f"{'known' if known else 'unknown'}_{m1}"
            data[f"fap_{key}"] = fap
            data[f"add_{key}"] = add
            data[f"p_{key}"] = d["p"]
            data[f"q_{key}"] = d["q"]
            n = (TEST_SIZE_KNOWN if known else TEST_SIZE_UNKNOWN)[m1] * scale
            print(f"  {key:<11s} n = {n:>6d}  "
                  f"{time.time() - t1:6.1f} s   FAP[-1] = {fap[-1]:10.1f}  "
                  f"ADD[-1] = {add[-1]:6.3f}")
    print(f"  stage 2 took {time.time() - t0:.1f} s")
    return data


def load_cache():
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


# -----------------------------------------------------------------------------
# plotting (cell 118, minus the title)
# -----------------------------------------------------------------------------
def make_figure(data, out_dir):
    fig, axes = new_figure(width="medium", height=3.6)
    ax = axes[0]

    for m1 in M_PLOT:
        # solid, next colour of the default cycle (C0, C1, C2)
        line = ax.plot(data[f"fap_known_{m1}"], data[f"add_known_{m1}"],
                       label=_label(m1, True))[0]
        # dashed, same colour: an explicit color= does not advance the cycle
        ax.plot(data[f"fap_unknown_{m1}"], data[f"add_unknown_{m1}"],
                label=_label(m1, False), color=line.get_color(), linestyle="--")

    ax.set_xlabel("FAP")
    ax.set_ylabel("ADD")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.grid(linestyle="--")

    # The source cell put the six-entry key inside the axes (loc='upper left') in
    # a 6.4 x 4.8 in figure.  At the manuscript width of 3.5 in the same box would
    # cover the m1 = 13 Max-sens curve between FAP ~ 1500 and ~ 3900, so the key
    # moves below the axes in two columns; the entries, their order, the line
    # styles and the colours are unchanged.
    fig.tight_layout(rect=[0, 0.115, 1, 1])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=2, frameon=False, fontsize=7, handlelength=1.6,
               columnspacing=0.9, handletextpad=0.4, labelspacing=0.32)
    return save_figure(fig, 15, out_dir=out_dir)


def report(data):
    """Print the plotted curves at a few FAP positions (self-check)."""
    probes = [100, 500, 1000, 2500, 5000]
    print("\nADD read off the plotted (FAP, ADD) curves:")
    print(f"  {'curve':<19s}" + "".join(f"FAP={x:<8d}" for x in probes))
    for m1 in M_PLOT:
        for tag in ("Max-KL", "Max-sens"):
            key = "known" if tag == "Max-KL" else "unknown"
            fap = data[f"fap_{key}_{m1}"]
            add = data[f"add_{key}_{m1}"]
            vals = [np.interp(x, fap, add) for x in probes]
            print(f"  {tag + ', m1 = ' + str(m1):<19s}" +
                  "".join(f"{v:<12.3f}" for v in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from data/derived/fig15_channel_add_vs_fap.npz")
    ap.add_argument("--test-scale", type=int, default=1,
                    help="multiply every Monte-Carlo sample size by this factor "
                         "(default 1 = the notebook's 1000/2000/2000/3000 runs; "
                         "larger values only reduce the sampling noise)")
    args = ap.parse_args()

    if args.plot_only:
        if not os.path.exists(CACHE):
            raise SystemExit(f"--plot-only needs {CACHE}; run without it first")
        data = load_cache()
        print(f"loaded cached curves from {CACHE}")
    else:
        data = compute(scale=args.test_scale)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, **data)
        print(f"cached curves to {CACHE}")

    report(data)
    png = make_figure(data, args.out_dir)
    print(f"wrote {png} and {png[:-4]}.pdf")


if __name__ == "__main__":
    main()
