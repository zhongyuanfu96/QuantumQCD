#!/usr/bin/env python3
"""Figure 13 -- detection performance with a noisy (depolarized) pre-change state.

Reproduces the published image ``unknown_FAPvsADD_noise.png``: ADD versus FAP under
the maximum-sensitivity measurement designed from the nominal pre-change state rho,
with the estimated post-change distribution q_hat_t (exponential window, eps = 0.99),
when the actual pre-change state is rho_nu = (1 - nu) rho + nu I_N / N.

Source of the numbers: ``scripts/noise_study.py::experiment_noise_robustness`` with
``null_mode='regular'`` (the "noise-aware" rule of the manuscript legend: for
R_rho in {6, 7} and nu > 0 the pre-change pmf p is computed from rho_nu and the
null-space outcome x_t = 0 enters the CUSUM-like update instead of raising an
immediate alarm).  This figure is NOT a notebook figure -- it comes from the
revision study whose published caches live in ``data/noise/``.

The simulation is fully seeded (numpy Generator(0)) and takes about 45 s; running it
here reproduces ``data/noise/noise_N8_R{6,7,8}_idx9_regular.npz`` bit for bit.
The plotted arrays are cached in ``data/derived/fig13_noise_regular.npz`` so that
``--plot-only`` re-plots without simulating (it falls back to the published caches
in ``data/noise/`` when the derived cache is absent).

Usage:
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        PYTHONPATH=src python3 scripts/fig13_noisy_pre_change.py [--plot-only]
"""
import argparse
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # code_release/
sys.path.insert(0, os.path.join(_ROOT, 'src', 'quantum_qcd'))   # flat module imports
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _HERE)                            # figstyle, noise_study

import figstyle                                       # noqa: E402  (sets Agg backend)
from figstyle import new_figure, save_figure          # noqa: E402

# ---------------------------------------------------------------------------
# Parameters -- identical to experiment_noise_robustness's defaults, i.e. to the
# settings of Fig. 11 (test_size = 2**11 FAP runs, add_multi = 100, eps = 0.99,
# base threshold grid np.linspace(0, 4.5, 100), state index 9 of each rank set).
# ---------------------------------------------------------------------------
R_LIST = (8, 7, 6)                  # plotting/simulation order of noise_study.py
NU_LIST = (0.0, 0.01, 0.05)
SET_INDEX = 9
NULL_MODE = 'regular'
SEED = 0
N_DIM = 8

DATA_DIR = os.path.join(_ROOT, 'data')
DERIVED = os.path.join(DATA_DIR, 'derived', 'fig13_noise_regular.npz')
PUBLISHED_CACHE = os.path.join(DATA_DIR, 'noise')   # noise_N8_R{R}_idx9_regular.npz

# Colours follow Fig. 11's rank order R = 5, 6, 7, 8 -> C0, C1, C2, C3, so that
# R = 6 is orange, R = 7 green and R = 8 red, as the manuscript legend states.
COLORS = {5: 'C0', 6: 'C1', 7: 'C2', 8: 'C3'}
STYLES = {0.0: dict(linestyle='--', linewidth=1.2),
          0.01: dict(linestyle='-'),
          0.05: dict(linestyle='-', marker='o', markevery=6, markersize=2.8)}


def simulate():
    """Run the noise-robustness study and return {(R, nu): (fap, add, h)}."""
    os.chdir(_ROOT)          # quantum_states loads data/X and data/adap_grad_jeremy
    from noise_study import experiment_noise_robustness

    # experiment_noise_robustness always writes its own .npz files and a draft
    # figure (with a title); send both to a throw-away directory and keep only the
    # returned arrays, which we re-cache under data/derived/ for --plot-only.
    with tempfile.TemporaryDirectory() as tmp:
        results = experiment_noise_robustness(
            nu_list=NU_LIST, R_list=R_LIST, set_index=SET_INDEX,
            null_mode=NULL_MODE, seed=SEED, show=False,
            out_dir=tmp, fig_path=os.path.join(tmp, 'noise_study_draft.png'))

    curves = {(R, nu): (results[(R, nu)]['fap'], results[(R, nu)]['add'],
                        results[(R, nu)]['h'])
              for R in R_LIST for nu in NU_LIST}

    os.makedirs(os.path.dirname(DERIVED), exist_ok=True)
    flat = {}
    for (R, nu), (fap, add, h) in curves.items():
        key = f'R{R}_nu{nu:g}'
        flat[f'fap_{key}'] = fap
        flat[f'add_{key}'] = add
        flat[f'h_{key}'] = h
    flat['R_list'] = np.array(R_LIST)
    flat['nu_list'] = np.array(NU_LIST)
    np.savez(DERIVED, **flat)
    print(f'\nCached simulated curves to {DERIVED}')
    return curves


def load_cached():
    """Load the plotted curves from data/derived/, else from the published caches."""
    if os.path.exists(DERIVED):
        d = np.load(DERIVED)
        print(f'Loaded cached curves from {DERIVED}')
        return {(R, nu): (d[f'fap_R{R}_nu{nu:g}'], d[f'add_R{R}_nu{nu:g}'],
                          d[f'h_R{R}_nu{nu:g}'])
                for R in R_LIST for nu in NU_LIST}

    curves = {}
    for R in R_LIST:
        path = os.path.join(PUBLISHED_CACHE, f'noise_N{N_DIM}_R{R}_idx{SET_INDEX}_regular.npz')
        d = np.load(path)
        nus = list(d['nu'])
        print(f'Loaded published cache {path}')
        for nu in NU_LIST:
            k = nus.index(nu)
            curves[(R, nu)] = (d[f'fap_{k}'], d[f'add_{k}'], d[f'h_{k}'])
    return curves


def make_figure(curves, out_dir):
    fig, axes = new_figure(width='medium', height=2.95)
    ax = axes[0]

    for R in R_LIST:
        for nu in NU_LIST:
            fap, add, _ = curves[(R, nu)]
            label = r'$R_{\rho}$' + f' = {R}, ' + r'$\nu$' + f' = {nu:g}'
            ax.plot(fap, add, color=COLORS[R], label=label,
                    **STYLES.get(nu, dict(linestyle='-')))

    ax.set_xlim(0, 5000)
    ax.set_xlabel('FAP')
    ax.set_ylabel('ADD')
    ax.grid(linestyle='--')

    # The legend fills column-wise; reorder so that each column is one nu and each
    # row one rank, exactly as in the published image.
    handles, labels = ax.get_legend_handles_labels()
    n_nu = len(NU_LIST)
    order = [i + j * n_nu for i in range(n_nu) for j in range(len(R_LIST))]
    if len(handles) == len(order):
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.17),
              ncol=n_nu, frameon=False, fontsize=7, columnspacing=1.0,
              handlelength=1.9, handletextpad=0.5, borderpad=0.0, labelspacing=0.4)

    png = save_figure(fig, 13, out_dir=out_dir)
    print(f'Wrote {png} and {png[:-4]}.pdf')
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', default=os.path.join(_ROOT, 'figures'),
                    help='directory for Fig13.png / Fig13.pdf')
    ap.add_argument('--plot-only', action='store_true',
                    help='re-plot from the cached arrays instead of simulating')
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    curves = load_cached() if args.plot_only else simulate()

    # Report the plotted values at a few FAP positions (self-check aid).
    print('\nADD at selected FAP values:')
    print('  R   nu     ' + '  '.join(f'{x:>8d}' for x in (500, 1000, 2500, 5000)))
    for R in R_LIST:
        for nu in NU_LIST:
            fap, add, _ = curves[(R, nu)]
            vals = [np.interp(x, fap, add) for x in (500, 1000, 2500, 5000)]
            print(f'  {R}  {nu:<5g}  ' + '  '.join(f'{v:8.3f}' for v in vals))

    make_figure(curves, out_dir)


if __name__ == '__main__':
    main()
