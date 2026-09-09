"""
Analysis, experiments, and visualization for quantum anomaly detection.

Provides experimental workflows to evaluate CUSUM detection algorithms
on quantum states with various ranks and measurement strategies.
"""

import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import sqrtm
from scipy.stats import entropy, unitary_group

from concurrent.futures import ThreadPoolExecutor
from quantum_detection import (
    cusum_fap_exp_single,
    cusum_add_exp_single,
    cusum_add_exp_single_exact,
    cusum_fap_exp_fast,
    cusum_add_exact,
    cusum_add_est,
    cusum_fap_gpu,
    cusum_add_est_gpu,
    cusum_add_exact_gpu,
)
from quantum_states import (
    generate_X_single,
    generate_sigma,
    generate_rho_set_rank
)
from quantum_utils import (
    povm_positive_eigen,
    distribution_computational_basis_measurement
)

# Set up data directory (local paths, replacing Google Drive)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def setup_experiment_base():
    """
    Initialize base experiment parameters and load quantum states.

    Returns:
    --------
    X_rho : ndarray
        Seed matrix for rho generation
    X_sigma : ndarray
        Seed matrix for sigma generation
    sigma : ndarray
        Reference quantum state (full rank)
    N : int
        Dimension of quantum states
    n_sets : int
        Number of rank-specific state sets
    """
    N = 8
    n_sets = 500

    X_rho, X_sigma = generate_X_single(N, load_exist=True)
    sigma = generate_sigma(X_sigma, N, load_exist=True)
    th_sigma = th.tensor(sigma, dtype=th.cfloat)

    return X_rho, X_sigma, sigma, N, n_sets


def experiment_rank_comparison():
    """
    Compare detection performance across different state ranks.

    Runs CUSUM detection with both optimized von Neumann measurements
    and standard computational basis measurements across ranks 5-8.
    Creates 2x2 subplot comparing FAP vs ADD curves.
    """
    X_rho, X_sigma, sigma, N, n_sets = setup_experiment_base()

    R_list = [5, 6, 7, 8]
    test_size = 2**11
    scheme = "cusum"
    T0 = 256
    forgot_val = 0.99
    max_fap = 5000
    set_num_rank_def = 7
    add_multi = 100

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 8))

    for R in R_list:
        rho_set = generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)
        th_rho_set = th.tensor(rho_set, dtype=th.cfloat)
        rho = rho_set[set_num_rank_def, :, :]

        # Optimized von Neumann measurement
        Q = povm_positive_eigen(rho, eps=1e-6)
        p_dist = np.diagonal(Q @ rho, offset=0, axis1=-1, axis2=-2).sum(-1).real
        q_dist = np.diagonal(Q @ sigma, offset=0, axis1=-1, axis2=-2).sum(-1).real

        print(f'\n\n\nRank = {R}')
        print('\nOptimized von Neumann Measurement')
        if R < 8:
            q_dist = np.append(q_dist, 1-sum(q_dist))
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}, {q_dist[-1]:.3f}')
        else:
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}')

        h_exp = np.linspace(0, 5, 100)

        fap_vN = cusum_fap_exp_single(test_size, p_dist, h_exp, forget_val=forgot_val)
        add_vN_est = cusum_add_exp_single(test_size, p_dist, q_dist, h_exp, forget_val=forgot_val)
        add_vN_exact = cusum_add_exp_single_exact(test_size, p_dist, q_dist, h_exp)

        axes[0, 0].plot(fap_vN, add_vN_est, label=r'$R_{\rho}$'+f' = {R}')
        axes[0, 1].plot(fap_vN, add_vN_exact, label=r'$R_{\rho}$'+f' = {R}')

        print('\nStandard Measurement')
        # Standard measurement
        p_dist_st, q_dist_st = distribution_computational_basis_measurement(N, rho, sigma)
        print(f'p: {sum(p_dist_st):.3f}, {p_dist_st.shape}')
        print(f'q: {sum(q_dist_st):.3f}, {q_dist_st.shape}')

        h_exp = np.linspace(0, 5, 100)

        fap_st = cusum_fap_exp_single(test_size, p_dist_st, h_exp, forget_val=forgot_val)
        add_st = cusum_add_exp_single(test_size, p_dist_st, q_dist_st, h_exp, forget_val=forgot_val)
        add_st_exact = cusum_add_exp_single_exact(test_size, p_dist_st, q_dist_st, h_exp)

        axes[1, 0].plot(fap_st, add_st, label=r'$R_{\rho}$'+f' = {R}')
        axes[1, 1].plot(fap_st, add_st_exact, label=r'$R_{\rho}$'+f' = {R}')

        for rows in range(2):
            for cols in range(2):
                if rows == 0:
                    meas = 'Optimized von Neumann Measurement'
                else:
                    meas = 'Standard Measurement'
                if cols == 0:
                    q_type = "Estimated q"
                else:
                    q_type = "Exact q"
                axes[rows, cols].set_title(f'{meas}, {q_type}')
                axes[rows, cols].set_xlim(0, max_fap)
                axes[rows, cols].grid(linestyle='--')

    axes[0, 0].set_ylim(0, 30)
    axes[0, 1].set_ylim(0, 15)
    axes[1, 0].set_ylim(0, 400)
    axes[1, 1].set_ylim(0, 200)

    fig.supxlabel('FAP', fontsize=12)
    fig.supylabel('ADD', fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def experiment_single_measurement_optimized():
    """
    Single measurement experiment focusing on optimized von Neumann strategy.

    Evaluates CUSUM detection using maximum-sensitivity measurements
    across different state ranks (5-8).
    """
    X_rho, X_sigma, sigma, N, n_sets = setup_experiment_base()

    R_list = [5, 6, 7, 8]
    test_size = 2**11
    scheme = "cusum"
    T0 = 256
    forgot_val = 0.99
    max_fap = 5000
    set_num_rank_def = 9
    add_multi = 100

    fig, axes = plt.subplots(figsize=(8, 5))

    for R in R_list:
        rho_set = generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)
        th_rho_set = th.tensor(rho_set, dtype=th.cfloat)
        rho = rho_set[set_num_rank_def, :, :]

        # Optimized von Neumann measurement
        Q = povm_positive_eigen(rho, eps=1e-6)
        p_dist = np.diagonal(Q @ rho, offset=0, axis1=-1, axis2=-2).sum(-1).real
        q_dist = np.diagonal(Q @ sigma, offset=0, axis1=-1, axis2=-2).sum(-1).real

        print(f'\n\n\nRank = {R}')
        print('\nOptimized von Neumann Measurement')
        if R < 8:
            q_dist = np.append(q_dist, 1-sum(q_dist))
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}, {q_dist[-1]:.3f}')
        else:
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}')

        h_exp = np.linspace(0, 4.5, 100)

        fap_vN = cusum_fap_exp_single(test_size, p_dist, h_exp, forget_val=forgot_val)
        add_vN_est = cusum_add_exp_single(test_size*add_multi, p_dist, q_dist, h_exp, forget_val=forgot_val)
        add_vN_exact = cusum_add_exp_single_exact(test_size*add_multi, p_dist, q_dist, h_exp)

        curve = axes.plot(fap_vN, add_vN_est, label=r'$R_{\rho}$'+f' = {R}, Estimated ' + r'$\hat{\mathbf{q}}_t$')
        line_color = curve[0].get_color()
        axes.plot(fap_vN, add_vN_exact, label=r'$R_{\rho}$'+f' = {R}, Exact ' + r'$\mathbf{q}$',
                 color=line_color, linestyle='--')

        axes.set_title(f'Maximum-sensitivity Measurement')
        axes.set_xlim(0, max_fap)
        axes.grid(linestyle='--')

    fig.supxlabel('FAP', fontsize=12)
    fig.supylabel('ADD', fontsize=12)
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def experiment_estimated_vs_exact():
    """
    Compare estimated vs exact post-change distributions.

    Evaluates how distribution estimation affects detection performance
    by comparing estimated q_hat vs exact q across ranks 5-8.
    Uses GPU-accelerated CUSUM and runs FAP/ADD in parallel.
    """
    X_rho, X_sigma, sigma, N, n_sets = setup_experiment_base()

    R_list = [8]
    test_size = 2**11
    forgot_val = 0.99
    max_fap = 5000
    set_num_rank_def = 9
    add_multi = 100

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    for R in R_list:
        rho_set = generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)
        rho = rho_set[set_num_rank_def, :, :]

        # Optimized von Neumann measurement
        Q = povm_positive_eigen(rho, eps=1e-6)
        p_dist = np.diagonal(Q @ rho, offset=0, axis1=-1, axis2=-2).sum(-1).real
        q_dist = np.diagonal(Q @ sigma, offset=0, axis1=-1, axis2=-2).sum(-1).real

        print(f'\n\n\nRank = {R}')
        print('\nOptimized von Neumann Measurement')
        if R < 8:
            q_dist = np.append(q_dist, 1-sum(q_dist))
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}, {q_dist[-1]:.3f}')
        else:
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}')

        h_exp = np.linspace(0, 3, 100)

        # Run FAP, ADD-est, ADD-exact in parallel via threads (GPU)
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_fap = pool.submit(cusum_fap_gpu, test_size, p_dist, h_exp,
                                  forget_val=forgot_val)
            fut_est = pool.submit(cusum_add_est_gpu, test_size * add_multi,
                                  p_dist, q_dist, h_exp, forget_val=forgot_val)
            fut_exact = pool.submit(cusum_add_exact_gpu, test_size * add_multi,
                                    p_dist, q_dist, h_exp)
            fap_vN = fut_fap.result()
            add_vN_est = fut_est.result()
            add_vN_exact = fut_exact.result()

        axes[0].plot(fap_vN, add_vN_est, label=r'$R_{\rho}$'+f' = {R}')
        axes[1].plot(fap_vN, add_vN_exact, label=r'$R_{\rho}$'+f' = {R}')

    for cols in range(2):
        axes[cols].set_xlim(0, max_fap)
        axes[cols].grid(linestyle='--')

    axes[0].set_title(r'Estimated $\hat{\mathbf{q}}_t$')
    axes[1].set_title(r'Exact $\mathbf{q}$')

    fig.supxlabel('FAP', fontsize=12)
    fig.supylabel('ADD', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('estimated_vs_exact.png', dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to estimated_vs_exact.png')
    plt.show()


def experiment_batch_fastforward_detection():
    """
    High-speed detection experiment with batched vectorized processing.

    Uses vectorized batch processing to rapidly evaluate detection performance
    with larger test sizes. Evaluates both estimated and exact distributions,
    and tracks detections by two different criteria (exclusive outcome vs threshold).
    """
    X_rho, X_sigma, sigma, N, n_sets = setup_experiment_base()

    R_list = [5, 6, 7]
    test_size = 50000
    scheme = "cusum"
    T0 = 256
    forgot_val = 0.99
    set_num_deficit = 9
    add_multi = 100

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    for R in R_list:
        rho_set = generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)
        rho = rho_set[set_num_deficit, :, :]
        th_rho = th.tensor(rho, dtype=th.cfloat)

        sigma = generate_sigma(X_sigma, N, load_exist=True)
        th_sigma = th.tensor(sigma, dtype=th.cfloat)

        Q = povm_positive_eigen(rho, eps=1e-6)
        p_dist = np.diagonal(Q @ rho, offset=0, axis1=-1, axis2=-2).sum(-1).real
        q_dist = np.diagonal(Q @ sigma, offset=0, axis1=-1, axis2=-2).sum(-1).real

        print(f'\nRank = {R}')
        if R < 8:
            q_dist = np.append(q_dist, 1-sum(q_dist))
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}, {q_dist[-1]:.3f}')
        else:
            print(f'p: {sum(p_dist):.3f}, {p_dist.shape}')
            print(f'q: {sum(q_dist):.3f}, {q_dist.shape}')

        if R == 5:
            h_exp = np.linspace(0, 4.5, 100)
        elif R == 6:
            h_exp = np.linspace(0, 5, 100)
        elif R == 7:
            h_exp = np.linspace(0, 5, 100)

        fap = cusum_fap_exp_fast(test_size, p_dist, h_exp, forget_val=forgot_val, batch_size=1000)
        add_est_xt, add_est_st, est_xt_count, est_st_count = cusum_add_est(test_size*add_multi, p_dist, q_dist, h_exp, forget_val=forgot_val)
        add_exact_xt, add_exact_st, exact_xt_count, exact_st_count = cusum_add_exact(test_size*add_multi, p_dist, q_dist, h_exp)

        first_index = np.where(fap >= 5000)[0][0]
        print(f'Est counter: {int(est_xt_count)}, {int(est_st_count)}')
        print(f'Est: FAP, ADD_xt, ADD_st: {fap[first_index]:.2f}, {add_est_xt[first_index]:.2f}, {add_est_st[first_index]:.2f}')
        print(f'Exact counter: {int(exact_xt_count)}, {int(exact_st_count)}')
        print(f'Exact: FAP, ADD_xt, ADD_st: {fap[first_index]:.2f}, {add_exact_xt[first_index]:.2f}, {add_exact_st[first_index]:.2f}')

        curve = ax[0].plot(fap, add_est_xt, label=r'$R_{\rho}$'+f' = {R}, ' + r'$x_{{t}}=0$')
        line_color = curve[0].get_color()
        ax[0].plot(fap, add_est_st, label=r'$R_{\rho}$'+f' = {R}, ' + r'$s_{{t}}>h$',
                  color=line_color, linestyle='--')

        curve = ax[1].plot(fap, add_exact_xt, label=r'$R_{\rho}$'+f' = {R}, ' + r'$x_{{t}}=0$')
        line_color = curve[0].get_color()
        ax[1].plot(fap, add_exact_st, label=r'$R_{\rho}$'+f' = {R}, ' + r'$s_{{t}}>h$',
                  color=line_color, linestyle='--')

    for i in range(2):
        ax[i].set_xlim(0, 5000)
        ax[i].grid(linestyle='--')

    ax[0].set_title(r'Estimated $\hat{\mathbf{q}}_t$')
    ax[1].set_title(r'Exact $\mathbf{q}$')
    ax[0].set_ylim(0, 30)

    fig.supxlabel('FAP', fontsize=12)
    fig.supylabel('ADD', fontsize=12)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def debug_state_inspection():
    """
    Debug utility to inspect quantum states and reference matrices.

    Prints and displays specific quantum states for verification.
    """
    X_rho, X_sigma, _, N, n_sets = setup_experiment_base()

    set_num_rank_def = 7
    N = 8
    R = 7

    rho_set = generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)
    th_rho_set = th.tensor(rho_set, dtype=th.cfloat)
    rho = rho_set[set_num_rank_def, :, :]
    print(rho)

    X_rho, X_sigma = generate_X_single(N, load_exist=True)
    sigma = generate_sigma(X_sigma, N, load_exist=True)
    print(sigma)


def experiment_noise_robustness(nu_list=(0.0, 0.01, 0.05), R_list=(8, 7, 6), set_index=9,
                                test_size=2**11, add_multi=100, forget_val=0.99,
                                h_vec=None, min_fap=5000, h_ceiling=12.0, h_chunk=1.0,
                                null_mode='alarm', out_dir=None, fig_path=None, show=True,
                                seed=0):
    """
    Robustness of the maximum-sensitivity design to a depolarized pre-change
    state (revision Concern 2).

    The POVM and the nominal pre-change distribution p are designed from the
    noiseless rho, while the actual pre-change state during monitoring is
    rho_hat = (1 - nu) rho + nu I / N; the post-change state sigma is unchanged.
    Same state pair per rank (set_index) and same detector settings as
    experiment_single_measurement_optimized (Fig. 11 of the paper).

    Parameters:
    -----------
    nu_list : iterable of float
        Depolarizing noise levels (0 gives the noiseless reference)
    R_list : iterable of int
        Ranks of the pre-change state
    set_index : int
        Index of the state within the rank-R set
    test_size : int
        Number of FAP experiments (ADD uses test_size * add_multi)
    h_vec : array-like, optional
        Base threshold grid (default np.linspace(0, 4.5, 100) as in Fig. 11)
    min_fap : float
        The grid is extended in chunks of h_chunk (same spacing) until the
        FAP reaches min_fap, the FAP stops growing (capped), or h_ceiling is hit.
        Set to 0 to disable the extension.
    null_mode : {'alarm', 'regular'}
        Treatment of the null-space outcome x_t = 0 for rank-deficient rho when
        nu > 0. 'alarm': the paper's one-shot rule (immediate alarm; the FAP is
        then capped at N / (nu (N - R))). 'regular': noise-aware rule that
        assumes nu known and treats x_t = 0 as a regular outcome with nominal
        pre-change probability nu (N - R) / N, i.e. p is designed from rho_hat.
        The nu = 0 reference always uses the one-shot rule.
    out_dir : str, optional
        Where the .npz results are written (default data/noise)
    fig_path : str, optional
        Where the figure is written (default <out_dir>/unknown_FAPvsADD_noise[_regular].png)

    Returns:
    --------
    results : dict
        {(R, nu): {'h', 'fap', 'add', 'p_true', 'p_nom', 'q', 'null_mode'}}
    """
    from quantum_utils import depolarize, measurement_distribution
    from quantum_detection import cusum_fap_mismatch, cusum_add_est_fast

    if null_mode not in ('alarm', 'regular'):
        raise ValueError("null_mode must be 'alarm' or 'regular'")
    X_rho, X_sigma, sigma, N, n_sets = setup_experiment_base()
    h_base = np.linspace(0, 4.5, 100) if h_vec is None else np.asarray(h_vec, dtype=float)
    dh = h_base[1] - h_base[0] if len(h_base) > 1 else 0.05
    out_dir = os.path.join(DATA_DIR, 'noise') if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    suffix = '' if null_mode == 'alarm' else '_regular'

    results = {}
    add_cache = {}
    for R in R_list:
        rho = generate_rho_set_rank(X_rho, N, R, n_sets, load_exist=True)[set_index]
        Q = povm_positive_eigen(rho, eps=1e-6)
        p_nom_base = measurement_distribution(Q, rho, append_null=False)   # length R
        q_dist = measurement_distribution(Q, sigma)                        # length R (+1)

        saved = {'nu': np.array(list(nu_list)), 'p_nom_base': p_nom_base, 'q': q_dist}
        for k, nu in enumerate(nu_list):
            if nu == 0:
                p_true, p_nom = p_nom_base, p_nom_base                      # one-shot rule
            else:
                p_true = measurement_distribution(Q, depolarize(rho, nu))  # length R (+1)
                p_nom = p_true if (null_mode == 'regular' and R < N) else p_nom_base

            # 1) Threshold range: extend the base grid in chunks, using a cheaper probe,
            #    until the FAP reaches min_fap (with a 10% margin), stops growing
            #    (capped by the null outcome), or h_ceiling is hit.
            h = h_base.copy()
            probe_size = max(128, test_size // 8)
            fap_probe = cusum_fap_mismatch(probe_size, p_true, p_nom, h,
                                           forget_val=forget_val, rng=rng)
            while min_fap > 0 and fap_probe.max() < 1.1 * min_fap and h[-1] < h_ceiling - 1e-9:
                h_ext = np.arange(h[-1] + dh, min(h[-1] + h_chunk, h_ceiling) + 1e-9, dh)
                if h_ext.size == 0:
                    break
                fap_ext = cusum_fap_mismatch(probe_size, p_true, p_nom, h_ext,
                                             forget_val=forget_val, rng=rng)
                capped = fap_ext.max() < 1.02 * fap_probe.max()   # FAP no longer grows with h
                h = np.concatenate([h, h_ext])
                fap_probe = np.concatenate([fap_probe, fap_ext])
                if capped:
                    break
            # 2) Final FAP: one simulation over the whole grid, so that the same runs
            #    cross the increasing thresholds in order (monotone curve, no stitching).
            fap = cusum_fap_mismatch(test_size, p_true, p_nom, h, forget_val=forget_val, rng=rng)
            # ADD depends on (p_nom, h) only, not on p_true: reuse it across nu
            add_key = (R, p_nom.tobytes(), h.tobytes())
            if add_key not in add_cache:
                add_cache[add_key] = cusum_add_est_fast(test_size * add_multi, p_nom, q_dist, h,
                                                        forget_val=forget_val, rng=rng)
            add = add_cache[add_key]

            results[(R, nu)] = {'h': h, 'fap': fap, 'add': add, 'p_true': p_true,
                                'p_nom': p_nom, 'q': q_dist, 'null_mode': null_mode}
            saved[f'h_{k}'] = h
            saved[f'fap_{k}'] = fap
            saved[f'add_{k}'] = add
            p_null = p_true[-1] if len(p_true) > R else 0.0
            print(f'R = {R}, nu = {nu:g} [{null_mode}]: P(null | pre-change) = {p_null:.5f}, '
                  f'h up to {h[-1]:.2f}, FAP max {fap.max():.1f}, '
                  f'ADD at FAP={min_fap}: '
                  + (f'{np.interp(min_fap, fap, add):.2f}' if fap.max() >= min_fap > 0 else 'n/a'))

        np.savez(os.path.join(out_dir, f'noise_N{N}_R{R}_idx{set_index}{suffix}.npz'), **saved)

    # Figure: same style as experiment_single_measurement_optimized (Fig. 11);
    # colours follow Fig. 11's order R = 5, 6, 7, 8 -> C0, C1, C2, C3.
    colors = {5: 'C0', 6: 'C1', 7: 'C2', 8: 'C3'}
    styles = {0.0: dict(linestyle='--', linewidth=1.2),
              0.01: dict(linestyle='-'),
              0.05: dict(linestyle='-', marker='o', markevery=6, markersize=3.5)}

    fig, axes = plt.subplots(figsize=(8, 5))
    for R in R_list:
        for nu in nu_list:
            res = results[(R, nu)]
            label = r'$R_{\rho}$' + f' = {R}, ' + r'$\nu$' + f' = {nu:g}'
            axes.plot(res['fap'], res['add'], color=colors.get(R),
                      label=label, **styles.get(nu, dict(linestyle='-')))
    axes.set_title('Maximum-sensitivity Measurement, Estimated ' + r'$\hat{\mathbf{q}}_t$')
    axes.set_xlim(0, 5000)
    axes.grid(linestyle='--')

    fig.supxlabel('FAP', fontsize=12)
    fig.supylabel('ADD', fontsize=12)
    handles, labels = axes.get_legend_handles_labels()
    # fig.legend fills column-wise; reorder so that each rank occupies one row
    n_nu = len(nu_list)
    order = [i + j * n_nu for i in range(n_nu) for j in range(len(R_list))]
    if len(handles) == len(order):
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.12 - 0.05 * max(0, len(R_list) - 2)),
               ncol=n_nu, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if fig_path is None:
        fig_path = os.path.join(out_dir, f'unknown_FAPvsADD_noise{suffix}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fig_path}')
    if show:
        plt.show()
    plt.close(fig)
    return results


if __name__ == '__main__':
    # Run analysis experiments
    # Uncomment to run specific experiments:

    # experiment_rank_comparison()
    # experiment_single_measurement_optimized()
    # experiment_estimated_vs_exact()
    # experiment_batch_fastforward_detection()
    # debug_state_inspection()

    # python3 analysis.py noise [fig_path] [alarm|regular]  -> noisy pre-change robustness figure
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'noise':
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        experiment_noise_robustness(
            show=False,
            fig_path=sys.argv[2] if len(sys.argv) > 2 else None,
            null_mode=sys.argv[3] if len(sys.argv) > 3 else 'alarm')
