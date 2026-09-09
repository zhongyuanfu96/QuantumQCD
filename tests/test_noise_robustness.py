"""
Tests for the noisy pre-change robustness experiment (revision Concern 2).

Run from anywhere:  python3 Max_Sensitivity_deff/test_noise_robustness.py
No pytest needed; each test_* function is executed by the runner at the bottom.

The Monte Carlo comparisons use tolerances of several standard errors, so a
failure indicates a real discrepancy rather than sampling noise.
"""

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)                      # data/ paths inside the modules are relative
os.environ.setdefault("MPLBACKEND", "Agg")

from quantum_states import generate_X_single, generate_sigma, generate_rho_set_rank
from quantum_utils import povm_positive_eigen
from quantum_detection import cusum_fap_exp_fast, cusum_add_exp_single

N = 8
SET_INDEX = 9          # the state pair used in Figs. 11 and 12 of the paper
SCRATCH = os.environ.get(
    "NOISE_TEST_SCRATCH",
    "/private/tmp/claude-501/-Users-zhongyuanfu-Documents-Claude-Projects-Quantum-QCD/"
    "00cd612c-bc05-415e-a549-16c5d43b29c1/scratchpad/noise_test",
)


def _states(R, idx=SET_INDEX):
    X_rho, X_sigma = generate_X_single(N, load_exist=True)
    sigma = generate_sigma(X_sigma, N, load_exist=True)
    rho = generate_rho_set_rank(X_rho, N, R, 500, load_exist=True)[idx]
    return rho, sigma


# --------------------------------------------------------------------------
# measurement_distribution / depolarize
# --------------------------------------------------------------------------

def test_depolarize_mixes_state_with_maximally_mixed_state():
    from quantum_utils import depolarize
    rho, _ = _states(8)
    rho_hat = depolarize(rho, 0.05)
    assert np.allclose(rho_hat, 0.95 * rho + 0.05 * np.eye(N) / N)
    # stored states were normalised in float32, so their trace is 1 only to ~1e-7
    assert abs(np.trace(rho_hat).real - 1) < 1e-5


def test_measurement_distribution_appends_null_outcome_for_rank_deficient_state():
    from quantum_utils import measurement_distribution, depolarize
    rho, sigma = _states(7)
    Q = povm_positive_eigen(rho, eps=1e-6)
    assert Q.shape[0] == 7

    p_nom = measurement_distribution(Q, rho, append_null=False)
    assert p_nom.shape == (7,)
    assert abs(p_nom.sum() - 1) < 1e-5          # float32-normalised stored state

    nu = 0.05
    p_true = measurement_distribution(Q, depolarize(rho, nu))   # append_null=True by default
    assert p_true.shape == (8,)
    assert abs(p_true[-1] - nu * (N - 7) / N) < 1e-6          # null outcome: 0.00625
    assert np.allclose(p_true[:7], (1 - nu) * p_nom + nu / N)

    q = measurement_distribution(Q, sigma)
    assert q.shape == (8,) and q[-1] > 0 and abs(q.sum() - 1) < 1e-9


def test_measurement_distribution_full_rank_has_no_null_outcome():
    from quantum_utils import measurement_distribution, depolarize
    rho, _ = _states(8)
    Q = povm_positive_eigen(rho, eps=1e-6)
    p_nom = measurement_distribution(Q, rho)
    assert p_nom.shape == (8,)
    p_true = measurement_distribution(Q, depolarize(rho, 0.05))
    assert p_true.shape == (8,)
    assert np.allclose(p_true, 0.95 * p_nom + 0.05 / N)


# --------------------------------------------------------------------------
# cusum_fap_mismatch
# --------------------------------------------------------------------------

def test_fap_mismatch_reduces_to_existing_fap_without_noise():
    from quantum_utils import measurement_distribution
    from quantum_detection import cusum_fap_mismatch
    rho, _ = _states(8)
    Q = povm_positive_eigen(rho, eps=1e-6)
    p = measurement_distribution(Q, rho)
    h = np.array([0.5, 1.0, 1.5, 2.0])
    np.random.seed(0)
    ref = cusum_fap_exp_fast(4096, p, h, forget_val=0.99, batch_size=4096)
    new = cusum_fap_mismatch(4096, p, p, h, forget_val=0.99, batch_size=4096,
                             rng=np.random.default_rng(1))
    assert np.allclose(new, ref, rtol=0.10), (new, ref)


def test_fap_mismatch_null_outcome_caps_fap_at_inverse_probability():
    from quantum_utils import measurement_distribution, depolarize
    from quantum_detection import cusum_fap_mismatch
    rho, _ = _states(7)
    Q = povm_positive_eigen(rho, eps=1e-6)
    p_nom = measurement_distribution(Q, rho, append_null=False)     # length 7
    p_true = measurement_distribution(Q, depolarize(rho, 0.05))     # length 8, last = 1/160
    fap = cusum_fap_mismatch(4096, p_true, p_nom, np.array([1e9]), forget_val=0.99,
                             batch_size=4096, rng=np.random.default_rng(2))
    expected = N / (0.05 * (N - 7))                                # 160
    assert abs(fap[0] - expected) / expected < 0.08, fap


def test_fap_mismatch_noise_shortens_false_alarm_period_for_full_rank_state():
    from quantum_utils import measurement_distribution, depolarize
    from quantum_detection import cusum_fap_mismatch
    rho, _ = _states(8)
    Q = povm_positive_eigen(rho, eps=1e-6)
    p_nom = measurement_distribution(Q, rho)
    p_true = measurement_distribution(Q, depolarize(rho, 0.05))
    h = np.array([3.0])
    clean = cusum_fap_mismatch(2048, p_nom, p_nom, h, forget_val=0.99, rng=np.random.default_rng(3))
    noisy = cusum_fap_mismatch(2048, p_true, p_nom, h, forget_val=0.99, rng=np.random.default_rng(4))
    assert noisy[0] < clean[0], (noisy, clean)


# --------------------------------------------------------------------------
# cusum_add_est_fast
# --------------------------------------------------------------------------

def test_add_est_fast_reduces_to_existing_add_full_rank():
    from quantum_utils import measurement_distribution
    from quantum_detection import cusum_add_est_fast
    rho, sigma = _states(8)
    Q = povm_positive_eigen(rho, eps=1e-6)
    p = measurement_distribution(Q, rho)
    q = measurement_distribution(Q, sigma)
    h = np.array([0.5, 1.5, 3.0])
    np.random.seed(0)
    ref = cusum_add_exp_single(3000, p, q, h, forget_val=0.99)
    new = cusum_add_est_fast(20000, p, q, h, forget_val=0.99, rng=np.random.default_rng(5))
    assert np.allclose(new, ref, rtol=0.10), (new, ref)


def test_add_est_fast_reduces_to_existing_add_rank_deficient():
    from quantum_utils import measurement_distribution
    from quantum_detection import cusum_add_est_fast
    rho, sigma = _states(7)
    Q = povm_positive_eigen(rho, eps=1e-6)
    p = measurement_distribution(Q, rho, append_null=False)   # length 7
    q = measurement_distribution(Q, sigma)                    # length 8, null outcome last
    h = np.array([0.5, 1.5, 3.0])
    np.random.seed(0)
    ref = cusum_add_exp_single(3000, p, q, h, forget_val=0.99)
    new = cusum_add_est_fast(20000, p, q, h, forget_val=0.99, rng=np.random.default_rng(6))
    assert np.allclose(new, ref, rtol=0.10), (new, ref)


# --------------------------------------------------------------------------
# experiment_noise_robustness (integration, tiny sizes)
# --------------------------------------------------------------------------

def test_experiment_writes_figure_and_data():
    from analysis import experiment_noise_robustness
    os.makedirs(SCRATCH, exist_ok=True)
    fig_path = os.path.join(SCRATCH, "noise_fig.png")
    if os.path.exists(fig_path):
        os.remove(fig_path)
    results = experiment_noise_robustness(
        nu_list=(0.0, 0.01), R_list=(8, 7), set_index=SET_INDEX,
        test_size=64, add_multi=2, h_vec=np.linspace(0, 1, 5), min_fap=0,
        out_dir=SCRATCH, fig_path=fig_path, show=False, seed=0)
    assert os.path.exists(fig_path)
    assert set(results.keys()) == {(8, 0.0), (8, 0.01), (7, 0.0), (7, 0.01)}
    for (R, nu), res in results.items():
        assert res["fap"].shape == (5,) and res["add"].shape == (5,)
        assert np.all(np.isfinite(res["fap"])) and np.all(np.isfinite(res["add"]))
    # ADD is independent of the pre-change noise by construction (sigma unchanged)
    assert np.allclose(results[(8, 0.0)]["add"], results[(8, 0.01)]["add"])
    for R in (8, 7):
        assert os.path.exists(os.path.join(SCRATCH, f"noise_N8_R{R}_idx{SET_INDEX}.npz"))


def test_experiment_extends_threshold_grid_until_min_fap_is_reached():
    from analysis import experiment_noise_robustness
    os.makedirs(SCRATCH, exist_ok=True)
    results = experiment_noise_robustness(
        nu_list=(0.0,), R_list=(8,), set_index=SET_INDEX,
        test_size=64, add_multi=2, h_vec=np.linspace(0, 1, 5), min_fap=300,
        out_dir=SCRATCH, fig_path=os.path.join(SCRATCH, "ext.png"), show=False, seed=0)
    res = results[(8, 0.0)]
    assert res["h"].max() > 1.0                       # grid was extended beyond the base grid
    assert res["fap"].max() >= 300, res["fap"].max()
    assert res["h"].shape == res["fap"].shape == res["add"].shape
    assert np.all(np.diff(res["h"]) > 0)


def test_experiment_stops_extending_when_fap_is_capped_by_null_outcome():
    from analysis import experiment_noise_robustness
    os.makedirs(SCRATCH, exist_ok=True)
    results = experiment_noise_robustness(
        nu_list=(0.05,), R_list=(7,), set_index=SET_INDEX,
        test_size=256, add_multi=1, h_vec=np.linspace(0, 1, 5), min_fap=5000, h_ceiling=6.0,
        out_dir=SCRATCH, fig_path=os.path.join(SCRATCH, "cap.png"), show=False, seed=0)
    res = results[(7, 0.05)]
    assert res["fap"].max() < 200                     # cap is 160: extension cannot help
    assert res["h"].max() <= 6.0 + 1e-9               # and the search gave up at the ceiling


def test_noise_aware_null_outcome_rule_lets_rank_deficient_fap_grow_past_cap():
    from analysis import experiment_noise_robustness
    os.makedirs(SCRATCH, exist_ok=True)
    results = experiment_noise_robustness(
        nu_list=(0.0, 0.05), R_list=(7,), set_index=SET_INDEX,
        test_size=256, add_multi=2, h_vec=np.linspace(0, 1, 5), min_fap=500,
        null_mode="regular", out_dir=SCRATCH, fig_path=os.path.join(SCRATCH, "reg.png"),
        show=False, seed=0)
    noisy = results[(7, 0.05)]
    assert noisy["fap"].max() >= 500                  # no longer capped at 160
    assert noisy["p_nom"].shape == (8,) and abs(noisy["p_nom"][-1] - 0.05 / 8) < 1e-6
    base = results[(7, 0.0)]
    assert base["p_nom"].shape == (7,)                # nu = 0 keeps the one-shot rule


def test_extended_fap_curve_is_monotone_in_threshold():
    """The final FAP must come from one simulation over the whole extended grid:
    the same runs cross increasing thresholds in order, so FAP is non-decreasing
    exactly, without stitching noise between extension chunks."""
    from analysis import experiment_noise_robustness
    os.makedirs(SCRATCH, exist_ok=True)
    results = experiment_noise_robustness(
        nu_list=(0.0,), R_list=(8,), set_index=SET_INDEX,
        test_size=8, add_multi=1, h_vec=np.linspace(0, 1, 5), min_fap=1e9, h_ceiling=6.0,
        out_dir=SCRATCH, fig_path=os.path.join(SCRATCH, "mono.png"), show=False, seed=0)
    res = results[(8, 0.0)]
    assert res["h"].max() >= 6.0 - 1e-9                       # several chunks were appended
    assert np.all(np.diff(res["fap"]) >= 0), res["fap"]
    assert np.all(np.diff(res["add"]) >= 0), res["add"]


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for f in tests:
        try:
            f()
            print(f"PASS  {f.__name__}")
        except Exception as e:          # noqa: BLE001
            failed += 1
            print(f"FAIL  {f.__name__} -> {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
