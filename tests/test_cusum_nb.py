"""
Tests for quantum_qcd/cusum_nb.py -- the Monte-Carlo CUSUM routines of
paper Figs. 6, 7, 11, 12 and 15.

Run from anywhere:  python3 code_release/tests/test_cusum_nb.py
No pytest needed; each test_* function is executed by the runner at the bottom.

Two layers:
  1. ``cusum_nb._selftest()`` -- the module's own self-test, which compares every
     ported routine with a direct transcription of its notebook cell (bit-exact
     under a shared RandomState where the notebook was already batched, within
     Monte-Carlo error against the serial cells otherwise).
  2. structural cross-checks against the diagnostics printed in the source
     notebooks, recomputed here from the shipped state files.
"""

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "quantum_qcd"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cusum_nb  # noqa: E402
from cusum_nb import (  # noqa: E402
    append_null_outcome,
    augment_with_null_outcome,
    cusum_add_batch,
    cusum_add_batch_rank,
    cusum_add_cond_est_batch,
    cusum_add_cond_exact_batch,
    cusum_add_exp_batch,
    cusum_fap_batch,
    cusum_fap_exp_batch,
)

N = 8
SET_INDEX = 9              # the state pair used by Figs. 6, 7, 11 and 12
STATE_DIR = os.path.join(
    ROOT, "notebooks_raw", "Quantum_Anomaly_Detection", "optimization_known",
    "adaptive_gradient", "adap_grad_jeremy")


def _max_sensitivity_pq(R):
    """p, q of the maximum-sensitivity (eigenbasis) measurement, as in Fig. 11/12."""
    from quantum_utils import povm_positive_eigen
    sigma = np.load(os.path.join(STATE_DIR, "sigma_8.npy"))
    rho = np.load(os.path.join(STATE_DIR,
                               f"rho_8_rank_{R}_set_500.npy"))[SET_INDEX]
    Q = povm_positive_eigen(rho, eps=1e-6)
    p = np.diagonal(Q @ rho, offset=0, axis1=-1, axis2=-2).sum(-1).real
    q = np.diagonal(Q @ sigma, offset=0, axis1=-1, axis2=-2).sum(-1).real
    return p, q


# --------------------------------------------------------------------------
# 1. the module self-test (notebook transcriptions)
# --------------------------------------------------------------------------

def test_module_selftest_passes():
    assert cusum_nb._selftest(verbose=False) == 0


# --------------------------------------------------------------------------
# 2. statistical definitions the figures depend on
# --------------------------------------------------------------------------

def test_cusum_fap_batch_does_not_normalise_q():
    """Figs. 6, 7 and 15 rely on the sub-normalised q of a rank-deficient pair;
    scaling q must therefore change the answer (a normalising implementation
    would return the same curve for both)."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.30, 0.25, 0.20])          # sums to 0.75
    h = np.linspace(0, 1.5, 4)
    sub = cusum_fap_batch(300, p, q, h, max_time=2_000, seed=1)
    nrm = cusum_fap_batch(300, p, q / q.sum(), h, max_time=2_000, seed=1)
    assert not np.allclose(sub, nrm)
    assert np.all(sub >= nrm - 1e-12)         # a smaller LLR delays every alarm


def test_cusum_fap_batch_censors_at_max_time():
    """A threshold that can never be reached must return exactly max_time."""
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])                  # LLR == 0, s stays at 0 forever
    h = np.array([0.0, 1.0])
    for max_time in (50, 137):
        fap = cusum_fap_batch(64, p, q, h, max_time=max_time, seed=3)
        assert fap[0] == 1.0                  # h = 0 fires at t = 1
        assert fap[1] == float(max_time)


def test_cusum_add_batch_rank_extra_symbol_fires_every_threshold():
    """With an unreachable threshold vector, every run stops on the extra symbol
    and the ADD equals the mean of a geometric with that symbol's probability."""
    p = np.array([0.5, 0.3, 0.2])
    q_aug = np.array([0.20, 0.15, 0.15, 0.50])
    h = np.array([40.0, 50.0])
    add = cusum_add_batch_rank(20_000, p, q_aug, h, seed=7)
    assert np.allclose(add[0], add[1])                    # both fire together
    assert abs(add[0] - 1 / 0.5) < 0.05                   # geometric mean 1/q_null


def test_conditional_add_empty_bucket_is_zero_not_nan():
    """The 1e-10 counters of adap_grad cell 64 / Sensitivity cell 31: a stopping
    cause that never occurs gives an identically-zero curve.  This is what pins
    the R = 5 dashed curve of Figs. 7 and 12 to the x axis."""
    p = np.array([0.5, 0.3, 0.2])
    q_aug = np.array([0.20, 0.15, 0.15, 0.50])
    h = np.array([40.0, 50.0])                # never crossed
    for fn, kw in ((cusum_add_cond_exact_batch, {}),
                   (cusum_add_cond_est_batch, {"forget_val": 0.99})):
        add_xt, add_st, c_xt, c_st = fn(2_000, p, q_aug, h, seed=11, **kw)
        assert c_st == 1e-10 and np.all(add_st == 0.0)
        assert np.isfinite(add_st).all()
        assert abs(c_xt - 2_000) < 1e-6 and np.all(add_xt > 0)


def test_conditional_add_buckets_partition_the_runs():
    p = np.array([0.45, 0.30, 0.25])
    q_aug = append_null_outcome(np.array([0.20, 0.28, 0.50]))
    h = np.linspace(0, 1.6, 5)
    _, _, c_xt, c_st = cusum_add_cond_exact_batch(5_000, p, q_aug, h, seed=13)
    assert abs((c_xt - 1e-10) + (c_st - 1e-10) - 5_000) < 1e-6
    assert c_xt > 1 and c_st > 1                          # both causes occur here


def test_seeds_are_honoured_by_every_public_routine():
    p = np.array([0.45, 0.30, 0.25])
    q_short = np.array([0.20, 0.28, 0.50])
    q_aug = append_null_outcome(q_short)
    p_aug, q_aug15 = augment_with_null_outcome(p, q_short)
    h = np.linspace(0, 1.4, 4)
    calls = [
        (cusum_fap_batch, (200, p, q_short, h), {"max_time": 2_000}),
        (cusum_add_batch, (200, p, q_short / q_short.sum(), h), {}),
        (cusum_add_batch_rank, (200, p, q_aug, h), {}),
        (cusum_add_cond_exact_batch, (200, p, q_aug, h), {}),
        (cusum_add_cond_est_batch, (200, p, q_aug, h, 0.99), {}),
        (cusum_fap_exp_batch, (200, p_aug, h, 0.99), {"max_steps": 2_000}),
        (cusum_add_exp_batch, (200, p_aug, q_aug15, h, 0.99), {"max_steps": 2_000}),
        (cusum_nb.cusum_fap_exp_single_batch, (200, p, h), {}),
        (cusum_nb.cusum_add_exp_single_batch, (200, p, q_aug, h), {}),
        (cusum_nb.cusum_add_exp_single_exact_batch, (200, p, q_aug, h), {}),
    ]
    for fn, args, kw in calls:
        a = fn(*args, seed=5, **kw)
        b = fn(*args, seed=5, **kw)
        c = fn(*args, seed=6, **kw)
        a0, b0, c0 = (np.asarray(x[0] if isinstance(x, tuple) else x)
                      for x in (a, b, c))
        assert np.array_equal(a0, b0), f"{fn.__name__} is not seed-reproducible"
        assert not np.array_equal(a0, c0), f"{fn.__name__} ignores the seed"

    # an explicit Generator / RandomState must be accepted as well
    assert np.array_equal(
        cusum_fap_batch(100, p, q_short, h, seed=np.random.default_rng(2)),
        cusum_fap_batch(100, p, q_short, h, seed=np.random.default_rng(2)))


def test_exp_window_wrappers_reject_mismatched_lengths():
    """probing_known cell 15 augments *both* pmfs; a length mismatch is a bug,
    not a null-space outcome (that is quantum_detection's convention, not this
    module's)."""
    p = np.array([0.5, 0.3, 0.2])
    q_aug = append_null_outcome(np.array([0.2, 0.3, 0.4]))
    try:
        cusum_add_exp_batch(10, p, q_aug, np.array([1.0]), 0.99, max_steps=10)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for len(q) != len(p)")


# --------------------------------------------------------------------------
# 3. cross-checks against the diagnostics printed in the source notebooks
# --------------------------------------------------------------------------

def test_null_outcome_masses_match_the_notebook_printout():
    """Sensitivity_FAP_ADD_rank_def cell 25 / 32 printed q_last = 0.394 / 0.282 /
    0.097 for R = 5 / 6 / 7."""
    if not os.path.isdir(STATE_DIR):
        print("    (skipped: state files not present)")
        return
    for R, want in ((5, 0.394), (6, 0.282), (7, 0.097)):
        _, q = _max_sensitivity_pq(R)
        assert abs(append_null_outcome(q)[-1] - want) < 5e-4, (R, q.sum())


def test_fig12_conditional_add_reproduces_the_notebook_diagnostics():
    """Reported by Sensitivity_FAP_ADD_rank_def cell 32 at 5e6 trials:

        R = 5  estimated  counters (5000000, 0)      ADD_xt 2.54
        R = 6  estimated  counters (4999629, 371)    ADD_xt 3.54
        R = 6  exact      counters (4975730, 24270)
        R = 7  exact      counters (3865741, 1134259)

    and the estimated-q ADD_xt at the first threshold whose FAP reaches 5000 was
    2.54 / 3.54 / 9.54 for R = 5 / 6 / 7.

    At R = 5 and R = 6 the s_t > h bucket is essentially empty, so ADD_xt has
    saturated by the top of the threshold grid at the mean of a geometric with
    the null-outcome probability, 1/q_last = 2.538 / 3.543 -- exactly the printed
    numbers.  200 000 trials are enough to check that saturation value, the
    printed ADD_xt (to +/- 0.15, since the notebook quotes it at the FAP = 5000
    threshold rather than at the top of the grid) and the fraction of runs in
    each bucket.
    """
    if not os.path.isdir(STATE_DIR):
        print("    (skipped: state files not present)")
        return
    n = 200_000
    for R, h_top, add_xt_ref, frac_st_est, frac_st_exact in (
            (5, 4.5, 2.54, 0.0, 0.0),
            (6, 5.0, 3.54, 371 / 5e6, 24270 / 5e6),
            (7, 5.0, 9.54, 105885 / 5e6, 1134259 / 5e6)):
        p, q = _max_sensitivity_pq(R)
        q = append_null_outcome(q)
        h = np.linspace(0, h_top, 100)

        xt, st, c_xt, c_st = cusum_add_cond_est_batch(n, p, q, h, 0.99, seed=R,
                                                      batch_size=50_000)
        assert abs(xt[-1] - add_xt_ref) < 0.15, (R, "est ADD_xt", xt[-1],
                                                 add_xt_ref)
        if frac_st_est < 1e-4:      # nothing crosses: the geometric identity holds
            assert abs(xt[-1] - 1 / q[-1]) < 0.05, (R, "est", xt[-1], 1 / q[-1])
        got = (c_st - 1e-10) / n
        assert abs(got - frac_st_est) < max(3e-3, 0.25 * frac_st_est), \
            (R, "est s_t>h fraction", got, frac_st_est)
        if frac_st_est == 0.0:
            assert np.all(st == 0.0)          # the flat dashed R = 5 curve

        xt, st, c_xt, c_st = cusum_add_cond_exact_batch(n, p, q, h, seed=100 + R,
                                                        batch_size=50_000)
        got = (c_st - 1e-10) / n
        assert abs(got - frac_st_exact) < max(3e-3, 0.25 * frac_st_exact), \
            (R, "exact s_t>h fraction", got, frac_st_exact)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
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
