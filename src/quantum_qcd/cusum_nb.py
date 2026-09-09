"""
Monte-Carlo CUSUM routines used by paper Figures 6, 7, 11, 12 and 15.

This module collects the CUSUM simulators that live in the analysis notebooks but
are *not* in :mod:`quantum_qcd.quantum_detection`, plus seeded / batched
equivalents of the notebook routines that are.  Every public routine takes a
``seed`` (an ``int``, a ``numpy.random.Generator`` or a
``numpy.random.RandomState``) and is vectorised over Monte-Carlo trials.

Provenance
----------
Three notebooks under ``notebooks_raw/Quantum_Anomaly_Detection`` supply the
definitions.  Each function below was AST-diffed (docstrings and comments
stripped) against the same-named function in ``quantum_detection.py``:

``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho.ipynb``
    cell 3  -> ``cusum_fap_batch``, ``cusum_add_batch``   (no counterpart: ported)
    cell 4  -> ``cusum_add_batch_rank``                   (no counterpart: ported)
    cell 62 -> the Fig. 6/7 parameter block (N = 8, n_sets = 500, test_size)
    cell 64 -> ``cusum_add``  == ``quantum_detection.cusum_add_exact`` (re-exported;
               batched port ``cusum_add_cond_exact_batch``)

``optimization_unknown/Sensitivity_FAP_ADD_rank_def.ipynb``
    cell 10 -> ``generate_pd_matrix`` / ``generate_pd_matrix_rank``  (already in
               ``quantum_states.py``; not repeated here)
    cell 11 -> ``generate_X_single`` / ``generate_sigma`` /
               ``generate_rho_set_rank``  (already in ``quantum_states.py``)
    cell 15 -> ``povm_positive_eigen``   (already in ``quantum_utils.py``)
    cell 30 -> ``cusum_fap_exp_fast``    == ``quantum_detection.cusum_fap_exp_fast``
               (re-exported; seeded wrapper ``cusum_fap_exp_fast_seeded``)
    cell 31 -> ``cusum_add_exact`` / ``cusum_add_est`` ==
               ``quantum_detection.cusum_add_exact`` / ``.cusum_add_est``
               (re-exported; batched ports ``cusum_add_cond_*_batch``)
    cells 4/5/6 (Fig. 11 helpers) -> ``cusum_fap_exp_single`` /
               ``cusum_add_exp_single`` / ``cusum_add_exp_single_exact``, all
               identical to ``quantum_detection`` (re-exported; batched ports
               ``cusum_*_exp_single_batch``)

``probe_known/probing_known.ipynb``
    cell 4  -> ``cusum_fap_batch``, ``cusum_add_batch``  (identical to the
               adap_grad cell-3 versions)
    cell 5  -> ``cusum_add_batch_rank``                  (identical to cell 4 above)
    cell 15 -> ``_simulate_exp_window`` + ``cusum_fap_exp_batch`` /
               ``cusum_add_exp_batch``                   (no counterpart: ported)
    cells 101/102/106/112 -> the Fig. 15 call sites (``exp_val = 0.99``,
               ``max_time`` = 50 000 / 100 000, the p/q null-outcome augmentation)

Which figure calls what
-----------------------
Fig. 6  (known q, ADD vs FAP by rank)
        ``cusum_fap_batch`` (max_time = 20 000), ``cusum_add_batch_rank`` for
        R < N, ``cusum_add_batch`` for R = N.
Fig. 7  (known q, ADD conditioned on the stopping cause)
        ``cusum_fap_batch``, ``cusum_add_cond_exact_batch`` (cell 64).
Fig. 11 (unknown q, estimated vs exact, by rank)
        ``cusum_fap_exp_single_batch``, ``cusum_add_exp_single_batch`` (panel a),
        ``cusum_add_exp_single_exact_batch`` (panel b).
Fig. 12 (unknown q, ADD conditioned on the stopping cause)
        ``cusum_fap_exp_fast_seeded`` (batch_size = 1000),
        ``cusum_add_cond_est_batch`` and ``cusum_add_cond_exact_batch``.
Fig. 15 (probing / channel case)
        ``cusum_fap_batch`` + ``cusum_add_batch_rank`` / ``cusum_add_batch`` for
        the maximum-KL curves, and ``cusum_fap_exp_batch`` /
        ``cusum_add_exp_batch`` (exp_val = 0.99, max_steps = 50 000) for the
        maximum-sensitivity curves.

Statistical conventions preserved verbatim from the notebooks
-------------------------------------------------------------
* ``h_vec`` is a *vector* of thresholds; one simulated trajectory feeds every
  threshold at once, so the resulting (FAP, ADD) pairs are perfectly correlated
  across thresholds -- that is what makes the published curves smooth.  A run
  stops when the **largest** threshold fires.
* ``cusum_fap_batch`` normalises **only** ``p_dist``.  The log-likelihood ratio
  therefore uses a deliberately sub-normalised ``q_dist`` whenever the
  rank-deficient post-change distribution has leaked mass outside the support of
  the pre-change state.  Normalising ``q_dist`` would change every rank-deficient
  curve in Figs. 6, 7 and 15.
* The ``max_time`` cap (20 000 in adap_grad cell 3, raised to 50 000 / 100 000 at
  the Fig. 15 call sites) censors a run at ``max_time`` and credits ``max_time``
  to every threshold that has not fired yet.
* The stopping-cause-conditioned ADD keeps two accumulators, one for runs that
  stopped because the null-space outcome was observed (``x_t = 0`` in the paper's
  notation) and one for runs that stopped because ``s_t > h``.  Both counters are
  initialised at ``1e-10``, so an empty bucket divides ``0`` by ``1e-10`` and
  yields an identically-zero curve (this is what pins the R = 5 dashed curve of
  Figs. 7 and 12 to the x axis; it is not a bug to be fixed).
* The null-space ("extra", "exclusive") outcome is always the **last** entry of
  the post-change pmf, appended as ``np.append(q, 1 - q.sum())``.  In the channel
  case of Fig. 15 the pre-change pmf is augmented as well, with the small but
  non-zero mass ``np.append(p, 1e-4)``, so the exponential-window estimator can
  form a finite log-likelihood ratio for that outcome.
* The exponential-window update ``q_hat <- forget * q_hat`` followed by
  ``q_hat[x] += 1 - forget`` is applied **after** the log-likelihood ratio and the
  threshold test, and ``q_hat`` is initialised at the pre-change pmf.

Deliberate deviations from the notebook text
--------------------------------------------
1. Every routine takes ``seed``.  The notebooks used the un-seeded global RNG.
2. Input arrays are copied before normalisation.  ``_simulate_exp_window`` in
   probing_known cell 15 normalised its inputs in place with ``/=`` after
   ``np.asanyarray``, silently mutating the caller's arrays; the results are
   unchanged (re-normalising a normalised pmf is a no-op) but the side effect is
   gone.
3. ``batch_size`` splits the trials into chunks so that 5e6-trial runs fit in
   memory.  Chunking changes nothing statistically (trials are i.i.d. and the
   accumulators are plain sums) but it does change the order in which random
   numbers are drawn, so two runs are bit-identical only at equal ``batch_size``;
   ``batch_size=None`` reproduces the notebook's single-block allocation exactly.
4. The per-threshold Python loops of ``cusum_fap_batch`` (the censoring branch)
   and ``_simulate_exp_window`` (the crossing test) are vectorised.  They consume
   no randomness, so the results are bit-identical.

Reproducing a notebook run bit-for-bit
--------------------------------------
The ported batch routines draw their observations in exactly the notebook's
order, so passing ``seed=np.random.RandomState(k)`` reproduces a notebook cell
run under ``np.random.seed(k)`` bit-for-bit.  Passing an ``int`` uses the modern
``np.random.default_rng`` stream instead (recommended for new runs).  The
re-exported legacy routines use the global RNG; wrap them in
:func:`legacy_seed` or call them through :func:`run_seeded`.

Run ``python3 cusum_nb.py`` to execute the self-test.
"""

from __future__ import annotations

import contextlib

import numpy as np

try:  # imported as part of the quantum_qcd package
    from .quantum_detection import (
        cusum_add_est,
        cusum_add_est_fast,
        cusum_add_exact,
        cusum_add_exp_single,
        cusum_add_exp_single_exact,
        cusum_fap_exp_fast,
        cusum_fap_exp_single,
        cusum_fap_mismatch,
    )
except ImportError:  # imported flat, as the release scripts and tests do
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quantum_detection import (  # type: ignore  # noqa: E402
        cusum_add_est,
        cusum_add_est_fast,
        cusum_add_exact,
        cusum_add_exp_single,
        cusum_add_exp_single_exact,
        cusum_fap_exp_fast,
        cusum_fap_exp_single,
        cusum_fap_mismatch,
    )

# ``cusum_add`` is the name adap_grad_rank_defic_rho cell 64 gives to the routine
# that Sensitivity_FAP_ADD_rank_def cell 31 calls ``cusum_add_exact``.  The two
# cell bodies are byte-identical after comment stripping, and both are identical
# to quantum_detection.cusum_add_exact, so the notebook name is kept as an alias.
cusum_add = cusum_add_exact

__all__ = [
    # --- re-exported unchanged from quantum_detection (notebook-identical) ---
    "cusum_fap_exp_single",
    "cusum_add_exp_single",
    "cusum_add_exp_single_exact",
    "cusum_fap_exp_fast",
    "cusum_add_exact",
    "cusum_add",
    "cusum_add_est",
    "cusum_fap_mismatch",
    "cusum_add_est_fast",
    # --- seeding helpers for the legacy (global-RNG) re-exports ---
    "legacy_seed",
    "run_seeded",
    "cusum_fap_exp_fast_seeded",
    # --- pmf helpers ---
    "append_null_outcome",
    "augment_with_null_outcome",
    # --- ported notebook routines (known q, Figs. 6, 7, 15) ---
    "cusum_fap_batch",
    "cusum_add_batch",
    "cusum_add_batch_rank",
    # --- ported notebook routines (stopping-cause conditioned, Figs. 7, 12) ---
    "cusum_add_cond_exact_batch",
    "cusum_add_cond_est_batch",
    # --- ported notebook routines (exponential window, Fig. 15) ---
    "cusum_fap_exp_batch",
    "cusum_add_exp_batch",
    # --- seeded, batched equivalents of the Fig. 11 serial routines ---
    "cusum_fap_exp_single_batch",
    "cusum_add_exp_single_batch",
    "cusum_add_exp_single_exact_batch",
]


# ---------------------------------------------------------------------------
# RNG plumbing
# ---------------------------------------------------------------------------

def _as_rng(seed):
    """
    Coerce ``seed`` into an object with a ``.choice(a, size=, p=)`` method.

    ``None``                      -> a fresh ``default_rng()``
    ``int`` / ``SeedSequence``    -> ``default_rng(seed)``
    ``Generator``/``RandomState`` -> returned unchanged

    Pass a ``RandomState`` to reproduce a notebook cell that was run under
    ``np.random.seed(k)``: the legacy ``np.random.choice`` is
    ``RandomState.choice``, and the ported routines draw in the notebook's order.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        return seed
    return np.random.default_rng(seed)


@contextlib.contextmanager
def legacy_seed(seed):
    """
    Temporarily seed the global (legacy) RNG used by the re-exported routines.

    The routines re-exported from ``quantum_detection`` are byte-identical
    transcriptions of the notebook cells and therefore call ``np.random.choice``,
    i.e. the process-wide ``RandomState``.  ``with legacy_seed(0): ...`` makes
    them reproducible and restores the previous global state on exit.
    ``seed=None`` is a no-op (fresh entropy, as in the notebooks).
    """
    if seed is None:
        yield
        return
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(
            "legacy_seed needs an int (the legacy routines use np.random.*); "
            f"got {type(seed).__name__}"
        )
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def run_seeded(func, *args, seed=None, **kwargs):
    """
    Call one of the legacy re-exports reproducibly.

    Example
    -------
    >>> fap = run_seeded(cusum_fap_exp_single, 32, p, h_vec,
    ...                  forget_val=0.99, seed=0)      # doctest: +SKIP
    """
    with legacy_seed(seed):
        return func(*args, **kwargs)


def cusum_fap_exp_fast_seeded(test_size, p_dist, h_vec, forget_val=0.99,
                              batch_size=5000, seed=None):
    """
    Seeded call of the re-exported :func:`cusum_fap_exp_fast` (Fig. 12 FAP axis).

    Thin wrapper -- the simulator itself is the unmodified
    ``quantum_detection.cusum_fap_exp_fast``, i.e. the byte-identical
    transcription of Sensitivity_FAP_ADD_rank_def cell 30, including its quirk of
    dividing by ``n_batches * batch_size`` rather than by ``test_size`` (keep
    ``test_size`` a multiple of ``batch_size``; Fig. 12 uses 50 000 and 1000).
    """
    with legacy_seed(seed):
        return cusum_fap_exp_fast(test_size, p_dist, h_vec,
                                  forget_val=forget_val, batch_size=batch_size)


# ---------------------------------------------------------------------------
# pmf helpers (the null-space outcome augmentation)
# ---------------------------------------------------------------------------

def append_null_outcome(q_dist):
    """
    ``np.append(q, 1 - q.sum())`` -- the augmentation used by Figs. 6, 7, 11, 12.

    ``q_dist`` holds the post-change probabilities of the R measurement outcomes
    that live in the support of the pre-change state.  Its deficit
    ``1 - sum(q)`` is the probability that the post-change state clicks outside
    that support; it becomes outcome index R, which is impossible before the
    change and therefore raises an alarm at every threshold at once.
    """
    q_dist = np.asarray(q_dist, dtype=float)
    return np.append(q_dist, 1.0 - q_dist.sum())


def augment_with_null_outcome(p_dist, q_dist, p_null=1e-4):
    """
    The Fig. 15 (channel-case) augmentation, probing_known cell 106::

        p = np.append(p, 1e-4)
        q = np.append(q, 1 - q.sum())

    Unlike :func:`append_null_outcome` this also extends the *pre-change* pmf, with
    a small but non-zero mass, so that the exponential-window estimator can form a
    finite log-likelihood ratio ``log(q_hat[R] / p[R])`` for the null outcome
    instead of stopping the run on sight.  The returned pmfs have equal length
    R + 1 and are normalised inside the simulators (``p`` sums to
    ``1 + p_null`` before normalisation, exactly as in the notebook).

    Returns
    -------
    (p_aug, q_aug) : tuple of ndarray, both of length R + 1
    """
    p_dist = np.asarray(p_dist, dtype=float)
    q_dist = np.asarray(q_dist, dtype=float)
    return np.append(p_dist, p_null), append_null_outcome(q_dist)


def _batch_sizes(test_size, batch_size):
    """Yield chunk sizes summing to ``test_size`` (one chunk if batch_size is None)."""
    test_size = int(test_size)
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    step = test_size if batch_size is None else int(batch_size)
    if step <= 0:
        raise ValueError("batch_size must be positive")
    remaining = test_size
    while remaining > 0:
        b = min(step, remaining)
        remaining -= b
        yield b


# ---------------------------------------------------------------------------
# Known post-change distribution: the batched routines of
# adap_grad_rank_defic_rho cells 3-4 == probing_known cells 4-5 (Figs. 6, 7, 15)
# ---------------------------------------------------------------------------

def cusum_fap_batch(test_size, p_dist, q_dist, h_vec, max_time=20_000,
                    seed=None, batch_size=None):
    """
    False alarm period of the CUSUM test with a **known** post-change pmf.

    Port of ``cusum_fap_batch`` (adap_grad_rank_defic_rho cell 3 == probing_known
    cell 4).  Used by Figs. 6, 7 and 15.  Observations are drawn from ``p_dist``;
    the statistic is ``s <- max(0, s + log(q_dist[x] / p_dist[x]))`` and the mean
    first-crossing time is reported for every threshold in ``h_vec``.

    Two notebook behaviours are preserved deliberately:

    * **Only ``p_dist`` is normalised.**  For a rank-deficient state pair
      ``q_dist`` sums to ``1 - q_0 < 1`` (``q_0`` is the mass of sigma outside
      the support of rho) and the sub-normalised likelihood ratio is exactly what
      the published curves use.  Only the first ``len(p_dist)`` entries of
      ``q_dist`` are read, so an already-augmented ``q`` may be passed
      unchanged -- probing_known cell 101 does precisely that.
    * **The ``max_time`` cap censors rather than discards.**  When a run reaches
      ``max_time`` every threshold that has not fired yet is credited with
      ``max_time`` and the run is dropped *before* its update at that step, so it
      contributes ``max_time - 1`` real observations.  The cap is 20 000 in
      cell 3; Fig. 15 raises it to 50 000 (cells 101, 102) and 100 000 (cell 112).

    Parameters
    ----------
    test_size : int
        Number of Monte-Carlo runs.
    p_dist : (R,) array_like
        Pre-change pmf (normalised internally).
    q_dist : (R,) or (R+1,) array_like
        Post-change pmf used in the log-likelihood ratio; **not** normalised.
    h_vec : (K,) array_like
        Thresholds, ascending (the last one governs stopping).
    max_time : int, optional
        Censoring horizon (default 20 000, the cell-3 value).
    seed : int, Generator, RandomState or None, optional
    batch_size : int or None, optional
        Runs simulated per chunk; ``None`` (default) allocates all ``test_size``
        runs at once, exactly as the notebook does.

    Returns
    -------
    (K,) ndarray
        Average false alarm period per threshold.
    """
    rng = _as_rng(seed)
    p_dist = np.asarray(p_dist, dtype=float)
    p_dist = p_dist / np.sum(p_dist)
    q_dist = np.asarray(q_dist, dtype=float)          # NOT normalised (notebook)
    h_vec = np.asarray(h_vec, dtype=float)
    R = len(p_dist)
    num_thresh = len(h_vec)
    if len(q_dist) < R:
        raise ValueError("q_dist must have at least len(p_dist) entries")
    log_ratio = np.log(q_dist[:R] / p_dist)

    false_alarm_sum = np.zeros(num_thresh)
    for B in _batch_sizes(test_size, batch_size):
        s = np.zeros(B)
        t = np.zeros(B, dtype=np.int64)
        flag = np.zeros((B, num_thresh), dtype=bool)
        active = np.ones(B, dtype=bool)

        while np.any(active):
            # 1) advance time
            t[active] += 1

            # 1a) censor the runs that have just hit the cap
            over_cap = np.where(active & (t >= max_time))[0]
            if over_cap.size:
                new_censored = ~flag[over_cap]
                false_alarm_sum += new_censored.sum(axis=0) * max_time
                flag[over_cap] = True
                active[over_cap] = False
                if not active.any():
                    break

            # 2) CUSUM update for the remaining active runs
            active_idx = np.where(active)[0]
            x = rng.choice(R, size=active_idx.size, p=p_dist)
            s[active_idx] = np.maximum(0, s[active_idx] + log_ratio[x])

            new_triggers = (~flag[active_idx]) & (s[active_idx][:, None] >= h_vec)
            false_alarm_sum += np.sum(t[active_idx][:, None] * new_triggers, axis=0)
            flag[active_idx] |= new_triggers
            active[active_idx] = ~flag[active_idx, -1]

    return false_alarm_sum / test_size


def cusum_add_batch(test_size, p_dist, q_dist, h_vec, seed=None, batch_size=None):
    """
    Average detection delay of the CUSUM test with a **known** post-change pmf,
    no null-space outcome.

    Port of ``cusum_add_batch`` (adap_grad_rank_defic_rho cell 3 == probing_known
    cell 4).  Used for the full-rank curve of Fig. 6 (R = N), the joint-optimised
    curves of Fig. 15 (cells 102 and 112) and, because the two definitions
    coincide when ``len(q) == len(p)``, as the batched equivalent of
    ``cusum_add_exp_single_exact`` for Fig. 11's R = N panel.

    Both pmfs are normalised here (unlike :func:`cusum_fap_batch`) and must have
    the same length: observations are drawn from ``q_dist`` and indexed into
    ``p_dist``.  There is no run-length cap -- every run continues until the
    largest threshold fires.

    Returns
    -------
    (K,) ndarray
        Average detection delay per threshold.
    """
    rng = _as_rng(seed)
    p_dist = np.asarray(p_dist, dtype=float)
    q_dist = np.asarray(q_dist, dtype=float)
    p_dist = p_dist / np.sum(p_dist)
    q_dist = q_dist / np.sum(q_dist)
    h_vec = np.asarray(h_vec, dtype=float)
    if len(q_dist) != len(p_dist):
        raise ValueError(
            "cusum_add_batch needs len(q_dist) == len(p_dist); use "
            "cusum_add_batch_rank when q_dist carries the extra null outcome"
        )
    M = len(q_dist)
    num_thresh = len(h_vec)
    log_ratio = np.log(q_dist / p_dist)

    detection_delay_sum = np.zeros(num_thresh)
    for B in _batch_sizes(test_size, batch_size):
        s = np.zeros(B)
        t = np.zeros(B, dtype=np.int64)
        flag = np.zeros((B, num_thresh), dtype=bool)
        active = np.ones(B, dtype=bool)

        while np.any(active):
            active_idx = np.where(active)[0]
            t[active_idx] += 1

            x = rng.choice(M, size=active_idx.size, p=q_dist)
            s[active_idx] = np.maximum(0, s[active_idx] + log_ratio[x])

            new_triggers = (~flag[active_idx]) & (s[active_idx][:, None] >= h_vec)
            detection_delay_sum += np.sum(t[active_idx][:, None] * new_triggers,
                                          axis=0)
            flag[active_idx] |= new_triggers
            active[active_idx] = ~flag[active_idx, -1]

    return detection_delay_sum / test_size


def cusum_add_batch_rank(test_size, p_dist, q_dist, h_vec, seed=None,
                         batch_size=None):
    """
    Average detection delay when ``q_dist`` carries the extra null-space outcome.

    Port of ``cusum_add_batch_rank`` (adap_grad_rank_defic_rho cell 4 ==
    probing_known cell 5).  Used by Fig. 6 (rank-deficient R < N) and Fig. 15
    (rank-minimised measurements, cell 101).  Statistically it is also the exact
    batched equivalent of ``quantum_detection.cusum_add_exp_single_exact`` with
    ``Qi_only=False``, so Fig. 11's "exact q" panel uses it for R < N.

    A run stops as soon as **either** the CUSUM statistic exceeds a threshold
    **or** the observation is the extra symbol at index ``len(p_dist)``, which is
    impossible before the change: that observation fires every threshold that has
    not fired yet, at the current time.  ``q_dist`` must be exactly one entry
    longer than ``p_dist`` (build it with :func:`append_null_outcome`); both pmfs
    are normalised here.

    Returns
    -------
    (K,) ndarray
        Average detection delay per threshold.
    """
    rng = _as_rng(seed)
    p_dist = np.asarray(p_dist, dtype=float)
    q_dist = np.asarray(q_dist, dtype=float)
    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()
    h_vec = np.asarray(h_vec, dtype=float)

    if len(q_dist) != len(p_dist) + 1:
        raise ValueError("q_dist must be exactly one element longer than p_dist.")

    extra_symbol = len(p_dist)      # index of the R+1 outcome present only under q
    M = len(q_dist)
    K = len(h_vec)
    log_ratio = np.log(q_dist[:extra_symbol] / p_dist)

    delay_sum = np.zeros(K)
    for B in _batch_sizes(test_size, batch_size):
        s = np.zeros(B)
        t = np.zeros(B, dtype=np.int64)
        triggered = np.zeros((B, K), dtype=bool)
        active = np.ones(B, dtype=bool)

        while active.any():
            idx = np.where(active)[0]
            t[idx] += 1

            x = rng.choice(M, size=idx.size, p=q_dist)
            is_extra = (x == extra_symbol)

            # (1) ordinary observations: CUSUM update
            if (~is_extra).any():
                idx_ord = idx[~is_extra]
                s[idx_ord] = np.maximum(0, s[idx_ord] + log_ratio[x[~is_extra]])

                new_hits = (~triggered[idx_ord]) & (s[idx_ord][:, None] >= h_vec)
                delay_sum += (t[idx_ord][:, None] * new_hits).sum(axis=0)
                triggered[idx_ord] |= new_hits

            # (2) the extra symbol: instant alarm at every threshold
            if is_extra.any():
                idx_ext = idx[is_extra]
                not_yet = ~triggered[idx_ext]
                delay_sum += (t[idx_ext][:, None] * not_yet).sum(axis=0)
                triggered[idx_ext] = True

            active[idx] = ~triggered[idx, -1]

    return delay_sum / test_size


# ---------------------------------------------------------------------------
# Stopping-cause-conditioned ADD (Figs. 7 and 12)
# ---------------------------------------------------------------------------

def _cond_add_batch(test_size, p_dist, q_dist, h_vec, forget_val, seed,
                    batch_size, estimated):
    """
    Shared engine for :func:`cusum_add_cond_exact_batch` (``estimated=False``,
    adap_grad cell 64 / Sensitivity cell 31 ``cusum_add_exact``) and
    :func:`cusum_add_cond_est_batch` (``estimated=True``, Sensitivity cell 31
    ``cusum_add_est``).

    Each run keeps a per-threshold vector of first-alarm times.  When the run
    ends, that whole vector is added to one of two accumulators according to
    *why* it ended:

    ``x_t = 0``   the last outcome index was observed (the null-space click);
                  every threshold that had not fired yet is credited with the
                  current time first;
    ``s_t > h``   the largest threshold was crossed.

    Both accumulators are divided by their own counter, and both counters start
    at ``1e-10`` exactly as in the notebooks, so a cause that never occurs yields
    an identically-zero curve rather than a NaN.
    """
    rng = _as_rng(seed)
    p_dist = np.asarray(p_dist, dtype=float)
    q_dist = np.asarray(q_dist, dtype=float)
    p_dist = p_dist / np.sum(p_dist)
    q_dist = q_dist / np.sum(q_dist)
    h_vec = np.asarray(h_vec, dtype=float)

    M = len(q_dist)
    null_symbol = M - 1                 # the notebook's `x_t == len(q_dist)-1`
    K = len(h_vec)
    if len(p_dist) < null_symbol:
        raise ValueError(
            "p_dist must have at least len(q_dist)-1 entries "
            "(the last entry of q_dist is the null-space outcome)"
        )
    if estimated and len(p_dist) != M - 1:
        raise ValueError(
            "cusum_add_cond_est_batch needs len(q_dist) == len(p_dist) + 1: "
            "q_hat is initialised as np.append(p_dist, 0)"
        )
    if not estimated:
        log_ratio = np.log(q_dist[:null_symbol] / p_dist[:null_symbol])

    cusum_detection_delays_xt = np.zeros(K)
    cusum_detection_delays_st = np.zeros(K)
    counter_xt = 1e-10
    counter_st = 1e-10

    for B in _batch_sizes(test_size, batch_size):
        t = np.zeros(B, dtype=np.int64)
        s = np.zeros(B)
        flag = np.zeros((B, K), dtype=bool)
        delays = np.zeros((B, K))
        cause = np.zeros(B, dtype=np.int8)      # 0 running, 1 = x_t = 0, 2 = s_t > h
        if estimated:
            q_hat = np.tile(np.append(p_dist, 0.0), (B, 1))
        active = np.arange(B)

        while active.size:
            t[active] += 1
            x = rng.choice(M, size=active.size, p=q_dist)

            # --- stop on the null-space outcome ------------------------------
            is_null = (x == null_symbol)
            if is_null.any():
                idx = active[is_null]
                delays[idx] += t[idx][:, None] * (~flag[idx])
                flag[idx] = True
                cause[idx] = 1

            # --- ordinary observations: CUSUM update -------------------------
            keep = ~is_null
            idx_o = active[keep]
            if idx_o.size:
                x_o = x[keep]
                if estimated:
                    llr = np.log(q_hat[idx_o, x_o] / p_dist[x_o])
                else:
                    llr = log_ratio[x_o]
                s[idx_o] = np.maximum(0, s[idx_o] + llr)

                trig = (~flag[idx_o]) & (s[idx_o][:, None] >= h_vec)
                delays[idx_o] += t[idx_o][:, None] * trig
                flag[idx_o] |= trig
                cause[idx_o[flag[idx_o, -1]]] = 2

                if estimated:
                    # exponential window, applied after the LLR and the test
                    q_hat[idx_o] *= forget_val
                    q_hat[idx_o, x_o] += (1 - forget_val)

            active = active[cause[active] == 0]

        hit_xt = cause == 1
        hit_st = cause == 2
        cusum_detection_delays_xt += delays[hit_xt].sum(axis=0)
        cusum_detection_delays_st += delays[hit_st].sum(axis=0)
        counter_xt += int(hit_xt.sum())
        counter_st += int(hit_st.sum())

    return (cusum_detection_delays_xt / counter_xt,
            cusum_detection_delays_st / counter_st,
            counter_xt, counter_st)


def cusum_add_cond_exact_batch(test_size, p_dist, q_dist, h_vec, seed=None,
                               batch_size=50_000):
    """
    ADD split by stopping cause, **exact** post-change pmf.  Figs. 7 and 12.

    Seeded, batched port of ``cusum_add`` (adap_grad_rank_defic_rho cell 64) ==
    ``cusum_add_exact`` (Sensitivity_FAP_ADD_rank_def cell 31), both of which are
    byte-identical to ``quantum_detection.cusum_add_exact`` (re-exported above as
    :func:`cusum_add_exact` / :func:`cusum_add`).  The port differs from the
    re-export only in being vectorised over trials and seeded: the trial-level
    statistics, the ``1e-10`` counters and the conditioning rule are unchanged.
    Figs. 7 and 12 run 50 000 and 5 000 000 trials respectively, which the serial
    version takes tens of minutes to do.

    ``q_dist`` must carry the null-space outcome as its last entry (build it with
    :func:`append_null_outcome`); ``p_dist`` has one entry fewer.  The
    log-likelihood ratio uses the exact ``q_dist``.

    Returns
    -------
    cusum_add_vec_xt : (K,) ndarray
        Mean delay over the runs that stopped on the null-space outcome.
    cusum_add_vec_st : (K,) ndarray
        Mean delay over the runs that stopped by crossing the largest threshold.
        Identically zero when that never happens (the R = 5 curves of Figs. 7
        and 12) -- ``0 / 1e-10``, as in the notebook.
    counter_xt, counter_st : float
        The two counters, still carrying their ``1e-10`` offset.
    """
    return _cond_add_batch(test_size, p_dist, q_dist, h_vec, None, seed,
                           batch_size, estimated=False)


def cusum_add_cond_est_batch(test_size, p_dist, q_dist, h_vec, forget_val=0.99,
                             seed=None, batch_size=50_000):
    """
    ADD split by stopping cause, post-change pmf **estimated** with an
    exponential window.  Fig. 12, left panel.

    Seeded, batched port of ``cusum_add_est`` (Sensitivity_FAP_ADD_rank_def
    cell 31), byte-identical to ``quantum_detection.cusum_add_est``
    (re-exported above).  ``q_hat`` starts at ``np.append(p_dist, 0)`` -- length
    R + 1, with zero mass on the null-space outcome -- and is updated as
    ``q_hat <- forget_val * q_hat`` then ``q_hat[x] += 1 - forget_val`` *after*
    the log-likelihood ratio and the threshold test.  ``q_hat`` is never
    renormalised, exactly as in the notebook.  Fig. 12 uses
    ``forget_val = 0.99`` and 5 000 000 trials per rank.

    Returns the same four objects as :func:`cusum_add_cond_exact_batch`.
    """
    return _cond_add_batch(test_size, p_dist, q_dist, h_vec, forget_val, seed,
                           batch_size, estimated=True)


# ---------------------------------------------------------------------------
# Exponential-window engine of probing_known cell 15 (Fig. 15)
# ---------------------------------------------------------------------------

def _simulate_exp_window(test_size, p_dist, sample_dist, h_vec, forget_val,
                         max_steps=1_000_000, seed=None, batch_size=None):
    """
    Vectorised Monte-Carlo engine for the exponentially-weighted CUSUM of
    probing_known cell 15.  Port of the notebook's ``_simulate_exp_window``.

    This is **not** the same test as ``quantum_detection._exp_window_stopping_time``
    even though both run an exponential window: there, a pmf one entry longer than
    the nominal one means a null-space outcome that stops the run on sight; here,
    ``p_dist`` and ``sample_dist`` have the *same* length and the null-space
    outcome is an ordinary symbol carrying the small pre-change mass ``1e-4``
    added by :func:`augment_with_null_outcome`, so it contributes a large but
    finite log-likelihood ratio instead of an automatic alarm.  That is the
    channel-case convention of Fig. 15 (cell 106).

    Two further notebook behaviours are preserved:

    * runs that never alarm within ``max_steps`` keep a stopping time of **0**
      (``hit_t`` is zero-initialised and the mean is taken over all
      ``test_size`` runs), so an over-tight ``max_steps`` biases the curve
      downwards rather than raising an error;
    * ``q_hat`` is updated after the crossing test, for every run that was active
      at the start of the step.

    Parameters
    ----------
    test_size : int
    p_dist : (K,) array_like
        Pre-change pmf used in the log-likelihood ratio and as the initial q_hat.
    sample_dist : (K,) array_like
        pmf the observations are drawn from: ``p_dist`` for the FAP, ``q_dist``
        for the ADD.
    h_vec : (J,) array_like
    forget_val : float
        Exponential forgetting factor (0.99 in Fig. 15).
    max_steps : int, optional
        Safety cap (50 000 at the Fig. 15 call sites).
    seed, batch_size : see :func:`cusum_fap_batch`.

    Returns
    -------
    (J,) ndarray
        Average stopping time per threshold.
    """
    rng = _as_rng(seed)
    p_dist = np.array(p_dist, dtype=float)          # copy: no in-place mutation
    sample_dist = np.array(sample_dist, dtype=float)
    p_dist /= p_dist.sum()
    sample_dist /= sample_dist.sum()
    h_vec = np.asanyarray(h_vec, dtype=float)

    K = p_dist.size
    J = h_vec.size
    if sample_dist.size != K:
        raise ValueError(
            "p_dist and sample_dist must have the same length; augment both with "
            "augment_with_null_outcome() before calling"
        )
    log_p = np.log(p_dist)

    hit_sum = np.zeros(J)
    for N in _batch_sizes(test_size, batch_size):
        q_hat = np.tile(p_dist, (N, 1))             # initial EWMA = p_dist (copy)
        S = np.zeros(N)
        alarm = np.zeros((N, J), dtype=bool)
        hit_t = np.zeros((N, J), dtype=np.int64)

        t = 0
        active = np.ones(N, dtype=bool)

        while active.any() and t < max_steps:
            t += 1

            idx = np.where(active)[0]
            x = rng.choice(K, size=idx.size, p=sample_dist)

            # CUSUM update
            llr = np.log(q_hat[idx, x] / p_dist[x])
            S[idx] = np.maximum(0.0, S[idx] + llr)

            # threshold crossings (vectorised over h_vec; consumes no randomness)
            newly = (~alarm[idx]) & (S[idx][:, None] >= h_vec)
            if newly.any():
                rows, cols = np.nonzero(newly)
                hit_t[idx[rows], cols] = t
                alarm[idx] |= newly

            active = ~alarm[:, -1]                  # the last threshold stops the run

            # exponential window update
            q_hat[idx] *= forget_val
            q_hat[idx, x] += (1.0 - forget_val)

        hit_sum += hit_t.sum(axis=0)

    return hit_sum / test_size


def cusum_fap_exp_batch(test_size, p_dist, h_vec, forget_val, *,
                        max_steps=1_000_000, seed=None, batch_size=None):
    """
    Average false-alarm period of the exponentially-weighted CUSUM (Fig. 15,
    maximum-sensitivity curves).  Port of probing_known cell 15.

    ``p_dist`` is the **augmented** pre-change pmf of length R + 1 whose last
    entry carries the ``1e-4`` mass of :func:`augment_with_null_outcome`; the
    Fig. 15 call site is
    ``cusum_fap_exp_batch(2000, p, h_vec, 0.99, max_steps=50_000)``.
    Note this uses log(q_hat/p) with q_hat initialised at ``p_dist``.
    """
    return _simulate_exp_window(test_size, p_dist, p_dist, h_vec, forget_val,
                                max_steps, seed=seed, batch_size=batch_size)


def cusum_add_exp_batch(test_size, p_dist, q_dist, h_vec, forget_val, *,
                        max_steps=1_000_000, seed=None, batch_size=None):
    """
    Average detection delay of the exponentially-weighted CUSUM when the pmf
    changes from ``p_dist`` to ``q_dist`` (Fig. 15, maximum-sensitivity curves).
    Port of probing_known cell 15.

    Both pmfs must be augmented to length R + 1 with
    :func:`augment_with_null_outcome`; the Fig. 15 call site is
    ``cusum_add_exp_batch(2000, p, q, h_vec, 0.99, max_steps=50_000)``.
    """
    return _simulate_exp_window(test_size, p_dist, q_dist, h_vec, forget_val,
                                max_steps, seed=seed, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Seeded, batched equivalents of the serial Fig. 11 routines
# ---------------------------------------------------------------------------

def cusum_fap_exp_single_batch(test_size, p_dist, h_vec, forget_val=0.99,
                               batch_size=2048, seed=None):
    """
    Seeded, batched equivalent of ``cusum_fap_exp_single``
    (Sensitivity_FAP_ADD_rank_def cell 4, re-exported above), used for the FAP
    axis of Fig. 11.

    No new simulator: this delegates to
    ``quantum_detection.cusum_fap_mismatch`` with ``p_true == p_nom``, which is
    the vectorised form of the same test (observations from ``p_dist``, q_hat
    initialised at ``p_dist``, exponential window applied after the crossing
    test).  Fig. 11 uses ``test_size = 2048`` and ``forget_val = 0.99``.
    """
    return cusum_fap_mismatch(test_size, p_dist, p_dist, h_vec,
                              forget_val=forget_val, batch_size=batch_size,
                              rng=_as_rng(seed))


def cusum_add_exp_single_batch(test_size, p_dist, q_dist, h_vec, forget_val=0.99,
                               batch_size=4096, seed=None):
    """
    Seeded, batched equivalent of ``cusum_add_exp_single(..., Qi_only=False)``
    (Sensitivity_FAP_ADD_rank_def cell 5, re-exported above): Fig. 11, panel a
    ("estimated q_hat_t").

    Delegates to ``quantum_detection.cusum_add_est_fast``.  ``q_dist`` may have
    R entries (full rank) or R + 1 with the null-space outcome last, in which
    case that outcome stops the run and credits the current time to every
    threshold.  Fig. 11 uses ``test_size = 204800`` and ``forget_val = 0.99``.
    """
    return cusum_add_est_fast(test_size, p_dist, q_dist, h_vec,
                              forget_val=forget_val, batch_size=batch_size,
                              rng=_as_rng(seed))


def cusum_add_exp_single_exact_batch(test_size, p_dist, q_dist, h_vec, seed=None,
                                     batch_size=None):
    """
    Seeded, batched equivalent of ``cusum_add_exp_single_exact(..., Qi_only=False)``
    (Sensitivity_FAP_ADD_rank_def cell 6, re-exported above): Fig. 11, panel b
    ("exact q").

    The cell-6 test is, symbol for symbol, ``cusum_add_batch_rank`` when
    ``len(q) == len(p) + 1`` and ``cusum_add_batch`` when the two lengths match,
    so this dispatches to those ports rather than duplicating them.  Fig. 11 uses
    ``test_size = 204800``.
    """
    p_dist = np.asarray(p_dist, dtype=float)
    q_dist = np.asarray(q_dist, dtype=float)
    if len(q_dist) == len(p_dist) + 1:
        return cusum_add_batch_rank(test_size, p_dist, q_dist, h_vec,
                                    seed=seed, batch_size=batch_size)
    if len(q_dist) == len(p_dist):
        return cusum_add_batch(test_size, p_dist, q_dist, h_vec,
                               seed=seed, batch_size=batch_size)
    raise ValueError("q_dist must have len(p_dist) or len(p_dist) + 1 entries")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(verbose=True):        # noqa: C901  (a linear script of checks)
    """
    Verify every ported routine against a direct transcription of its notebook
    cell.  Returns the number of failed checks (0 = all good).

    The four routines that the notebooks already wrote in batched form
    (``cusum_fap_batch``, ``cusum_add_batch``, ``cusum_add_batch_rank``,
    ``_simulate_exp_window``) are transcribed verbatim below and compared
    **bit-for-bit** under a shared ``RandomState``.  The routines that the
    notebooks wrote as per-trial Python loops cannot share an RNG stream with a
    batched port, so they are compared against the byte-identical serial
    transcriptions already re-exported from ``quantum_detection`` -- means over a
    large sample, to within Monte-Carlo error.
    """
    import os
    import sys

    failures = []

    @contextlib.contextmanager
    def quiet_legacy(seed):
        """legacy_seed() plus suppression of the reference routines' tqdm bars."""
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull), legacy_seed(seed):
                yield

    def check(name, ok, detail=""):
        status = "ok  " if ok else "FAIL"
        if verbose:
            print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    # ---- notebook transcriptions (verbatim, only np.random -> rs) ----------
    def nb_cusum_fap_batch(test_size, p_dist, q_dist, h_vec, rs, max_time=20_000):
        p_dist = p_dist / np.sum(p_dist)
        num_thresh = len(h_vec)
        false_alarm_sum = np.zeros(num_thresh)
        s = np.zeros(test_size)
        t = np.zeros(test_size, dtype=int)
        flag = np.zeros((test_size, num_thresh), dtype=bool)
        active = np.ones(test_size, dtype=bool)
        while np.any(active):
            t[active] += 1
            over_cap = np.where(active & (t >= max_time))[0]
            if over_cap.size:
                for j in range(num_thresh):
                    new_censored = ~flag[over_cap, j]
                    false_alarm_sum[j] += new_censored.sum() * max_time
                    flag[over_cap, j] = True
                active[over_cap] = False
                if not active.any():
                    break
            active_idx = np.where(active)[0]
            x = rs.choice(len(p_dist), size=active_idx.size, p=p_dist)
            llr = np.log(q_dist[x] / p_dist[x])
            s[active_idx] = np.maximum(0, s[active_idx] + llr)
            new_triggers = (~flag[active_idx]) & (s[active_idx][:, None] >= h_vec)
            false_alarm_sum += np.sum(t[active_idx][:, None] * new_triggers, axis=0)
            flag[active_idx] |= new_triggers
            active[active_idx] = ~flag[active_idx, -1]
        return false_alarm_sum / test_size

    def nb_cusum_add_batch(test_size, p_dist, q_dist, h_vec, rs):
        p_dist = p_dist / np.sum(p_dist)
        q_dist = q_dist / np.sum(q_dist)
        num_thresh = len(h_vec)
        detection_delay_sum = np.zeros(num_thresh)
        s = np.zeros(test_size)
        t = np.zeros(test_size, dtype=int)
        flag = np.zeros((test_size, num_thresh), dtype=bool)
        active = np.ones(test_size, dtype=bool)
        while np.any(active):
            active_idx = np.where(active)[0]
            t[active_idx] += 1
            x = rs.choice(len(q_dist), size=len(active_idx), p=q_dist)
            llr = np.log(q_dist[x] / p_dist[x])
            s[active_idx] = np.maximum(0, s[active_idx] + llr)
            new_triggers = (~flag[active_idx]) & (s[active_idx][:, None] >= h_vec)
            detection_delay_sum += np.sum((t[active_idx])[:, None] * new_triggers,
                                          axis=0)
            flag[active_idx] |= new_triggers
            active[active_idx] = ~flag[active_idx, -1]
        return detection_delay_sum / test_size

    def nb_cusum_add_batch_rank(test_size, p_dist, q_dist, h_vec, rs):
        p_dist = p_dist / p_dist.sum()
        q_dist = q_dist / q_dist.sum()
        if len(q_dist) != len(p_dist) + 1:
            raise ValueError("q_dist must be exactly one element longer than p_dist.")
        extra_symbol = len(p_dist)
        K = len(h_vec)
        delay_sum = np.zeros(K)
        s = np.zeros(test_size)
        t = np.zeros(test_size, dtype=int)
        triggered = np.zeros((test_size, K), dtype=bool)
        active = np.ones(test_size, dtype=bool)
        while active.any():
            idx = np.where(active)[0]
            t[idx] += 1
            x = rs.choice(len(q_dist), size=idx.size, p=q_dist)
            is_extra = (x == extra_symbol)
            if (~is_extra).any():
                idx_ord = idx[~is_extra]
                llr = np.log(q_dist[x[~is_extra]] / p_dist[x[~is_extra]])
                s[idx_ord] = np.maximum(0, s[idx_ord] + llr)
                new_hits = (~triggered[idx_ord]) & (s[idx_ord][:, None] >= h_vec)
                delay_sum += ((t[idx_ord])[:, None] * new_hits).sum(axis=0)
                triggered[idx_ord] |= new_hits
            if is_extra.any():
                idx_ext = idx[is_extra]
                not_yet = ~triggered[idx_ext]
                delay_sum += ((t[idx_ext])[:, None] * not_yet).sum(axis=0)
                triggered[idx_ext] = True
            active[idx] = ~triggered[idx, -1]
        return delay_sum / test_size

    def nb_simulate_exp_window(test_size, p_dist, sample_dist, h_vec, forget_val,
                               rs, max_steps=1_000_000):
        p_dist = np.asanyarray(p_dist, dtype=float).copy()
        sample_dist = np.asanyarray(sample_dist, dtype=float).copy()
        p_dist /= p_dist.sum()
        sample_dist /= sample_dist.sum()
        h_vec = np.asanyarray(h_vec, dtype=float)
        K = p_dist.size
        J = h_vec.size
        N = int(test_size)
        q_hat = np.tile(p_dist, (N, 1))
        S = np.zeros(N)
        alarm = np.zeros((N, J), dtype=bool)
        hit_t = np.zeros((N, J), dtype=int)
        t = 0
        active = np.ones(N, dtype=bool)
        while active.any() and t < max_steps:
            t += 1
            idx = np.where(active)[0]
            m = idx.size
            x = rs.choice(K, size=m, p=sample_dist)
            llr = np.log(q_hat[idx, x] / p_dist[x])
            S[idx] = np.maximum(0.0, S[idx] + llr)
            for j, h in enumerate(h_vec):
                newly = (~alarm[idx, j]) & (S[idx] >= h)
                if newly.any():
                    alarm[idx[newly], j] = True
                    hit_t[idx[newly], j] = t
            active = ~alarm[:, -1]
            q_hat[idx] *= forget_val
            q_hat[idx, x] += (1.0 - forget_val)
        return hit_t.mean(axis=0)

    # ---- a small rank-deficient-looking test case --------------------------
    p = np.array([0.45, 0.30, 0.25])
    q_short = np.array([0.20, 0.25, 0.30])          # sums to 0.75: q_0 = 0.25
    q_aug = append_null_outcome(q_short)            # [.20 .25 .30 .25]
    h = np.linspace(0.0, 1.6, 6)

    if verbose:
        print("cusum_nb self-test")
        print(f"  numpy {np.__version__}")
        print("  bit-exact checks against the notebook transcriptions "
              "(shared RandomState):")

    # 1) cusum_fap_batch, including the max_time censoring branch
    for max_time in (20_000, 12):
        got = cusum_fap_batch(400, p, q_short, h, max_time=max_time,
                              seed=np.random.RandomState(7))
        ref = nb_cusum_fap_batch(400, p.copy(), q_short.copy(), h,
                                 np.random.RandomState(7), max_time=max_time)
        check(f"cusum_fap_batch (max_time={max_time})",
              np.array_equal(got, ref), f"max|d| = {np.abs(got - ref).max():.3g}")

    # batching must not change the result of an un-capped run
    got_b = cusum_fap_batch(400, p, q_short, h, seed=np.random.RandomState(7),
                            batch_size=400)
    check("cusum_fap_batch batch_size=test_size identical",
          np.array_equal(got_b, nb_cusum_fap_batch(400, p.copy(), q_short.copy(), h,
                                                   np.random.RandomState(7))))

    # 2) cusum_add_batch (equal-length pmfs)
    q_full = q_short / q_short.sum()
    got = cusum_add_batch(500, p, q_full, h, seed=np.random.RandomState(3))
    ref = nb_cusum_add_batch(500, p.copy(), q_full.copy(), h, np.random.RandomState(3))
    check("cusum_add_batch", np.array_equal(got, ref),
          f"max|d| = {np.abs(got - ref).max():.3g}")

    # 3) cusum_add_batch_rank (extra null-space symbol)
    got = cusum_add_batch_rank(500, p, q_aug, h, seed=np.random.RandomState(5))
    ref = nb_cusum_add_batch_rank(500, p.copy(), q_aug.copy(), h,
                                  np.random.RandomState(5))
    check("cusum_add_batch_rank", np.array_equal(got, ref),
          f"max|d| = {np.abs(got - ref).max():.3g}")

    # 4) _simulate_exp_window and its two public wrappers (Fig. 15 convention)
    p_aug, q_aug15 = augment_with_null_outcome(p, q_short, p_null=1e-4)
    got = cusum_fap_exp_batch(300, p_aug, h, 0.99, max_steps=5_000,
                              seed=np.random.RandomState(11))
    ref = nb_simulate_exp_window(300, p_aug, p_aug, h, 0.99,
                                 np.random.RandomState(11), max_steps=5_000)
    check("cusum_fap_exp_batch (exp window, exp_val=0.99)",
          np.array_equal(got, ref), f"max|d| = {np.abs(got - ref).max():.3g}")

    got = cusum_add_exp_batch(300, p_aug, q_aug15, h, 0.99, max_steps=5_000,
                              seed=np.random.RandomState(13))
    ref = nb_simulate_exp_window(300, p_aug, q_aug15, h, 0.99,
                                 np.random.RandomState(13), max_steps=5_000)
    check("cusum_add_exp_batch (exp window, exp_val=0.99)",
          np.array_equal(got, ref), f"max|d| = {np.abs(got - ref).max():.3g}")

    # inputs must not be mutated (the notebook normalised in place)
    p_aug_before = p_aug.copy()
    cusum_fap_exp_batch(20, p_aug, h, 0.99, max_steps=200, seed=0)
    check("_simulate_exp_window leaves its inputs untouched",
          np.array_equal(p_aug, p_aug_before))

    # 5) reproducibility, and batch invariance up to Monte-Carlo error
    #    (chunking reorders the RNG draws, so equality holds only at equal
    #     batch_size -- the statistics are unaffected).  A second pmf pair with a
    #     small null-space mass (0.02) and a positive drift is used here so that
    #     *both* stopping causes are well populated and both curves are tested.
    q_mix = np.array([0.20, 0.28, 0.50])            # sums to 0.98: q_0 = 0.02
    q_mix_aug = append_null_outcome(q_mix)
    h_mix = np.linspace(0.0, 1.6, 5)
    if verbose:
        print("  reproducibility / batch invariance (both buckets populated):")
    for label, fn in (("cusum_add_cond_exact_batch",
                       lambda **kw: cusum_add_cond_exact_batch(
                           20_000, p, q_mix_aug, h_mix, **kw)),
                      ("cusum_add_cond_est_batch",
                       lambda **kw: cusum_add_cond_est_batch(
                           20_000, p, q_mix_aug, h_mix, 0.99, **kw))):
        r1 = fn(seed=42, batch_size=20_000)
        r2 = fn(seed=42, batch_size=20_000)
        same = all(np.array_equal(x, y) for x, y in zip(r1[:2], r2[:2])) \
            and r1[2:] == r2[2:]
        check(f"{label} is seed-reproducible", same)
        r3 = fn(seed=43, batch_size=997)
        ok = (np.allclose(r1[0], r3[0], rtol=0.05, atol=0.05)
              and np.allclose(r1[1], r3[1], rtol=0.05, atol=0.05)
              and abs(r1[2] - r3[2]) < 0.03 * 20_000)
        check(f"{label} is batch-invariant up to Monte-Carlo error", ok,
              f"n_xt/n_st {r1[2]:.0f}/{r1[3]:.0f} (1 batch) vs "
              f"{r3[2]:.0f}/{r3[3]:.0f} (997/batch)")

    # 6) the 1e-10 empty-bucket convention: with h so high that s_t never crosses,
    #    every run stops on the null outcome and the s_t > h curve must be exactly 0
    h_hi = np.array([50.0, 60.0])
    add_xt, add_st, c_xt, c_st = cusum_add_cond_exact_batch(500, p, q_aug, h_hi,
                                                            seed=1)
    check("empty s_t>h bucket yields an identically-zero curve (0 / 1e-10)",
          np.all(add_st == 0.0) and c_st == 1e-10 and np.isclose(c_xt, 500.0)
          and np.all(add_xt > 0))

    # 7) conditional ADD ports vs the serial notebook transcriptions
    #    (quantum_detection.cusum_add_exact / cusum_add_est are AST-identical to
    #     adap_grad cell 64 and Sensitivity cell 31; compare means)
    if verbose:
        print("  Monte-Carlo agreement with the serial notebook routines "
              "(40 000 trials):")
    n_mc = 40_000
    h_mc = h_mix
    with quiet_legacy(2024):
        ref_xt, ref_st, ref_cxt, ref_cst = cusum_add_exact(n_mc, p, q_mix_aug, h_mc)
    got_xt, got_st, got_cxt, got_cst = cusum_add_cond_exact_batch(
        n_mc, p, q_mix_aug, h_mc, seed=2024)
    ok = (np.allclose(got_xt, ref_xt, rtol=0.05, atol=0.05)
          and np.allclose(got_st, ref_st, rtol=0.05, atol=0.05)
          and abs(got_cxt - ref_cxt) < 0.03 * n_mc)
    check("cusum_add_cond_exact_batch == cusum_add_exact (cell 64 / cell 31)",
          ok, f"ADD_xt {np.array2string(got_xt, precision=3)} vs "
              f"{np.array2string(np.asarray(ref_xt), precision=3)}; "
              f"n_xt {got_cxt:.0f} vs {ref_cxt:.0f}")

    with quiet_legacy(2025):
        ref_xt, ref_st, ref_cxt, ref_cst = cusum_add_est(n_mc, p, q_mix_aug, h_mc,
                                                         0.99)
    got_xt, got_st, got_cxt, got_cst = cusum_add_cond_est_batch(
        n_mc, p, q_mix_aug, h_mc, 0.99, seed=2025)
    ok = (np.allclose(got_xt, ref_xt, rtol=0.05, atol=0.05)
          and np.allclose(got_st, ref_st, rtol=0.05, atol=0.05)
          and abs(got_cxt - ref_cxt) < 0.03 * n_mc)
    check("cusum_add_cond_est_batch == cusum_add_est (cell 31)",
          ok, f"ADD_xt {np.array2string(got_xt, precision=3)} vs "
              f"{np.array2string(np.asarray(ref_xt), precision=3)}; "
              f"n_xt {got_cxt:.0f} vs {ref_cxt:.0f}")

    # 8) the Fig. 11 batched equivalents vs their serial notebook originals
    if verbose:
        print("  Fig. 11 batched equivalents vs the serial cells 4/5/6 "
              "(8 000 trials; 2 500 for the slow FAP loop):")
    n_11 = 8_000
    h_11 = np.linspace(0.0, 1.2, 4)
    # the FAP reference is the slowest routine in the file (a scalar Python loop
    # whose runs last thousands of steps), so it gets a smaller sample and a
    # tolerance sized to the Monte-Carlo error at that sample
    n_fap = 2_500
    with quiet_legacy(31):
        ref = cusum_fap_exp_single(n_fap, p, h_11, forget_val=0.99)
    got = cusum_fap_exp_single_batch(n_fap, p, h_11, forget_val=0.99, seed=31)
    check("cusum_fap_exp_single_batch == cusum_fap_exp_single (cell 4)",
          np.allclose(got, ref, rtol=0.15, atol=1.0),
          f"max rel {np.max(np.abs(got - ref) / np.maximum(ref, 1e-9)):.3f}")

    with quiet_legacy(32):
        ref = cusum_add_exp_single(n_11, p, q_aug, h_11, forget_val=0.99)
    got = cusum_add_exp_single_batch(n_11, p, q_aug, h_11, forget_val=0.99, seed=32)
    check("cusum_add_exp_single_batch == cusum_add_exp_single (cell 5)",
          np.allclose(got, ref, rtol=0.05, atol=0.02),
          f"{np.array2string(got, precision=3)} vs "
          f"{np.array2string(np.asarray(ref), precision=3)}")

    with quiet_legacy(33):
        ref = cusum_add_exp_single_exact(n_11, p, q_aug, h_11)
    got = cusum_add_exp_single_exact_batch(n_11, p, q_aug, h_11, seed=33)
    check("cusum_add_exp_single_exact_batch == cusum_add_exp_single_exact (cell 6)",
          np.allclose(got, ref, rtol=0.05, atol=0.02),
          f"{np.array2string(got, precision=3)} vs "
          f"{np.array2string(np.asarray(ref), precision=3)}")

    # the same, for a full-rank q (no null-space outcome): dispatches to cusum_add_batch
    with quiet_legacy(34):
        ref = cusum_add_exp_single_exact(n_11, p, q_full, h_11)
    got = cusum_add_exp_single_exact_batch(n_11, p, q_full, h_11, seed=34)
    check("cusum_add_exp_single_exact_batch, len(q) == len(p)",
          np.allclose(got, ref, rtol=0.05, atol=0.02),
          f"{np.array2string(got, precision=3)} vs "
          f"{np.array2string(np.asarray(ref), precision=3)}")

    # 9) the re-exports really are the quantum_detection objects, and cusum_add
    #    is the cell-64 alias of cusum_add_exact
    check("cusum_add is the cell-64 name of cusum_add_exact",
          cusum_add is cusum_add_exact)

    # 10) seeded legacy wrapper is reproducible and restores the global state
    np.random.seed(999)
    before = np.random.get_state()[1][0]
    r1 = cusum_fap_exp_fast_seeded(200, p, h_11, forget_val=0.99, batch_size=100,
                                   seed=8)
    r2 = cusum_fap_exp_fast_seeded(200, p, h_11, forget_val=0.99, batch_size=100,
                                   seed=8)
    check("cusum_fap_exp_fast_seeded is reproducible and restores np.random",
          np.array_equal(r1, r2) and np.random.get_state()[1][0] == before)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        s1 = run_seeded(cusum_fap_exp_single, 20, p, h_11, forget_val=0.99, seed=4)
        s2 = run_seeded(cusum_fap_exp_single, 20, p, h_11, forget_val=0.99, seed=4)
        s3 = run_seeded(cusum_fap_exp_single, 20, p, h_11, forget_val=0.99, seed=5)
    check("run_seeded makes a legacy re-export reproducible",
          np.array_equal(s1, s2) and not np.array_equal(s1, s3))

    if verbose:
        if failures:
            print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        else:
            print("\nall checks passed")
        sys.stdout.flush()
    return len(failures)


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
