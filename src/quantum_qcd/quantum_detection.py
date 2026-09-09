"""
CUSUM (Cumulative Sum) detection functions for quantum anomaly detection.

Contains implementations of False Alarm Period (FAP) and Average Detection Delay (ADD)
algorithms, with both single-run and batched variants.
"""

import numpy as np
import torch
from tqdm import tqdm


def cusum_fap_exp(test_size, p_dist, h_vec, forget_val):
    """
    Compute False Alarm Period (FAP) for CUSUM with exponential forgetting.

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution
    h_vec : array-like
        Vector of detection thresholds
    forget_val : float
        Forgetting factor for distribution estimation

    Returns:
    --------
    cusum_fap_vec : ndarray
        Average false alarm period for each threshold
    """
    p_dist = p_dist/np.sum(p_dist)
    cusum_false_alarm_periods = np.zeros(len(h_vec))

    for jj in range(test_size):
        # initialization for a new experiment
        t = 0
        s_t = 0
        q_hat = p_dist
        cusum_flag_vec = np.zeros(len(h_vec))

        # new experiment
        while cusum_flag_vec[-1] == 0:
            t += 1
            # observation at time t (pre-change)
            x_t = np.random.choice(len(p_dist), p=p_dist)
            # log-likelihood ratio
            llr_value = np.log(q_hat[x_t]/p_dist[x_t])
            # state update
            s_t = np.max([0, s_t + llr_value])
            print(f'p: {p_dist[x_t]:.3f}, q_hat: {q_hat[x_t]:.3f}, llr: {llr_value:.3f}, s_t: {s_t:.3f}')
            # update the alarm flags and false alarm periods
            cusum_false_alarm_periods += t*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)

            q_hat = forget_val*q_hat
            q_hat[x_t] += (1-forget_val)

    # compute the average false alarm period (fap)
    cusum_fap_vec = cusum_false_alarm_periods/test_size
    return cusum_fap_vec


def cusum_add_exp(test_size, p_dist, q_dist, h_vec, forget_val):
    """
    Compute Average Detection Delay (ADD) for CUSUM with exponential forgetting.

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution
    q_dist : array-like
        Post-change probability distribution
    h_vec : array-like
        Vector of detection thresholds
    forget_val : float
        Forgetting factor for distribution estimation

    Returns:
    --------
    cusum_add_vec : ndarray
        Average detection delay for each threshold
    """
    p_dist = p_dist/np.sum(p_dist)
    q_dist = q_dist/np.sum(q_dist)
    cusum_detection_delays = np.zeros(len(h_vec))

    for jj in range(test_size):
        # initialization for a new experiment
        t = 0
        s_t = 0
        q_hat = p_dist
        cusum_flag_vec = np.zeros(len(h_vec))

        # new experiment
        while cusum_flag_vec[-1] == 0:
            t += 1
            # observation at time t (post-change)
            x_t = np.random.choice(len(q_dist), p=q_dist)
            q_hat = forget_val*q_hat
            q_hat[x_t] += (1-forget_val)

            # log-likelihood ratio
            llr_value = np.log(q_hat[x_t]/p_dist[x_t])
            # state update
            s_t = np.max([0, s_t + llr_value])
            # update the alarm flags and detection delays
            cusum_detection_delays += (t)*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)

    # compute the average detection delay
    cusum_add_vec = cusum_detection_delays/test_size
    return cusum_add_vec


def cusum_fap_exp_single(test_size, p_dist, h_vec, forget_val=0.99):
    """
    Compute False Alarm Period with progress bar (single-run variant).

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution
    h_vec : array-like
        Vector of detection thresholds
    forget_val : float, optional
        Forgetting factor (default: 0.99)

    Returns:
    --------
    cusum_fap_vec : ndarray
        Average false alarm period for each threshold
    """
    p_dist = p_dist/np.sum(p_dist)
    R = len(p_dist)
    cusum_false_alarm_periods = np.zeros(len(h_vec))

    for jj in tqdm(range(test_size)):
        # initialization for a new experiment
        t = 0
        s_t = 0
        cusum_flag_vec = np.zeros(len(h_vec))
        q_hat = p_dist

        # new experiment
        while cusum_flag_vec[-1] == 0:
            t += 1
            x_t = np.random.choice(len(p_dist), p=p_dist)

            # log-likelihood ratio
            llr_value = np.log(q_hat[x_t]/p_dist[x_t])
            s_t = np.maximum(0, s_t + llr_value)

            # update the alarm flags and false alarm periods
            cusum_false_alarm_periods += t*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)

            q_hat = forget_val*q_hat
            q_hat[x_t] += (1-forget_val)

    # compute the average false alarm period (fap)
    cusum_fap_vec = cusum_false_alarm_periods/test_size
    return cusum_fap_vec


def cusum_add_exp_single(test_size, p_dist, q_dist, h_vec, forget_val=0.9, Qi_only=False):
    """
    Compute Average Detection Delay with optional exclusive outcome handling.

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution (size R)
    q_dist : array-like
        Post-change probability distribution (size M = R or R+1)
    h_vec : array-like
        Vector of detection thresholds
    forget_val : float, optional
        Forgetting factor (default: 0.9)
    Qi_only : bool, optional
        If True and R < M, skip exclusive outcome until relevant (default: False)

    Returns:
    --------
    cusum_add_vec : ndarray
        Average detection delay for each threshold
    """
    p_dist = p_dist/np.sum(p_dist)
    q_dist = q_dist/np.sum(q_dist)
    R = len(p_dist)
    M = len(q_dist)
    cusum_detection_delays = np.zeros(len(h_vec))

    for jj in tqdm(range(test_size)):
        # initialization for a new experiment
        t = 0
        s_t = 0
        cusum_flag_vec = np.zeros(len(h_vec))
        if R < M:
            q_hat = np.append(p_dist, 0)
        else:
            q_hat = p_dist

        while cusum_flag_vec[-1] == 0:
            t += 1
            # observation at time t (post-change)
            x_t = np.random.choice(len(q_dist), p=q_dist)

            if R < M:
                if Qi_only == False:
                    if x_t == R:
                        cusum_detection_delays += (t)*(cusum_flag_vec == 0)
                        break
                else:  # Qi_only == True
                    while x_t == R:
                        q_hat = forget_val*q_hat
                        q_hat[x_t] += (1-forget_val)
                        t += 1
                        x_t = np.random.choice(len(q_dist), p=q_dist)

            # log-likelihood ratio
            llr_value = np.log(q_hat[x_t]/p_dist[x_t])
            # state update
            s_t = np.max([0, s_t + llr_value])

            # update the alarm flags and detection delays
            cusum_detection_delays += (t)*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)

            q_hat = forget_val*q_hat
            q_hat[x_t] += (1-forget_val)

    # compute the average detection delay
    cusum_add_vec = cusum_detection_delays/test_size
    return cusum_add_vec


def cusum_add_exp_single_exact(test_size, p_dist, q_dist, h_vec, Qi_only=False):
    """
    Compute Average Detection Delay with exact (non-estimated) post-change distribution.

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution (size R)
    q_dist : array-like
        Post-change probability distribution (size M = R or R+1)
    h_vec : array-like
        Vector of detection thresholds
    Qi_only : bool, optional
        If True and R < M, skip exclusive outcome until relevant (default: False)

    Returns:
    --------
    cusum_add_vec : ndarray
        Average detection delay for each threshold
    """
    p_dist = p_dist/np.sum(p_dist)
    q_dist = q_dist/np.sum(q_dist)
    R = len(p_dist)
    M = len(q_dist)
    cusum_detection_delays = np.zeros(len(h_vec))

    for jj in tqdm(range(test_size)):
        # initialization for a new experiment
        t = 0
        s_t = 0
        cusum_flag_vec = np.zeros(len(h_vec))

        while cusum_flag_vec[-1] == 0:
            t += 1
            # observation at time t (post-change)
            x_t = np.random.choice(len(q_dist), p=q_dist)

            if R < M:
                if Qi_only == False:
                    if x_t == R:
                        cusum_detection_delays += (t)*(cusum_flag_vec == 0)
                        break
                else:  # Qi_only == True
                    while x_t == R:
                        t += 1
                        x_t = np.random.choice(len(q_dist), p=q_dist)

            # log-likelihood ratio (using exact q_dist)
            llr_value = np.log(q_dist[x_t]/p_dist[x_t])
            # state update
            s_t = np.max([0, s_t + llr_value])

            # update the alarm flags and detection delays
            cusum_detection_delays += (t)*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)

    # compute the average detection delay
    cusum_add_vec = cusum_detection_delays/test_size
    return cusum_add_vec


def cusum_fap_exp_fast(test_size, p_dist, h_vec, forget_val=0.99, batch_size=5000):
    """
    Compute False Alarm Period using vectorized batch processing for speed.

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution
    h_vec : array-like
        Vector of detection thresholds
    forget_val : float, optional
        Forgetting factor (default: 0.99)
    batch_size : int, optional
        Number of parallel experiments per batch (default: 5000)

    Returns:
    --------
    cusum_fap_vec : ndarray
        Average false alarm period for each threshold
    """
    p_dist = p_dist / np.sum(p_dist)
    R = len(p_dist)
    n_h = len(h_vec)

    cusum_false_alarm_periods = np.zeros(n_h)
    n_batches = int(np.ceil(test_size / batch_size))

    for _ in tqdm(range(n_batches)):
        this_batch_size = min(batch_size, test_size)
        test_size -= this_batch_size

        # initialize
        t = np.zeros(this_batch_size, dtype=int)
        s_t = np.zeros(this_batch_size)
        q_hat = np.tile(p_dist, (this_batch_size, 1))
        cusum_flag_mat = np.zeros((this_batch_size, n_h), dtype=bool)

        active = np.arange(this_batch_size)

        while active.size > 0:
            t[active] += 1
            x_t = np.random.choice(R, size=active.size, p=p_dist)

            # log-likelihood ratio
            llr = np.log(q_hat[active, x_t] / p_dist[x_t])
            s_t[active] = np.maximum(0, s_t[active] + llr)

            # threshold checks (vectorized across thresholds)
            triggered = (s_t[active, None] >= h_vec[None, :]) & (~cusum_flag_mat[active])
            if np.any(triggered):
                cusum_false_alarm_periods += (t[active, None] * triggered).sum(axis=0)
                cusum_flag_mat[active] |= triggered

            # update q_hat
            q_hat[active] *= forget_val
            q_hat[active, x_t] += (1 - forget_val)

            # drop finished experiments (all thresholds satisfied)
            still_running = ~cusum_flag_mat[active, -1]
            active = active[still_running]

    cusum_fap_vec = cusum_false_alarm_periods / (n_batches * batch_size)
    return cusum_fap_vec


def cusum_add_exact(test_size, p_dist, q_dist, h_vec):
    """
    Compute Average Detection Delay with exact post-change distribution.

    Tracks two types of detection events:
    - xt: detection by observing exclusive outcome (if exists)
    - st: detection by crossing threshold

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution
    q_dist : array-like
        Post-change probability distribution
    h_vec : array-like
        Vector of detection thresholds

    Returns:
    --------
    cusum_add_vec_xt : ndarray
        Average detection delay by exclusive outcome
    cusum_add_vec_st : ndarray
        Average detection delay by threshold crossing
    counter_xt : int
        Count of detections by exclusive outcome
    counter_st : int
        Count of detections by threshold crossing
    """
    p_dist = p_dist/np.sum(p_dist)
    q_dist = q_dist/np.sum(q_dist)
    cusum_detection_delays_xt = np.zeros(len(h_vec))
    cusum_detection_delays_st = np.zeros(len(h_vec))
    counter_xt = 1e-10
    counter_st = 1e-10

    for jj in tqdm(range(test_size)):
        # initialization for a new experiment
        t = 0
        s_t = 0
        cusum_flag_vec = np.zeros(len(h_vec))
        cusum_detection_delays = np.zeros(len(h_vec))

        # new experiment
        while cusum_flag_vec[-1] == 0:
            t += 1
            # observation at time t (post-change)
            x_t = np.random.choice(len(q_dist), p=q_dist)
            if x_t == len(q_dist)-1:
                cusum_detection_delays += (t)*(cusum_flag_vec == 0)
                cusum_detection_delays_xt += cusum_detection_delays
                counter_xt += 1
                break
            # log-likelihood ratio
            llr_value = np.log(q_dist[x_t]/p_dist[x_t])
            # state update
            s_t = np.max([0, s_t + llr_value])
            cusum_detection_delays += (t)*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)
            if cusum_flag_vec[-1] == 1:
                cusum_detection_delays_st += cusum_detection_delays
                counter_st += 1
                break

    # compute the average detection delays
    cusum_add_vec_xt = cusum_detection_delays_xt/counter_xt
    cusum_add_vec_st = cusum_detection_delays_st/counter_st
    return cusum_add_vec_xt, cusum_add_vec_st, counter_xt, counter_st


def cusum_add_est(test_size, p_dist, q_dist, h_vec, forget_val):
    """
    Compute Average Detection Delay with estimated post-change distribution.

    Tracks two types of detection events:
    - xt: detection by observing exclusive outcome (if exists)
    - st: detection by crossing threshold

    Parameters:
    -----------
    test_size : int
        Number of experiments to run
    p_dist : array-like
        Pre-change probability distribution
    q_dist : array-like
        Post-change probability distribution
    h_vec : array-like
        Vector of detection thresholds
    forget_val : float
        Forgetting factor for distribution estimation

    Returns:
    --------
    cusum_add_vec_xt : ndarray
        Average detection delay by exclusive outcome
    cusum_add_vec_st : ndarray
        Average detection delay by threshold crossing
    counter_xt : int
        Count of detections by exclusive outcome
    counter_st : int
        Count of detections by threshold crossing
    """
    p_dist = p_dist/np.sum(p_dist)
    q_dist = q_dist/np.sum(q_dist)
    cusum_detection_delays_xt = np.zeros(len(h_vec))
    cusum_detection_delays_st = np.zeros(len(h_vec))
    counter_xt = 1e-10
    counter_st = 1e-10

    for jj in tqdm(range(test_size)):
        # initialization for a new experiment
        t = 0
        s_t = 0
        cusum_flag_vec = np.zeros(len(h_vec))
        cusum_detection_delays = np.zeros(len(h_vec))
        q_hat = np.append(p_dist, 0)

        # new experiment
        while cusum_flag_vec[-1] == 0:
            t += 1
            # observation at time t (post-change)
            x_t = np.random.choice(len(q_hat), p=q_dist)
            if x_t == len(q_dist)-1:
                cusum_detection_delays += (t)*(cusum_flag_vec == 0)
                cusum_detection_delays_xt += cusum_detection_delays
                counter_xt += 1
                break
            # log-likelihood ratio
            llr_value = np.log(q_hat[x_t]/p_dist[x_t])
            # state update
            s_t = np.max([0, s_t + llr_value])
            cusum_detection_delays += (t)*(cusum_flag_vec == 0)*(s_t >= h_vec)
            cusum_flag_vec += (cusum_flag_vec == 0)*(s_t >= h_vec)
            if cusum_flag_vec[-1] == 1:
                cusum_detection_delays_st += cusum_detection_delays
                counter_st += 1
                break
            q_hat = forget_val*q_hat
            q_hat[x_t] += (1-forget_val)

    # compute the average detection delays
    cusum_add_vec_xt = cusum_detection_delays_xt/counter_xt
    cusum_add_vec_st = cusum_detection_delays_st/counter_st
    return cusum_add_vec_xt, cusum_add_vec_st, counter_xt, counter_st


# ---------------------------------------------------------------------------
# Vectorized CPU implementations with a possibly mismatched pre-change model
# (revision Concern 2: robustness to a noisy pre-change state)
# ---------------------------------------------------------------------------

def _exp_window_stopping_time(test_size, sample_pmf, p_nom, h_vec, forget_val,
                              batch_size, rng):
    """
    Mean stopping time of the exponential-window CUSUM-like test, vectorized
    over experiments and thresholds.

    Outcomes are drawn i.i.d. from ``sample_pmf`` (length R or R + 1); the test
    uses the nominal pre-change pmf ``p_nom`` (length R) in the log-likelihood
    ratio and initialises q_hat at p_nom, exactly as cusum_fap_exp_single and
    cusum_add_exp_single do. If ``sample_pmf`` has R + 1 entries, the LAST
    outcome is the null-space outcome, which is impossible under p_nom and
    stops the test immediately for every threshold. q_hat is updated after the
    LLR, as in the *_single routines that produced Figs. 11-12.
    """
    p_nom = np.asarray(p_nom, dtype=float)
    p_nom = p_nom / p_nom.sum()
    sample_pmf = np.asarray(sample_pmf, dtype=float)
    sample_pmf = sample_pmf / sample_pmf.sum()
    h_vec = np.asarray(h_vec, dtype=float)
    R, M, n_h = len(p_nom), len(sample_pmf), len(h_vec)
    if M not in (R, R + 1):
        raise ValueError(f"sample pmf must have {R} or {R + 1} entries, got {M}")
    has_null = (M == R + 1)
    log_p_nom = np.log(p_nom)

    total = np.zeros(n_h)
    remaining = int(test_size)
    while remaining > 0:
        B = min(batch_size, remaining)
        remaining -= B

        t = np.zeros(B, dtype=np.int64)
        s = np.zeros(B)
        q_hat = np.tile(p_nom, (B, 1))
        flag = np.zeros((B, n_h), dtype=bool)
        active = np.arange(B)

        while active.size:
            t[active] += 1
            x = rng.choice(M, size=active.size, p=sample_pmf)

            if has_null:
                is_null = (x == R)
                if is_null.any():
                    idx = active[is_null]
                    total += (t[idx][:, None] * ~flag[idx]).sum(axis=0)
                    flag[idx] = True
                active_ne = active[~is_null]
                x_ne = x[~is_null]
            else:
                active_ne, x_ne = active, x

            if active_ne.size:
                llr = np.log(q_hat[active_ne, x_ne]) - log_p_nom[x_ne]
                s[active_ne] = np.maximum(0.0, s[active_ne] + llr)

                trig = (s[active_ne][:, None] >= h_vec[None, :]) & ~flag[active_ne]
                if trig.any():
                    total += (t[active_ne][:, None] * trig).sum(axis=0)
                    flag[active_ne] |= trig

                q_hat[active_ne] *= forget_val
                q_hat[active_ne, x_ne] += 1.0 - forget_val

            active = active[~flag[active, -1]]

    return total / test_size


def cusum_fap_mismatch(test_size, p_true, p_nom, h_vec, forget_val=0.99,
                       batch_size=2048, rng=None):
    """
    False alarm period when the pre-change outcomes follow ``p_true`` while the
    test is designed for the nominal pmf ``p_nom``.

    Parameters:
    -----------
    test_size : int
        Number of experiments
    p_true : array-like
        Actual pre-change pmf, length R, or R + 1 with the null-space outcome
        LAST (it triggers an immediate false alarm)
    p_nom : array-like
        Nominal pre-change pmf (length R) used in the LLR and to initialise q_hat
    h_vec : array-like
        Increasing thresholds
    forget_val : float, optional
        Forgetting factor of the exponential window (default 0.99)
    batch_size : int, optional
        Experiments simulated in parallel (default 2048)
    rng : numpy.random.Generator, optional

    Returns:
    --------
    cusum_fap_vec : ndarray
        Average false alarm period for each threshold
    """
    rng = np.random.default_rng() if rng is None else rng
    return _exp_window_stopping_time(test_size, p_true, p_nom, h_vec, forget_val,
                                     batch_size, rng)


def cusum_add_est_fast(test_size, p_nom, q_dist, h_vec, forget_val=0.99,
                       batch_size=4096, rng=None):
    """
    Vectorized equivalent of cusum_add_exp_single (Qi_only=False): average
    detection delay with the exponential-window estimate q_hat, post-change
    outcomes drawn from ``q_dist`` (length R, or R + 1 with the null-space
    outcome LAST, which stops the test immediately).

    Returns:
    --------
    cusum_add_vec : ndarray
        Average detection delay for each threshold
    """
    rng = np.random.default_rng() if rng is None else rng
    return _exp_window_stopping_time(test_size, q_dist, p_nom, h_vec, forget_val,
                                     batch_size, rng)


# ---------------------------------------------------------------------------
# GPU-accelerated batched implementations
# ---------------------------------------------------------------------------

def _get_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def cusum_fap_gpu(test_size, p_dist, h_vec, forget_val=0.99, batch_size=2048):
    """
    GPU-accelerated False Alarm Period using batched parallel experiments.

    Parameters
    ----------
    test_size : int
        Total number of experiments.
    p_dist : array-like
        Pre-change probability distribution.
    h_vec : array-like
        Vector of detection thresholds.
    forget_val : float
        Forgetting factor for distribution estimation.
    batch_size : int
        Number of experiments to run in parallel per batch.

    Returns
    -------
    cusum_fap_vec : ndarray
        Average false alarm period for each threshold.
    """
    device = _get_device()
    p_dist = np.asarray(p_dist, dtype=np.float32)
    p_dist = p_dist / p_dist.sum()
    R = len(p_dist)
    n_h = len(h_vec)

    p_t = torch.tensor(p_dist, dtype=torch.float32, device=device)
    h_t = torch.tensor(h_vec, dtype=torch.float32, device=device)
    log_p = torch.log(p_t)

    total_fap = torch.zeros(n_h, dtype=torch.float32, device=device)
    remaining = test_size

    pbar = tqdm(total=test_size, desc="FAP (GPU)")
    while remaining > 0:
        B = min(batch_size, remaining)
        remaining -= B

        t = torch.zeros(B, dtype=torch.long, device=device)
        s_t = torch.zeros(B, dtype=torch.float32, device=device)
        q_hat = p_t.unsqueeze(0).expand(B, -1).clone()
        flag = torch.zeros(B, n_h, dtype=torch.bool, device=device)
        fap_accum = torch.zeros(B, n_h, dtype=torch.float32, device=device)

        active = torch.arange(B, device=device)

        while active.numel() > 0:
            n_active = active.numel()
            t[active] += 1

            # sample from p_dist
            x_t = torch.multinomial(p_t.expand(n_active, -1), 1).squeeze(1)

            # log-likelihood ratio: log(q_hat[x_t] / p[x_t])
            llr = torch.log(q_hat[active].gather(1, x_t.unsqueeze(1)).squeeze(1)) - log_p[x_t]
            s_t[active] = torch.clamp(s_t[active] + llr, min=0.0)

            # threshold check
            triggered = (s_t[active].unsqueeze(1) >= h_t.unsqueeze(0)) & (~flag[active])
            newly = triggered.any(dim=1)
            if newly.any():
                fap_accum[active] += t[active].unsqueeze(1).float() * triggered.float()
                flag[active] |= triggered

            # update q_hat
            q_hat[active] *= forget_val
            q_hat[active].scatter_add_(1, x_t.unsqueeze(1),
                                       torch.full((n_active, 1), 1 - forget_val,
                                                  dtype=torch.float32, device=device))

            # drop finished experiments
            still_running = ~flag[active, -1]
            active = active[still_running]

        total_fap += fap_accum.sum(dim=0)
        pbar.update(B)

    pbar.close()
    cusum_fap_vec = (total_fap / test_size).cpu().numpy()
    return cusum_fap_vec


def cusum_add_est_gpu(test_size, p_dist, q_dist, h_vec, forget_val=0.9, batch_size=2048):
    """
    GPU-accelerated ADD with estimated post-change distribution.

    Parameters
    ----------
    test_size : int
        Total number of experiments.
    p_dist : array-like
        Pre-change probability distribution (size R).
    q_dist : array-like
        Post-change probability distribution (size M, may be R+1).
    h_vec : array-like
        Vector of detection thresholds.
    forget_val : float
        Forgetting factor for distribution estimation.
    batch_size : int
        Number of experiments to run in parallel per batch.

    Returns
    -------
    cusum_add_vec : ndarray
        Average detection delay for each threshold.
    """
    device = _get_device()
    p_dist = np.asarray(p_dist, dtype=np.float32)
    q_dist = np.asarray(q_dist, dtype=np.float32)
    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()
    R = len(p_dist)
    M = len(q_dist)
    n_h = len(h_vec)

    p_t = torch.tensor(p_dist, dtype=torch.float32, device=device)
    q_t = torch.tensor(q_dist, dtype=torch.float32, device=device)
    h_t = torch.tensor(h_vec, dtype=torch.float32, device=device)
    log_p = torch.log(p_t)

    total_add = torch.zeros(n_h, dtype=torch.float32, device=device)
    remaining = test_size

    pbar = tqdm(total=test_size, desc="ADD est (GPU)")
    while remaining > 0:
        B = min(batch_size, remaining)
        remaining -= B

        t = torch.zeros(B, dtype=torch.long, device=device)
        s_t = torch.zeros(B, dtype=torch.float32, device=device)
        flag = torch.zeros(B, n_h, dtype=torch.bool, device=device)
        add_accum = torch.zeros(B, n_h, dtype=torch.float32, device=device)

        # q_hat initialisation: append 0 if R < M
        if R < M:
            q_hat = torch.cat([p_t.unsqueeze(0).expand(B, -1),
                               torch.zeros(B, M - R, dtype=torch.float32, device=device)], dim=1)
        else:
            q_hat = p_t.unsqueeze(0).expand(B, -1).clone()

        active = torch.arange(B, device=device)

        while active.numel() > 0:
            n_active = active.numel()
            t[active] += 1

            # sample from q_dist (post-change)
            x_t = torch.multinomial(q_t.expand(n_active, -1), 1).squeeze(1)

            # handle exclusive outcome (x_t == R when R < M)
            if R < M:
                exclusive = (x_t == R)
                if exclusive.any():
                    exc_idx = active[exclusive]
                    not_flagged = ~flag[exc_idx]
                    add_accum[exc_idx] += t[exc_idx].unsqueeze(1).float() * not_flagged.float()
                    flag[exc_idx] = True
                    # update q_hat for exclusive outcomes
                    q_hat[exc_idx] *= forget_val
                    q_hat[exc_idx, R] += (1 - forget_val)

                non_exc = ~exclusive
                if not non_exc.any():
                    still_running = ~flag[active, -1]
                    active = active[still_running]
                    continue
                active_ne = active[non_exc]
                x_t_ne = x_t[non_exc]
            else:
                active_ne = active
                x_t_ne = x_t

            n_ne = active_ne.numel()

            # log-likelihood ratio: log(q_hat[x_t] / p[x_t])
            llr = torch.log(q_hat[active_ne].gather(1, x_t_ne.unsqueeze(1)).squeeze(1)) - log_p[x_t_ne]
            s_t[active_ne] = torch.clamp(s_t[active_ne] + llr, min=0.0)

            # threshold check
            triggered = (s_t[active_ne].unsqueeze(1) >= h_t.unsqueeze(0)) & (~flag[active_ne])
            if triggered.any():
                add_accum[active_ne] += t[active_ne].unsqueeze(1).float() * triggered.float()
                flag[active_ne] |= triggered

            # update q_hat for non-exclusive outcomes
            q_hat[active_ne] *= forget_val
            q_hat[active_ne].scatter_add_(1, x_t_ne.unsqueeze(1),
                                          torch.full((n_ne, 1), 1 - forget_val,
                                                     dtype=torch.float32, device=device))

            # drop finished experiments
            still_running = ~flag[active, -1]
            active = active[still_running]

        total_add += add_accum.sum(dim=0)
        pbar.update(B)

    pbar.close()
    cusum_add_vec = (total_add / test_size).cpu().numpy()
    return cusum_add_vec


def cusum_add_exact_gpu(test_size, p_dist, q_dist, h_vec, batch_size=2048):
    """
    GPU-accelerated ADD with exact post-change distribution.

    Parameters
    ----------
    test_size : int
        Total number of experiments.
    p_dist : array-like
        Pre-change probability distribution (size R).
    q_dist : array-like
        Post-change probability distribution (size M, may be R+1).
    h_vec : array-like
        Vector of detection thresholds.
    batch_size : int
        Number of experiments to run in parallel per batch.

    Returns
    -------
    cusum_add_vec : ndarray
        Average detection delay for each threshold.
    """
    device = _get_device()
    p_dist = np.asarray(p_dist, dtype=np.float32)
    q_dist = np.asarray(q_dist, dtype=np.float32)
    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()
    R = len(p_dist)
    M = len(q_dist)
    n_h = len(h_vec)

    p_t = torch.tensor(p_dist, dtype=torch.float32, device=device)
    q_t = torch.tensor(q_dist, dtype=torch.float32, device=device)
    h_t = torch.tensor(h_vec, dtype=torch.float32, device=device)
    # precompute log(q/p) for each outcome in the shared support
    log_qp = torch.log(q_t[:R] / p_t)

    total_add = torch.zeros(n_h, dtype=torch.float32, device=device)
    remaining = test_size

    pbar = tqdm(total=test_size, desc="ADD exact (GPU)")
    while remaining > 0:
        B = min(batch_size, remaining)
        remaining -= B

        t = torch.zeros(B, dtype=torch.long, device=device)
        s_t = torch.zeros(B, dtype=torch.float32, device=device)
        flag = torch.zeros(B, n_h, dtype=torch.bool, device=device)
        add_accum = torch.zeros(B, n_h, dtype=torch.float32, device=device)

        active = torch.arange(B, device=device)

        while active.numel() > 0:
            n_active = active.numel()
            t[active] += 1

            # sample from q_dist (post-change)
            x_t = torch.multinomial(q_t.expand(n_active, -1), 1).squeeze(1)

            # handle exclusive outcome
            if R < M:
                exclusive = (x_t == R)
                if exclusive.any():
                    exc_idx = active[exclusive]
                    not_flagged = ~flag[exc_idx]
                    add_accum[exc_idx] += t[exc_idx].unsqueeze(1).float() * not_flagged.float()
                    flag[exc_idx] = True

                non_exc = ~exclusive
                if not non_exc.any():
                    still_running = ~flag[active, -1]
                    active = active[still_running]
                    continue
                active_ne = active[non_exc]
                x_t_ne = x_t[non_exc]
            else:
                active_ne = active
                x_t_ne = x_t

            # log-likelihood ratio: precomputed log(q[x]/p[x])
            llr = log_qp[x_t_ne]
            s_t[active_ne] = torch.clamp(s_t[active_ne] + llr, min=0.0)

            # threshold check
            triggered = (s_t[active_ne].unsqueeze(1) >= h_t.unsqueeze(0)) & (~flag[active_ne])
            if triggered.any():
                add_accum[active_ne] += t[active_ne].unsqueeze(1).float() * triggered.float()
                flag[active_ne] |= triggered

            # drop finished experiments
            still_running = ~flag[active, -1]
            active = active[still_running]

        total_add += add_accum.sum(dim=0)
        pbar.update(B)

    pbar.close()
    cusum_add_vec = (total_add / test_size).cpu().numpy()
    return cusum_add_vec
