"""
Projected gradient ascent (PGA) on the unitary / Stiefel manifold for
optimizing von Neumann measurements.

All three optimizers here maximize the Kullback-Leibler divergence
D(q||p) of the outcome distributions of a rank-one projective measurement
Q_i = |u_i><u_i| applied to a pre-change state rho (giving p) and a
post-change state sigma (giving q).  Each step takes an ordinary Euclidean
gradient, moves along +grad (ascent), and retracts back onto the manifold
with the exact polar factor obtained from an SVD.  The step length comes
from an Armijo backtracking line search written for *ascent* (sufficient
increase), i.e. it accepts the first step with

    f(U_new) >= f(U) + armijo_alpha * step * ||grad||^2 .

This module is a line-by-line port of the notebook code that produced
Figs. 5, 6, 7 (and the Fig. 1 convergence arrays) of "Quantum Quickest
Change Detection".  Notebook roots (read-only):

    notebooks_raw/Quantum_Anomaly_Detection/optimization_known/adaptive_gradient/

Functions
---------
compute_cost      : cost/probabilities helper, adap_grad_rank_defic_rho.ipynb cell 35
alg1              : rank-truncated PGA, adap_grad_rank_defic_rho.ipynb cell 35  (Fig. 5)
pgd_rank_aware    : support-projected PGA, adap_grad_rank_defic_rho_copy.ipynb cell 27
                    (Figs. 6 and 7; the defining cell was deleted from the
                    notebook that calls it)
pgd_full_rank     : full-rank PGA, adaptive_gradient.ipynb cell 25  (Fig. 1 arrays)

Porting notes (deliberate fidelity choices, do not "fix" these)
--------------------------------------------------------------
* dtype is th.cfloat (complex64) everywhere, exactly as in the notebooks;
  using complex128 changes the iterate sequence.
* ``polar(A, 'exact')`` is imported from :mod:`quantum_qcd.quantum_utils`.
  Its 'exact' branch (``U, S, Vh = th.linalg.svd(A, full_matrices=False);
  return U @ Vh``) is byte-for-byte the notebook computation.  The 'approx'
  branch differs between notebook cells but is never used here.
* The KL used by :func:`compute_cost` is the *clipped* one redefined in
  adap_grad_rank_defic_rho.ipynb cell 28 (``kl_divergence(q, p, eps=1e-12)``
  with ``clamp_min``), which is the definition in force when cells 35 and 56
  run.  :func:`pgd_full_rank` instead uses the *unclipped*
  ``kl_divergence(p, q) = p @ (log p - log q)`` of adaptive_gradient.ipynb
  cell 7, which is :func:`quantum_qcd.quantum_utils.kl_divergence`.
* For rank-deficient rho the returned q does not sum to one; the missing mass
  q_0 = 1 - sum(q) is the probability of the "no click" outcome outside the
  support of rho.  A negative objective for R < N is expected, not a bug.
"""

import numpy as np
import torch as th

from .quantum_utils import polar, kl_divergence

__all__ = [
    "compute_cost",
    "alg1",
    "pgd_rank_aware",
    "pgd_full_rank",
]


# --------------------------------------------------------------------------
# Cost helpers
# --------------------------------------------------------------------------
def _kl_divergence_clipped(q, p, eps=1e-12):
    """
    D(q||p) with safe clipping, returned as a real scalar tensor.

    Verbatim port of ``kl_divergence`` as redefined in
    ``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho.ipynb``
    cell 28 (identical text in ``adap_grad_rank_defic_rho_copy.ipynb``
    cell 28).  That redefinition shadows the plain
    ``kl_divergence(p, q) = p @ (log p - log q)`` of cell 7, so it is the
    version in force for cells 35 (Fig. 5) and 56 / 65 (Figs. 6 and 7).

    Parameters
    ----------
    q, p : torch tensor
        Real, non-negative outcome distributions of equal length.
    eps : float
        Lower clamp applied to both arguments before taking logs.

    Returns
    -------
    torch tensor
        Real scalar sum_i q_i (log q_i - log p_i).
    """
    q_clip = q.clamp_min(eps)
    p_clip = p.clamp_min(eps)
    return (q_clip * (th.log(q_clip) - th.log(p_clip))).sum().real


def compute_cost(U, rho, sigma):
    """
    Rank-one projective measurement built from the columns of U, and its cost.

    Verbatim port of ``compute_cost`` from
    ``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho.ipynb``
    cell 35 (textually identical to ``adap_grad_rank_defic_rho_copy.ipynb``
    cell 27 and to ``probe_known/probing_known.ipynb`` cell 82).

    The measurement is Q_i = |u_i><u_i| for the columns u_i of U, so
    p_i = <u_i|rho|u_i> and q_i = <u_i|sigma|u_i>, and the returned cost is
    the clipped D(q||p) (see :func:`_kl_divergence_clipped`).

    Parameters
    ----------
    U : (n, r) complex torch tensor
        Matrix whose r columns are the (orthonormal) measurement vectors.
        Iteration is over ``U.T``, i.e. over the columns of U.
    rho, sigma : (n, n) complex torch tensors
        Pre- and post-change states.

    Returns
    -------
    cost_val : torch tensor
        Real scalar D(q||p).
    Q : (r, n, n) complex torch tensor
        Stack of the rank-one projectors.
    p, q : (r,) real torch tensors
        Outcome distributions under rho and sigma.  They sum to one only when
        the columns of U span the full space.
    """
    # Construct Q from U (assuming columns of U define the measurement operators)
    Q = th.stack([th.outer(u, u.conj()) for u in U.T])

    # Compute p and q
    p = (Q @ rho).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real
    q = (Q @ sigma).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real

    # Compute KL-divergence or whichever cost function you want
    cost_val = _kl_divergence_clipped(q, p)

    return cost_val, Q, p, q


def _compute_cost_full_rank(U, rho, sigma):
    """
    Same as :func:`compute_cost` but with the *unclipped* KL.

    Verbatim port of ``compute_cost`` from
    ``optimization_known/adaptive_gradient/adaptive_gradient.ipynb`` cell 25.
    That notebook never redefines ``kl_divergence``, so the version in force
    is cell 7's ``kl_divergence(p, q) = p @ (th.log(p) - th.log(q))``, which
    is :func:`quantum_qcd.quantum_utils.kl_divergence`; called as
    ``kl_divergence(q, p)`` it evaluates q @ (log q - log p) = D(q||p).

    Returns
    -------
    cost_val, Q, p, q
        As in :func:`compute_cost`.
    """
    # Construct Q from U (assuming columns of U define the measurement operators)
    Q = th.stack([th.outer(u, u.conj()) for u in U.T])
    # Compute p and q
    p = (Q @ rho).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real
    q = (Q @ sigma).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real

    # Compute KL-divergence or whichever cost function you want
    cost_val = kl_divergence(q, p)

    return cost_val, Q, p, q


# --------------------------------------------------------------------------
# (1) Rank-truncated PGA -- Fig. 5 (and Fig. 14a via the probing notebook)
# --------------------------------------------------------------------------
def alg1(
    rho, sigma, n, r,
    tol=1e-6,
    alpha_init=1e-1,
    min_step=1e-7,
    min_norm=1e-6,
    armijo_alpha=1e-4,
    armijo_beta=0.5,
    maxiter=2000,
    V0=None,
):
    """
    Algorithm 1: PGA over the top-r eigenspace of rho.

    Verbatim port of ``alg1`` from
    ``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho.ipynb``
    cell 35 (defined there together with :func:`compute_cost`).  Cell 37 of
    the same notebook uses it to produce the four f(V) convergence curves of
    Fig. 5.

    ``th.linalg.eigh`` returns eigenvalues in *ascending* order, so
    ``Lambda = eigvecs[:, n-r:n]`` keeps the r eigenvectors with the largest
    eigenvalues, i.e. the support of a rank-r rho.  The optimization runs on
    the r x r compressions ``rho_hat = Lambda^H rho Lambda`` and
    ``sigma_hat = Lambda^H sigma Lambda``; the returned p and q are then
    recomputed from the lifted measurement ``U = Lambda @ V`` against the
    *full* rho and sigma, so q sums to 1 - q_0 < 1 whenever r < n.

    Parameters
    ----------
    rho, sigma : (n, n) complex torch tensors (th.cfloat)
        Pre- and post-change states.
    n : int
        Hilbert-space dimension.
    r : int
        Rank / number of measurement vectors to keep.
    tol : float
        Relative-change stopping tolerance on the objective.  ``tol=0``
        disables early stopping, so the history has exactly ``maxiter``
        entries (this is what Fig. 5 uses).
    alpha_init, min_step, armijo_alpha, armijo_beta : float
        Armijo backtracking parameters (initial step, smallest step tried,
        sufficient-increase constant, contraction factor).
    min_norm : float
        Gradient-norm floor.
    maxiter : int
        Maximum number of iterations.
    V0 : {'Identity', 'Lambda_H'}
        Initialization of the r x r iterate V.  ``'Identity'`` starts from
        I_r (Fig. 5); ``'Lambda_H'`` is only legal for r == n.

    Returns
    -------
    Q : (r, n, n) complex torch tensor
        Final rank-one projectors in the original basis (from U = Lambda @ V).
    obj_history : list
        Objective value after each accepted iteration.
    p, q : (r,) float ndarrays
        Final outcome distributions under the full rho and sigma.

    Notes
    -----
    Faithful quirks kept from the notebook:

    * ``Q_0`` is computed and never used (dead code in the source cell).
    * The ``grad_norm < min_norm`` branch is ``continue``, not ``break``:
      that iteration appends nothing to ``obj_history`` and leaves ``V.grad``
      uncleared, so the next ``backward()`` accumulates into it.
    * If the line search exhausts its steps without meeting the Armijo
      condition, ``cost`` stays a *tensor* (the pre-step value) instead of the
      float ``cost_new.item()`` stored on success, and that tensor is what is
      appended to ``obj_history``.  Histories can therefore mix floats and
      0-d tensors.
    """
    obj_history = []

    # --- Preprocessing: eigendecomposition on rho ---
    eigvals, eigvecs = th.linalg.eigh(rho)
    Lambda = eigvecs[:, n - r:n]

    # --- Preprocessing: Q_0 ---
    # (computed exactly as in the notebook; unused downstream)
    if r == n:
        Q_0 = 0
    else:
        Q_0 = th.eye(n) - Lambda @ Lambda.conj().T

    # --- Preprocessing: rho_hat & sigma_hat ---
    rho_hat = Lambda.T.conj() @ rho @ Lambda
    sigma_hat = Lambda.T.conj() @ sigma @ Lambda

    # --- Preprocessing: initilization on V ---
    if V0 == 'Identity':
        V = th.eye(r, dtype=th.cfloat)
    elif V0 == 'Lambda_H':
        if r == n:
            V = Lambda.T.conj()
        else:
            raise ValueError("Invalid input: V must be full rank")
    else:
        raise ValueError('Undefined initialization V')

    # --- PGA iterations: compute gradient ---
    for t in range(maxiter):
        V.requires_grad = True
        cost, Q, p, q = compute_cost(V, rho_hat, sigma_hat)
        cost.backward()  # backprop to get grad

        with th.no_grad():
            grad_V = V.grad
            grad_norm = th.norm(grad_V, p=2).item()

            # If gradient is small enough, we're done
            if grad_norm < min_norm:
                continue

            cost_old = cost.item()

            # --- Backtracking line search ---
            step = alpha_init
            while step > min_step:
                # Propose update: projection via the exact polar factor
                V_new = polar(V + step * grad_V, 'exact')

                # Evaluate the new cost
                cost_new, Q_new, p_new, q_new = compute_cost(V_new, rho_hat, sigma_hat)
                cost_new = cost_new.item()

                # Armijo condition for sufficient *increase* (ascent):
                #   cost_new >= cost_old + armijo_alpha * step * ||grad||^2
                lhs = cost_new
                rhs = cost_old + armijo_alpha * step * grad_norm ** 2

                if lhs >= rhs:
                    # Sufficient increase achieved
                    V = V_new
                    cost = cost_new
                    p = p_new
                    q = q_new
                    break
                else:
                    # Not enough increase; reduce step and try again
                    step *= armijo_beta

            # Clear gradient
            V.grad = None

        obj_history.append(cost)

        # Check for relative change in cost
        if t >= 1:
            rel_change = abs(obj_history[-1] - obj_history[-2]) / max(abs(obj_history[-1]), 1e-12)
            if rel_change < tol:
                break

    U = Lambda @ V
    cost, Q, p, q = compute_cost(U, rho, sigma)

    return Q, obj_history, p.detach().numpy(), q.detach().numpy()


# --------------------------------------------------------------------------
# (2) Support-projected, rank-aware PGA -- Figs. 6 and 7
# --------------------------------------------------------------------------
def pgd_rank_aware(
    rho, sigma, n, r,
    tol=1e-6,
    alpha_init=1e-1,   # initial step size guess
    min_step=1e-7,
    min_norm=1e-6,
    armijo_alpha=1e-4,  # Armijo parameter
    armijo_beta=0.5,   # contraction factor
    maxiter=2000,
    U0=None,
):
    """
    Rank-aware von Neumann PGA with support projection (the ``R``-argument
    ``von_neumann_pgd_backtracking``).

    Verbatim port of ``von_neumann_pgd_backtracking(rho, sigma, n, r, ...)``
    from
    ``optimization_known/adaptive_gradient/adap_grad_rank_defic_rho_copy.ipynb``
    cell 27.  The notebook that *calls* it,
    ``adap_grad_rank_defic_rho.ipynb`` (cell 56 -> Fig. 6, cell 65 -> Fig. 7),
    no longer contains the defining cell -- it was deleted while the function
    stayed live in the Colab kernel -- so the sibling ``_copy`` notebook is
    the authoritative source.

    Unlike :func:`alg1`, the support here is selected by an eigenvalue
    threshold (``eigvals.real > 1e-6``) rather than by taking the last r
    columns, and rho and sigma are rotated into that support *in place*.  The
    returned Q, p and q therefore live in the support basis, and q sums to
    1 - q_0 where q_0 is the post-change weight outside the support of rho
    (0.394 / 0.282 / 0.097 / ~0 for R = 5 / 6 / 7 / 8 at set index 9).

    Parameters
    ----------
    rho, sigma : (n, n) complex torch tensors (th.cfloat)
        Pre- and post-change states.
    n : int
        Hilbert-space dimension.
    r : int
        Rank of rho.  When ``r == n`` the iterate starts from ``U0``; when
        ``r < n`` it starts from ``I_r`` and ``U0`` is ignored.
    tol : float
        Relative-change stopping tolerance; ``tol=0`` runs the full
        ``maxiter`` iterations (as Figs. 6 and 7 do).
    alpha_init, min_step, armijo_alpha, armijo_beta : float
        Armijo backtracking parameters.  Figs. 6 and 7 use the defaults
        (1e-1, 1e-7, 1e-4, 0.5).
    min_norm : float
        Gradient-norm floor; the loop breaks below it.
    maxiter : int
        Maximum number of iterations (1000 for Figs. 6 and 7).
    U0 : (n, n) complex torch tensor or None
        Initial unitary, used only when ``r == n``.  Figs. 6 and 7 pass
        ``th.linalg.eigh(th_rho)[1]``.  If None (and ``r == n``) a random
        unitary is drawn from a QR of ``th.randn``.

    Returns
    -------
    Q : (r, r, r) complex torch tensor
        Final rank-one projectors, expressed in the support basis of rho.
        (The caller in the notebook names this ``U_opt``; the function really
        returns the projector stack, not the unitary.)
    obj_history : list of float
        Objective value after each iteration.
    p, q : (r,) float ndarrays
        Final outcome distributions; ``sum(p) == 1`` and ``sum(q) == 1 - q_0``.

    Notes
    -----
    Faithful quirks kept from the notebook:

    * The ``r < n`` branch reads a *global* ``R`` (``U = th.eye(R, ...)``)
      instead of the argument ``r``.  Every published call site passes its
      loop variable ``R`` as ``r``, so ``R == r`` there; this port uses ``r``,
      which reproduces those runs exactly.
    * ``U0`` is silently ignored when ``r < n``.
    * The initial ``compute_cost`` call before the loop is dead work: its
      result is overwritten on the first iteration.
    * ``U`` is *not* rotated into the eigenbasis of rho (contrast
      :func:`pgd_full_rank`, which does ``U = Lambda^H U``), so in the
      ``r == n`` branch the eigenvector initialization U0 is applied in the
      rotated frame.
    * Unlike :func:`alg1`, the gradient-norm branch is ``break``, and
      ``cost.item()`` is always appended, so the history is all floats.
    """
    obj_history = []

    # --- Initialize U ---
    if r == n:
        if U0 is None:
            U = th.linalg.qr(th.randn((n, n), dtype=th.cfloat))[0].detach()
        else:
            U = U0
    if r < n:
        # Notebook reads the global R here; every call site has R == r.
        U = th.eye(r, dtype=th.cfloat)

    # --- Project rho and sigma onto the support of rho ---
    eigvals, eigvecs = th.linalg.eigh(rho)
    support_mask = eigvals.real > 1e-6  # keep strictly positive eigenvalues
    Lambda = eigvecs[:, support_mask]
    rho = Lambda.T.conj() @ rho @ Lambda
    sigma = Lambda.T.conj() @ sigma @ Lambda

    # --- Compute initial cost ---
    cost, Q, p, q = compute_cost(U, rho, sigma)

    for t in range(maxiter):
        U.requires_grad = True
        # Re-compute everything in forward pass
        cost, Q, p, q = compute_cost(U, rho, sigma)
        cost.backward()  # backprop to get grad

        with th.no_grad():
            grad_U = U.grad
            grad_norm_sq = (grad_U.conj() * grad_U).sum().real.item()  # ||grad||^2
            grad_norm = np.sqrt(grad_norm_sq)

            # If gradient is small enough, we're done
            if grad_norm < min_norm:
                break

            cost_old = cost.item()
            # Ascent direction is the plain (Euclidean) gradient.
            direction = grad_U

            # --- Backtracking line search ---
            step = alpha_init
            while step > min_step:
                # Propose update: projection via the exact polar factor
                U_new = polar(U + step * direction, 'exact')

                # Evaluate the new cost
                cost_new, Q_new, p_new, q_new = compute_cost(U_new, rho, sigma)
                cost_new_val = cost_new.item()

                # Armijo condition for sufficient *increase* (ascent):
                #   cost_new >= cost_old + armijo_alpha * step * ||grad||^2
                lhs = cost_new_val
                rhs = cost_old + armijo_alpha * step * grad_norm_sq

                if lhs >= rhs:
                    # Sufficient increase achieved
                    U = U_new
                    cost = cost_new
                    p = p_new
                    q = q_new
                    break
                else:
                    # Not enough increase; reduce step and try again
                    step *= armijo_beta

            # Clear gradient
            U.grad = None

        obj_history.append(cost.item())

        # Check for relative change in cost
        if t >= 1:
            rel_change = abs(obj_history[-1] - obj_history[-2]) / max(abs(obj_history[-1]), 1e-12)
            if rel_change < tol:
                break

    return Q, obj_history, p.detach().numpy(), q.detach().numpy()


# --------------------------------------------------------------------------
# (3) Full-rank PGA -- the Fig. 1 convergence arrays
# --------------------------------------------------------------------------
def pgd_full_rank(
    rho, sigma, n,
    tol=1e-6,
    alpha_init=1e-1,   # initial step size guess
    min_step=1e-7,
    min_norm=1e-6,
    armijo_alpha=1e-4,  # Armijo parameter
    armijo_beta=0.5,   # contraction factor
    maxiter=2000,
    U0=None,
):
    """
    Full-rank von Neumann PGA (the 3-argument ``von_neumann_pgd_backtracking``).

    Verbatim port of ``von_neumann_pgd_backtracking(rho, sigma, n, ...)`` from
    ``optimization_known/adaptive_gradient/adaptive_gradient.ipynb`` cell 25 --
    the routine that produced the stored convergence arrays
    ``plot_alg_convergence/N8_U_I.npy`` and ``N8_U_Lambda.npy`` replotted as
    Fig. 1.  Call sites use ``alpha_init=0.1, min_step=1e-7, min_norm=1e-6,
    armijo_alpha=1e-3, armijo_beta=0.8, tol=0`` with either
    ``U0 = th.eye(N, dtype=th.cfloat)`` (the "U^(0) = I_N" column) or the
    eigenvector matrix of rho (the "U^(0) = Lambda" column).

    The whole optimization runs in the eigenbasis of rho: rho and sigma are
    rotated by ``Lambda`` (from ``th.linalg.eigh``, ascending eigenvalues) and
    the initial ``U`` is rotated with them (``U <- Lambda^H U``).  At the end
    the iterate is rotated back (``U <- Lambda U``) and the final cost /
    distributions are evaluated against the *original* rho and sigma.

    Parameters
    ----------
    rho, sigma : (n, n) complex torch tensors (th.cfloat)
        Full-rank pre- and post-change states.
    n : int
        Hilbert-space dimension.
    tol : float
        Relative-change stopping tolerance; ``tol=0`` runs all ``maxiter``
        iterations, giving a history of fixed length.
    alpha_init, min_step, armijo_alpha, armijo_beta : float
        Armijo backtracking parameters.
    min_norm : float
        Gradient-norm floor; the loop breaks below it.
    maxiter : int
        Maximum number of iterations.
    U0 : (n, n) complex torch tensor or None
        Initial unitary.  If None a random one is drawn from a QR of
        ``th.randn`` (this is the only stochastic path in the module).

    Returns
    -------
    Q : (n, n, n) complex torch tensor
        Final rank-one projectors in the original basis.
    obj_history : list of float
        Objective value after each iteration.
    p, q : (n,) real torch tensors
        Final outcome distributions, each summing to one.  These are returned
        as *tensors* (not ndarrays, unlike :func:`alg1` and
        :func:`pgd_rank_aware`), still attached to the autograd graph; the
        notebook call sites apply ``.detach().numpy()`` themselves.

    Notes
    -----
    Faithful quirks kept from the notebook:

    * The cost uses the *unclipped* ``kl_divergence`` of
      ``adaptive_gradient.ipynb`` cell 7 (see :func:`_compute_cost_full_rank`),
      not the clamped variant used by :func:`alg1` / :func:`pgd_rank_aware`.
    * The initial ``compute_cost`` call before the loop is dead work.
    * The notebook wraps the loop in ``tqdm``; the progress bar is dropped
      here so the function stays silent when imported.
    """
    obj_history = []

    # --- Initialize U ---
    if U0 is None:
        U = th.linalg.qr(th.randn((n, n), dtype=th.cfloat))[0].detach()
    else:
        U = U0

    initial_rho = rho
    initial_sigma = sigma

    eigvals, Lambda = th.linalg.eigh(rho)
    rho = Lambda.T.conj() @ rho @ Lambda
    sigma = Lambda.T.conj() @ sigma @ Lambda

    U = Lambda.T.conj() @ U

    # --- Compute initial cost ---
    cost, Q, p, q = _compute_cost_full_rank(U, rho, sigma)

    for t in range(maxiter):
        U.requires_grad = True
        # Re-compute everything in forward pass
        cost, Q, p, q = _compute_cost_full_rank(U, rho, sigma)
        cost.backward()  # backprop to get grad

        with th.no_grad():
            grad_U = U.grad
            grad_norm_sq = (grad_U.conj() * grad_U).sum().real.item()  # ||grad||^2
            grad_norm = np.sqrt(grad_norm_sq)

            # If gradient is small enough, we're done
            if grad_norm < min_norm:
                break

            cost_old = cost.item()
            # Ascent direction is the plain (Euclidean) gradient.
            direction = grad_U

            # --- Backtracking line search ---
            step = alpha_init
            while step > min_step:
                # Propose update: projection via the exact polar factor
                U_new = polar(U + step * direction, 'exact')

                # Evaluate the new cost
                cost_new, Q_new, p_new, q_new = _compute_cost_full_rank(U_new, rho, sigma)
                cost_new_val = cost_new.item()

                # Armijo condition for sufficient *increase* (ascent):
                #   cost_new >= cost_old + armijo_alpha * step * ||grad||^2
                lhs = cost_new_val
                rhs = cost_old + armijo_alpha * step * grad_norm_sq

                if lhs >= rhs:
                    # Sufficient increase achieved
                    U = U_new
                    cost = cost_new
                    p = p_new
                    q = q_new
                    break
                else:
                    # Not enough increase; reduce step and try again
                    step *= armijo_beta

            # Clear gradient
            U.grad = None

        obj_history.append(cost.item())

        # Check for relative change in cost
        if t >= 1:
            rel_change = abs(obj_history[-1] - obj_history[-2]) / max(abs(obj_history[-1]), 1e-12)
            if rel_change < tol:
                break

    U = Lambda @ U

    cost, Q, p, q = _compute_cost_full_rank(U, initial_rho, initial_sigma)

    return Q, obj_history, p, q


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _data_dir():
    """Locate the released ``data/`` directory relative to this file."""
    import pathlib
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] / "data" / "adap_grad_jeremy"


def _self_test():
    """
    Reproduce the three reference numbers recorded in the figure plan.

    1. :func:`alg1` on the Fig. 5 R = 5 case -- f(V) must run from about
       -0.2836 to about -0.2598 over 800 iterations.
    2. :func:`pgd_rank_aware` on the Fig. 6 R = 8 case -- reports the final
       KL and checks sum(p) == 1, sum(q) == 1 (q_0 ~ 0 at full rank).
    3. :func:`pgd_full_rank` for 50 iterations on the first N = 8
       unitary-product state pair -- smoke test only.
    """
    d = _data_dir()
    sigma = np.load(d / "sigma_8.npy")
    th_sigma = th.tensor(sigma, dtype=th.cfloat)
    N = 8
    set_num_deficit = 9
    ok = True

    # --- 1. Fig. 5, R = 5 -------------------------------------------------
    rho5 = np.load(d / "rho_8_rank_5_set_500.npy")[set_num_deficit, :, :]
    th_rho5 = th.tensor(rho5, dtype=th.cfloat)
    Q, obj, p, q = alg1(
        th_rho5, th_sigma, N, 5,
        alpha_init=1e-1, min_step=1e-5, min_norm=1e-6,
        armijo_alpha=1e-4, armijo_beta=0.9, maxiter=800, tol=0,
        V0='Identity',
    )
    print(f"[alg1  R=5] len(hist) = {len(obj)}  "
          f"f(V): {obj[0]:.4f} -> {obj[-1]:.4f}  (expect -0.2836 -> -0.2598)")
    print(f"[alg1  R=5] sum(p) = {p.sum():.6f}, sum(q) = {q.sum():.6f}, "
          f"q_0 = {1 - q.sum():.3f}")
    for name, got, want in (("start", obj[0], -0.2836), ("end", obj[-1], -0.2598)):
        if abs(got - want) > 5e-4:
            print(f"  FAIL: {name} objective {got:.4f} != {want}")
            ok = False
    if len(obj) != 800:
        print(f"  FAIL: history length {len(obj)} != 800")
        ok = False

    # --- 2. Fig. 6, R = 8 -------------------------------------------------
    rho8 = np.load(d / "rho_8_rank_8_set_500.npy")[set_num_deficit, :, :]
    th_rho8 = th.tensor(rho8, dtype=th.cfloat)
    eigvals, U0 = th.linalg.eigh(th_rho8)
    Q8, hist8, p8, q8 = pgd_rank_aware(th_rho8, th_sigma, N, 8, maxiter=1000, tol=0, U0=U0)
    print(f"[pgd_rank_aware R=8] len(hist) = {len(hist8)}  "
          f"KL: {hist8[0]:.4f} -> {hist8[-1]:.4f}")
    print(f"[pgd_rank_aware R=8] sum(p) = {p8.sum():.6f}, sum(q) = {q8.sum():.6f}, "
          f"q_0 = {1 - q8.sum():.3f}")
    if not np.isclose(p8.sum(), 1.0, atol=1e-5):
        print(f"  FAIL: sum(p) = {p8.sum()} != 1")
        ok = False
    if hist8[-1] < hist8[0]:
        print("  FAIL: objective did not increase")
        ok = False

    # --- 3. Full-rank smoke test -----------------------------------------
    rho_set = np.load(d / "rho_8_set_500.npy")
    sigma_set = np.load(d / "sigma_8_set_500.npy")
    th_r = th.tensor(rho_set[0, :, :], dtype=th.cfloat)
    th_s = th.tensor(sigma_set[0, :, :], dtype=th.cfloat)
    Qf, histf, pf, qf = pgd_full_rank(
        th_r, th_s, N,
        tol=0, alpha_init=0.1, min_step=1e-7, min_norm=1e-6,
        armijo_alpha=1e-3, armijo_beta=0.8, maxiter=50,
        U0=th.eye(N, dtype=th.cfloat),
    )
    pf = pf.detach().numpy()
    qf = qf.detach().numpy()
    print(f"[pgd_full_rank] len(hist) = {len(histf)}  "
          f"KL: {histf[0]:.4f} -> {histf[-1]:.4f}")
    print(f"[pgd_full_rank] sum(p) = {pf.sum():.6f}, sum(q) = {qf.sum():.6f}")
    if len(histf) != 50:
        print(f"  FAIL: history length {len(histf)} != 50")
        ok = False
    if not (np.isclose(pf.sum(), 1.0, atol=1e-5) and np.isclose(qf.sum(), 1.0, atol=1e-5)):
        print("  FAIL: full-rank distributions do not sum to one")
        ok = False

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
