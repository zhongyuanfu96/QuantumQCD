"""
Quantum-channel change detection: probe design and measurement design.

Faithful torch port of the code in

    notebooks_raw/Quantum_Anomaly_Detection/probe_known/probing_known.ipynb

that produces Fig. 14 (rank minimisation / joint optimisation objective
histories) and the measurement designs consumed by Fig. 15 (ADD-vs-FAP).

Physical set-up
---------------
A probe state ``rho`` on an ``N_a``-dimensional system is sent through one of
two completely positive trace preserving maps on an ``N_b``-dimensional output,

    pre-change   N(rho) = sum_{j<m1} K_j rho K_j^H      (Kraus set K, seed 11)
    post-change  M(rho) = sum_{j<m2} L_j rho L_j^H      (Kraus set L, seed 12)

and a rank-one projective measurement on the output turns the two states into
the two distributions ``p`` (pre-change) and ``q`` (post-change) that drive the
CUSUM detector.  Two designs are ported:

* **known post-change** -- the measurement maximises ``KL(q||p)``
  (``alg1``, projected gradient ascent on the Stiefel manifold);
* **unknown post-change** -- the measurement is the eigenbasis of ``N(rho)``
  and the probe maximises the sensitivity ``sum_i 1/lambda_i(N(rho))``.

Both designs first shrink the rank of ``N(rho)`` with a log-det heuristic
(``rank_minimization``), because a rank-deficient pre-change output creates
outcomes with ``p_i = 0 < q_i`` that the detector can exploit.

Cell provenance (all line-for-line ports, seeds preserved)
----------------------------------------------------------
* cells 2 / 22  -> :func:`make_kraus`, :func:`apply_channel`,
  :func:`generate_pure_state`, :func:`load_X_rho`
* cell 97       -> :func:`rank_minimization`, :func:`rank_minimization_sweep`,
  :func:`measurement_known`   (known post-change, Fig. 14a)
* cells 79 + 84 -> :func:`joint_optimization_known`   (Fig. 14b:
  cell 84's driver, ``max_iter_U = 5``, on top of **cell 79's** definition of
  ``von_neumann_pgd_back_probing_alg1``.  Cells 79 and 82 both define that
  function with execution count 25 and they differ: cell 79 backtracks along
  the probe gradient, cell 82 takes a fixed unit step.  Cell 84's stored
  ``p``/``q`` are reproduced to five significant digits only by cell 79's
  version -- see :func:`joint_optimization_known` -- while cell 82's own
  stored output is reproduced by its own fixed step at ``max_iter_U = 10``
  (plateau 1.354, not the published 1.852).  The figure plan lists only cells
  82/84 for this panel, which is not enough to reproduce it.)
* cell 105      -> :func:`measurement_unknown`   (same seeded rank
  minimisation, but the measurement is the eigenbasis of ``N(rho)`` and
  ``p``/``q`` are truncated to their last ``r`` entries)
* cell 109      -> :func:`joint_optimization_unknown`   (Fig. 14c)
* cell 118's branch test -> :func:`design_known` / :func:`design_unknown`

Determinism
-----------
Every entry point is deterministic: the Kraus operators come from
``numpy.random.default_rng(11 / 12)``, the rank-minimisation probe from
``th.manual_seed(10)``, the known joint probe from ``generate_pure_state(8, 5)``
(which seeds with 10 -> 5 internally) and the unknown joint probe from the
stored ``data/X/X_8_rho.npy``.  Nothing else touches a random number generator.

Note on ``alg1`` / ``compute_cost``
-----------------------------------
Private copies live here rather than being imported from
``src/quantum_qcd/pga.py``, deliberately.  ``pga.alg1`` / ``pga.compute_cost``
port ``adap_grad_rank_defic_rho.ipynb`` cell 35, which is a *different*
function from the one probing_known runs:

* it returns ``(Q, obj_history, p, q)``; probing_known cell 82's returns
  ``(Q, U, cost, obj_history)`` and cells 84/97 need the ``U``;
* its cost uses the clipped KL of that notebook's cell 28, not the plain
  ``kl_divergence`` in force in probing_known;
* it carries quirks of its own (a global ``R`` in the ``r < n`` branch,
  uncleared gradients on a skipped iteration).

The copies below are therefore cell 82's, which is the definition in force
for cells 84 / 97 / 117.  ``polar`` and ``kl_divergence`` are identical to the
versions already in ``quantum_utils`` and are imported from there.
"""

import os

import numpy as np
import torch as th

try:                                    # flat layout used by the other modules
    from quantum_utils import polar, kl_divergence
except ImportError:                     # package-relative fallback
    from .quantum_utils import polar, kl_divergence

try:
    from tqdm import tqdm
except ImportError:                     # tqdm is optional here
    def tqdm(it, **kw):
        return it


# --------------------------------------------------------------------------
# Defaults of probing_known.ipynb cells 84 / 97 / 105 / 109
# --------------------------------------------------------------------------
N_A = 8                       # probe (channel input) dimension
N_B = 6                       # channel output dimension
SEED_K = 11                   # pre-change Kraus seed
SEED_L = 12                   # post-change Kraus seed
SEED_RANK_MIN = 10            # th.manual_seed(10) at the top of cells 97/105
SEED_JOINT_PROBE = 5          # generate_pure_state(N_a, num_seed=5) in cell 84

M_LIST_RANK_MIN = [3, 4, 6, 12, 13]     # cell 97 / 105 sweep (Fig. 14a)
M_LIST_JOINT = [13, 25]                 # cell 84 / 109 sweep (Fig. 14b, 14c)

_HERE = os.path.dirname(os.path.abspath(__file__))
# src/quantum_qcd/channel.py -> <repo>/data/X
_DATA_X = os.path.join(_HERE, os.pardir, os.pardir, "data", "X")


# --------------------------------------------------------------------------
# Cell 2 / 22 helpers: channel construction and probe states
# --------------------------------------------------------------------------
def make_kraus(N_a=N_A, N_b=N_B, m=13, seed_num=None, verify=False):
    """
    Random Kraus operators of a CPTP map from C^{N_a} to C^{N_b}.

    Draws a complex Gaussian ``(m*N_b, N_a)`` matrix, orthonormalises its
    columns with a QR decomposition (so ``Q^H Q = I_{N_a}``) and slices it into
    ``m`` blocks of shape ``(N_b, N_a)``.  The completeness relation
    ``sum_j K_j^H K_j = I_{N_a}`` then holds by construction.

    Parameters
    ----------
    N_a, N_b : int
        Input and output dimensions.  Requires ``m * N_b >= N_a``.
    m : int
        Number of Kraus operators (``m1`` or ``m2`` in the paper).
    seed_num : int or None
        Seed for ``numpy.random.default_rng``; 11 for the pre-change channel
        and 12 for the post-change channel throughout the paper.
    verify : bool
        If True also return the Frobenius completeness error.

    Returns
    -------
    kraus_ops : ndarray, shape (m, N_b, N_a)
    err : float
        Only when ``verify=True``.
    """
    if m * N_b < N_a:
        raise ValueError(
            f"Need m*N_b >= N_a for completeness (got m*N_b={m * N_b} < N_a={N_a})."
        )

    rng = np.random.default_rng(seed_num)
    A = (rng.standard_normal((m * N_b, N_a)) +
         1j * rng.standard_normal((m * N_b, N_a))) / np.sqrt(2)

    Q, _ = np.linalg.qr(A)                        # orthonormal columns
    kraus_ops = [Q[i * N_b:(i + 1) * N_b, :] for i in range(m)]

    if verify:
        acc = sum(K.conj().T @ K for K in kraus_ops)
        err = np.linalg.norm(acc - np.eye(N_a), "fro")
        return np.array(kraus_ops), err

    return np.array(kraus_ops)


def kraus_pair(N_a=N_A, N_b=N_B, m1=13, m2=None, seed_K=SEED_K, seed_L=SEED_L,
               dtype=th.cfloat):
    """
    Convenience wrapper: the pre- and post-change Kraus sets as torch tensors.

    Returns ``(th_K, th_L, err_K, err_L)`` with ``th_K`` of shape
    ``(m1, N_b, N_a)`` and ``th_L`` of shape ``(m2, N_b, N_a)``; ``m2``
    defaults to ``m1`` as in every cell of the notebook.
    """
    if m2 is None:
        m2 = m1
    K_operator, err_K = make_kraus(N_a, N_b, m1, seed_num=seed_K, verify=True)
    L_operator, err_L = make_kraus(N_a, N_b, m2, seed_num=seed_L, verify=True)
    return (th.tensor(K_operator, dtype=dtype),
            th.tensor(L_operator, dtype=dtype),
            err_K, err_L)


def apply_channel(mat, num_mat, matrix, N, is_rho=False):
    """
    Apply a quantum channel: ``N(rho) = sum_j mat[j] rho mat[j]^H / tr(rho)``.

    This is the notebook's ``channel_matrix``.  Two input conventions:

    * ``is_rho=True``  -- ``matrix`` already is the density matrix ``rho``;
    * ``is_rho=False`` -- ``matrix`` is a factor ``X`` and ``rho = X^H X``
      (the parameterisation used by the joint optimisations, which keeps
      ``rho`` positive semidefinite for free).

    The output is divided by ``tr(rho)``, so an unnormalised ``X`` is fine.

    Parameters
    ----------
    mat : tensor, shape (>= num_mat, N, N_a)
        Kraus operators.
    num_mat : int
        How many of them to sum over (``m1`` or ``m2``).
    matrix : tensor
        ``rho`` or its factor ``X``.
    N : int
        Output dimension ``N_b``.
    is_rho : bool
        Selects the convention above.

    Returns
    -------
    tensor, shape (N, N) -- the channel output, trace one.
    """
    chan_sum = th.zeros((N, N), dtype=matrix.dtype, device=matrix.device)
    rho = matrix if is_rho else matrix.conj().T @ matrix

    for i in range(num_mat):
        chan_sum = chan_sum + mat[i] @ rho @ mat[i].conj().T

    return chan_sum / th.trace(rho)


def generate_pure_state(N_a, num_seed):
    """
    Seeded random pure state, as in probing_known cells 82 / 109.

    ``th.manual_seed(num_seed)`` is called first, so the result depends only on
    ``num_seed`` and not on the ambient RNG state.

    Returns
    -------
    psi : tensor, shape (N_a,)
        Normalised complex state vector.
    matrix_psi : tensor, shape (N_a, N_a)
        ``psi`` written into row 0 of a zero matrix.  This is the ``X`` factor
        the joint optimisation starts from: ``X^H X = conj(psi) outer psi`` is
        the rank-one density matrix of ``conj(psi)``.
    """
    th.manual_seed(num_seed)

    psi = th.randn(N_a, dtype=th.cfloat) + 1j * th.randn(N_a, dtype=th.cfloat)
    psi = psi / th.linalg.norm(psi)

    matrix_psi = th.zeros((N_a, N_a), dtype=th.cfloat)
    matrix_psi[0, :] = psi

    return psi, matrix_psi


def load_X_rho(n=N_A, data_dir=None):
    """
    Load the stored seed matrix ``X_{n}_rho.npy`` (notebook ``generate_X_single``
    with ``load_exist=True``).

    Cell 109 starts the unknown-post-change joint optimisation from this file,
    which is why that run is deterministic without any seed call.  Searches
    ``data_dir``, then ``<repo>/data/X``, then ``./data/X``.
    """
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir)
    candidates += [_DATA_X, os.path.join("data", "X")]

    for d in candidates:
        path = os.path.join(d, f"X_{n}_rho.npy")
        if os.path.exists(path):
            return np.load(path)

    raise FileNotFoundError(
        f"X_{n}_rho.npy not found in any of: {candidates}"
    )


# --------------------------------------------------------------------------
# Private copies of probing_known cell 82's compute_cost / alg1
# --------------------------------------------------------------------------
def compute_cost(U, rho, sigma):
    """
    KL cost of the rank-one projective measurement whose vectors are the
    columns of ``U``.

    ``Q[i] = |u_i><u_i|``, ``p_i = tr(Q_i rho)``, ``q_i = tr(Q_i sigma)`` and
    the returned cost is ``KL(q||p)`` -- note the argument order: the
    post-change distribution comes first, matching the CUSUM drift.

    Returns ``(cost, Q, p, q)``.
    """
    Q = th.stack([th.outer(u, u.conj()) for u in U.T])

    p = (Q @ rho).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real
    q = (Q @ sigma).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real

    cost_val = kl_divergence(q, p)
    return cost_val, Q, p, q


def _backtracking_line_search(eval_step_fn, cost_old, expected_increase,
                              alpha_init=1e-1, min_step=1e-7,
                              armijo_alpha=1e-4, armijo_beta=0.5):
    """
    Armijo backtracking for *ascent* (both notebook copies test
    ``cost_new >= cost_old + armijo_alpha * step * expected_increase``; cell
    109's docstring says "decrease" but its body is identical to cell 82's).

    ``eval_step_fn(step)`` must return a tuple whose first element is the new
    cost.  Returns ``(success, step_outputs)`` and ``(False, None)`` if the
    step shrinks past ``min_step`` without satisfying Armijo.
    """
    step = alpha_init
    while step > min_step:
        step_outputs = eval_step_fn(step)
        cost_new = step_outputs[0]
        cost_new_val = cost_new.item() if hasattr(cost_new, 'item') else cost_new

        if cost_new_val >= cost_old + armijo_alpha * step * expected_increase:
            return True, step_outputs

        step *= armijo_beta

    return False, None


def alg1(rho, sigma, n, r,
         tol=1e-6, alpha_init=1e-1, min_step=1e-7, min_norm=1e-6,
         armijo_alpha=1e-4, armijo_beta=0.5, maxiter=2000, V0=None):
    """
    Algorithm 1 of the paper: projected gradient ascent for the maximum-KL
    von Neumann measurement on the rank-``r`` support of ``rho``.

    The eigenvectors of ``rho`` belonging to its ``r`` largest eigenvalues span
    ``Lambda``; the search then runs over an ``r x r`` unitary ``V``, projected
    back with an exact polar decomposition after every trial step, and the
    measurement returned is ``U = Lambda V`` padded implicitly by the ``n - r``
    directions on which ``p`` vanishes.

    Parameters follow the notebook exactly.  ``V0`` must be ``'Identity'``
    (used everywhere here) or ``'Lambda_H'`` (only legal when ``r == n``).
    ``tol=0`` disables the relative-change stopping rule, so the loop runs the
    full ``maxiter`` iterations.  ``min_norm`` is accepted for signature
    fidelity but, as in the notebook, is not consulted.

    Returns ``(Q, U, cost, obj_history)`` -- the cell-82 signature.
    """
    obj_history = []
    rho = rho.detach()
    sigma = sigma.detach()

    _, eigvecs = th.linalg.eigh(rho)
    Lambda = eigvecs[:, n - r:n]

    rho_hat = Lambda.T.conj() @ rho @ Lambda
    sigma_hat = Lambda.T.conj() @ sigma @ Lambda

    if V0 == 'Identity':
        V = th.eye(r, dtype=rho.dtype, device=rho.device)
    elif V0 == 'Lambda_H':
        if r == n:
            V = Lambda.T.conj()
        else:
            raise ValueError("Invalid input: V must be full rank")
    else:
        raise ValueError('Undefined initialization V')

    for t in range(maxiter):
        V = V.detach().requires_grad_(True)
        cost, Q, p, q = compute_cost(V, rho_hat, sigma_hat)
        cost.backward()

        with th.no_grad():
            grad_V = V.grad
            grad_norm = th.norm(grad_V, p=2).item()
            cost_old = cost.item()

            def eval_step(step):
                V_new = polar(V + step * grad_V, 'exact')
                cost_new, _, p_new, q_new = compute_cost(V_new, rho_hat, sigma_hat)
                return cost_new, V_new, p_new, q_new

            success, outputs = _backtracking_line_search(
                eval_step, cost_old, grad_norm ** 2,
                alpha_init, min_step, armijo_alpha, armijo_beta
            )

            if success:
                cost, V, p, q = outputs
                cost_val = cost.item() if hasattr(cost, 'item') else cost
            else:
                cost_val = cost_old
                V.grad = None
                break

            V.grad = None

        obj_history.append(cost_val)

        if t >= 2:
            rel_change = (abs(obj_history[-1] - obj_history[-2]) /
                          max(abs(obj_history[-1]), 1e-12))
            if rel_change < tol:
                break

    with th.no_grad():
        U = Lambda @ V
        cost, Q, p, q = compute_cost(U, rho, sigma)

    return Q, U, cost, obj_history


# --------------------------------------------------------------------------
# Cell 97 / 105 stage 1: log-det rank minimisation of the probe
# --------------------------------------------------------------------------
def rank_minimization(N_a=N_A, N_b=N_B, m1=13, m2=None, seed=SEED_RANK_MIN,
                      skip_draws=0, max_iter=2000, lr=0.002, eps=1e-6,
                      rank_tol=1e-5, seed_K=SEED_K, seed_L=SEED_L,
                      eye_dtype=th.float64, progress=False):
    """
    Shrink the rank of the pre-change channel output by minimising
    ``log det(N(rho) + eps I)`` over pure probe states.

    Port of the inner body of probing_known cell 97 (cell 105 runs the same
    optimisation).  A complex vector ``psi`` is optimised with Adam at
    ``lr=0.002`` for ``max_iter`` steps; at each step ``psi`` is normalised,
    ``rho = |psi><psi|`` is pushed through the channel and the log-det of the
    output is back-propagated.  The log-det is a standard smooth surrogate for
    the rank, so the plateau it reaches encodes ``R*``.

    Seeding
    -------
    The notebook seeds ``th.manual_seed(10)`` **once** before its ``m1`` loop,
    so the probe drawn for the k-th value of ``m1`` is the (k+1)-th
    ``th.randn`` of that stream.  Pass ``skip_draws=k`` to replay the stream
    for a single ``m1`` in isolation (``design_known`` / ``design_unknown`` do
    this automatically), or use :func:`rank_minimization_sweep` to run the
    whole loop as the notebook does.

    ``eye_dtype`` is the dtype of the ``eps * I`` regulariser.  In the notebook
    this is ``th.eye(N_b)``, i.e. whatever ``th.set_default_dtype`` was last
    set to; float32 and float64 give the same ranks and the same log-det curve
    to better than 1e-3, and float64 is used here so the result does not depend
    on a global.

    Returns
    -------
    psi : tensor, shape (N_a,)
        The optimised, normalised probe state vector (detached).
    history : list of float
        ``max_iter`` log-det values, one per Adam step (Fig. 14a).
    rank : int
        ``R* = #{ eigenvalues of N(rho) > rank_tol }``.
    """
    if m2 is None:
        m2 = m1

    th.manual_seed(seed)
    for _ in range(skip_draws):          # replay the notebook's RNG stream
        th.randn(N_a, dtype=th.cfloat)

    th_K, _, _, _ = kraus_pair(N_a, N_b, m1, m2, seed_K, seed_L)

    psi = th.randn(N_a, dtype=th.cfloat, requires_grad=True)
    optimizer = th.optim.Adam([psi], lr=lr)
    eye = eps * th.eye(N_b, dtype=eye_dtype)

    history = []
    for _ in tqdm(range(max_iter), disable=not progress):
        optimizer.zero_grad()

        psi_norm = psi / th.linalg.norm(psi)
        th_X_rho = th.outer(psi_norm, psi_norm.conj())
        # cell 97 calls channel_matrix without is_rho, so the projector is
        # treated as an X factor; X^H X == X for a normalised pure state.
        channel_N = apply_channel(th_K, m1, th_X_rho, N_b)

        cost = th.logdet(channel_N + eye).real
        cost.backward()
        optimizer.step()

        history.append(cost.item())

    with th.no_grad():
        psi_norm = (psi / th.linalg.norm(psi)).detach()
        th_X_rho = th.outer(psi_norm, psi_norm.conj())
        final_eigvals = th.linalg.eigvalsh(
            apply_channel(th_K, m1, th_X_rho, N_b))
        rank = int((final_eigvals > rank_tol).sum().item())

    return psi_norm, history, rank


def rank_minimization_sweep(m_list=None, N_a=N_A, N_b=N_B, seed=SEED_RANK_MIN,
                            **kwargs):
    """
    Run :func:`rank_minimization` for every ``m1`` in ``m_list`` off a single
    ``th.manual_seed(seed)`` stream -- exactly probing_known cell 97's loop.

    This is what Fig. 14a plots: for ``m_list = [3, 4, 6, 12, 13]`` and
    ``N_a, N_b = 8, 6`` it returns ranks ``R* = 2, 3, 4, 5, 6``.

    Returns
    -------
    dict : ``m1 -> (psi, history, rank)``
    """
    if m_list is None:
        m_list = M_LIST_RANK_MIN

    out = {}
    for k, m1 in enumerate(m_list):
        out[m1] = rank_minimization(N_a=N_a, N_b=N_b, m1=m1, seed=seed,
                                    skip_draws=k, **kwargs)
    return out


# --------------------------------------------------------------------------
# Cell 97 / 105 stage 2: the measurement on the rank-reduced probe
# --------------------------------------------------------------------------
def measurement_known(psi, rank, N_a=N_A, N_b=N_B, m1=13, m2=None,
                      max_iter_U=1000, seed_K=SEED_K, seed_L=SEED_L,
                      alpha_init=1, min_step=1e-6, min_norm=1e-6,
                      armijo_alpha=1e-3, armijo_beta=0.9, tol=0):
    """
    Maximum-KL measurement for a *known* post-change channel (cell 97, part 2).

    Runs :func:`alg1` on ``(N(rho), M(rho))`` restricted to the rank-``rank``
    support of ``N(rho)``, then reads off ``p_i = tr(Q_i N(rho))`` and
    ``q_i = tr(Q_i M(rho))``.

    Returns ``(p, q, Q)`` with ``p, q`` numpy arrays of length ``rank``.
    ``p`` sums to one; ``q`` generally does not, the deficit
    ``1 - sum(q)`` being the post-change probability of the outcomes on which
    the pre-change output has no support (Fig. 15's CUSUM appends it as an
    extra outcome).
    """
    if m2 is None:
        m2 = m1

    th_K, th_L, _, _ = kraus_pair(N_a, N_b, m1, m2, seed_K, seed_L)

    th_X_rho = th.outer(psi, psi.conj())
    channel_N = apply_channel(th_K, m1, th_X_rho, N_b)
    channel_M = apply_channel(th_L, m2, th_X_rho, N_b)

    Q, U, cost, _ = alg1(
        channel_N, channel_M, N_b, rank,
        tol=tol, alpha_init=alpha_init, min_step=min_step, min_norm=min_norm,
        armijo_alpha=armijo_alpha, armijo_beta=armijo_beta,
        maxiter=max_iter_U, V0='Identity',
    )

    p = (Q @ channel_N).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real
    q = (Q @ channel_M).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real
    return p.detach().numpy(), q.detach().numpy(), Q.detach()


def measurement_unknown(psi, rank, N_a=N_A, N_b=N_B, m1=13, m2=None,
                        seed_K=SEED_K, seed_L=SEED_L):
    """
    Maximum-sensitivity measurement for an *unknown* post-change channel
    (cell 105, part 2).

    No optimisation is needed: the measurement is the eigenbasis of ``N(rho)``,
    ``Q_i = |v_i><v_i|`` with ``v_i`` the eigenvectors returned by
    ``th.linalg.eigh`` in ascending eigenvalue order.  Because the first
    ``N_b - rank`` eigenvalues are numerically zero, both ``p`` and ``q`` are
    truncated to their **last** ``rank`` entries, which is where the notebook's
    ``p[N_b-r:]`` comes from.

    Returns ``(p, q, Q)`` with ``p, q`` numpy arrays of length ``rank``.
    """
    if m2 is None:
        m2 = m1

    th_K, th_L, _, _ = kraus_pair(N_a, N_b, m1, m2, seed_K, seed_L)

    th_X_rho = th.outer(psi, psi.conj())
    channel_N = apply_channel(th_K, m1, th_X_rho, N_b)
    channel_M = apply_channel(th_L, m2, th_X_rho, N_b)

    _, eigvecs = th.linalg.eigh(channel_N)
    Q = th.stack([th.outer(u, u.conj()) for u in eigvecs.T])

    p = (Q @ channel_N).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real
    q = (Q @ channel_M).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).real

    p = p.detach().numpy()[N_b - rank:]
    q = q.detach().numpy()[N_b - rank:]
    return p, q, Q.detach()


# --------------------------------------------------------------------------
# Cell 84: joint (probe + measurement) optimisation, known post-change
# --------------------------------------------------------------------------
def joint_optimization_known(N_a=N_A, N_b=N_B, m1=13, m2=None,
                             max_iter=200, max_iter_U=5,
                             seed_K=SEED_K, seed_L=SEED_L,
                             probe_seed=SEED_JOINT_PROBE, X0=None,
                             tol=0, tol_rho=1e-5, alpha_init=1,
                             min_step=1e-7, min_norm=1e-7,
                             armijo_alpha=1e-3, armijo_beta=0.9,
                             x_line_search=True,
                             progress=False, return_state=False):
    """
    Alternating maximisation of ``KL(q||p)`` over probe **and** measurement,
    known post-change channel.  Port of ``von_neumann_pgd_back_probing_alg1``
    driven with cell 84's parameters.

    One outer iteration:

    1. an ascent step along the normalised probe gradient,
       ``X <- X + step * G/||G||``, with ``step`` chosen by the same Armijo
       backtracking used inside ``alg1`` (``x_line_search=True``);
    2. ``alg1`` re-optimises the measurement for ``max_iter_U`` inner steps,
       cold-started from the identity every time;
    3. record ``KL(q||p)``.

    Which definition of the outer step?
    -----------------------------------
    probing_known defines ``von_neumann_pgd_back_probing_alg1`` twice with the
    same execution count (25): **cell 79** takes the backtracking step above,
    **cell 82** replaces it with a fixed ``X <- X + alpha_init * G/||G||``.
    Cell 84 (exec 26, the published Fig. 14b run) used **cell 79's**
    definition -- its stored ``p``/``q`` are reproduced to five significant
    digits by ``x_line_search=True`` and not at all by the fixed step.  Set
    ``x_line_search=False`` with ``max_iter_U=10`` to reproduce cell 82's own
    stored output instead (KL plateau 1.354 for ``m1=13``, 0.805 for
    ``m1=25``); that run is *not* the published panel.

    The probe starts from ``generate_pure_state(N_a, probe_seed)`` unless an
    ``X0`` factor is supplied.  ``tol=0`` disables early stopping, so the
    history has ``max_iter`` points -- the loop also breaks if the smallest
    eigenvalue of ``N(rho)`` drops below ``tol_rho``, which does not happen at
    the published settings.

    Fig. 14b uses ``m1 = m2 = 13`` and ``25`` with ``max_iter=200``,
    ``max_iter_U=5`` (cell 84).  Cell 82's ``max_iter_U=10`` is a different
    run.

    Returns
    -------
    ``(history, p, q)``, or ``(history, p, q, Q, X)`` when
    ``return_state=True``.
    """
    if m2 is None:
        m2 = m1

    th_K, th_L, _, _ = kraus_pair(N_a, N_b, m1, m2, seed_K, seed_L)

    if X0 is None:
        _, X = generate_pure_state(N_a, num_seed=probe_seed)
    else:
        X = X0.clone()

    obj_history = []

    with th.no_grad():
        X = X.detach()
        channel_N = apply_channel(th_K, m1, X, N_b)
        channel_M = apply_channel(th_L, m2, X, N_b)

    r = int(th.linalg.matrix_rank(channel_N).item())

    Q, U, cost, _ = alg1(
        channel_N, channel_M, N_b, r,
        tol=tol, alpha_init=alpha_init, min_step=min_step, min_norm=min_norm,
        armijo_alpha=armijo_alpha, armijo_beta=armijo_beta,
        maxiter=max_iter_U, V0='Identity',
    )

    for t in tqdm(range(max_iter), disable=not progress):
        X = X.detach().requires_grad_(True)

        channel_N = apply_channel(th_K, m1, X, N_b)
        channel_M = apply_channel(th_L, m2, X, N_b)
        r = int(th.linalg.matrix_rank(channel_N).item())

        cost, Q, p, q = compute_cost(U, channel_N, channel_M)
        cost.backward()

        with th.no_grad():
            G_X = X.grad
            norm_G_X = th.norm(G_X, p=2).item()

            if norm_G_X >= min_norm:
                if x_line_search:                    # cell 79 (published)
                    cost_old = cost.item()

                    def eval_step(step):
                        X_new = X + step * G_X / norm_G_X
                        c_N = apply_channel(th_K, m1, X_new, N_b)
                        c_M = apply_channel(th_L, m2, X_new, N_b)
                        cost_new, _, p_new, q_new = compute_cost(U, c_N, c_M)
                        return cost_new, X_new, p_new, q_new, c_N, c_M

                    success, outputs = _backtracking_line_search(
                        eval_step, cost_old, norm_G_X,
                        alpha_init, min_step, armijo_alpha, armijo_beta
                    )
                    if success:
                        # a failed line search leaves X untouched, no break
                        cost, X, p, q, channel_N, channel_M = outputs
                else:                                # cell 82 (fixed step)
                    X = X + alpha_init * (G_X / norm_G_X)
                    channel_N = apply_channel(th_K, m1, X, N_b)
                    channel_M = apply_channel(th_L, m2, X, N_b)

            X.grad = None

        eigvals, _ = th.linalg.eigh(channel_N)

        channel_N = apply_channel(th_K, m1, X, N_b)
        channel_M = apply_channel(th_L, m2, X, N_b)

        Q, U, cost, _ = alg1(
            channel_N, channel_M, N_b, r,
            tol=tol, alpha_init=alpha_init, min_step=min_step,
            min_norm=min_norm, armijo_alpha=armijo_alpha,
            armijo_beta=armijo_beta, maxiter=max_iter_U, V0='Identity',
        )

        with th.no_grad():
            cost, _, p, q = compute_cost(U, channel_N, channel_M)
            obj_history.append(cost.item())

        if t >= 1:
            rel_change = (abs(obj_history[-1] - obj_history[-2]) /
                          max(abs(obj_history[-1]), 1e-12))
            if rel_change < tol:
                break

        if eigvals[0] < tol_rho:
            break

    with th.no_grad():
        cost, Q, p, q = compute_cost(U, channel_N, channel_M)

    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()

    if return_state:
        return obj_history, p, q, Q.detach(), X.detach()
    return obj_history, p, q


# --------------------------------------------------------------------------
# Cell 109: joint sensitivity optimisation, unknown post-change
# --------------------------------------------------------------------------
def joint_optimization_unknown(N_a=N_A, N_b=N_B, m1=13, m2=None,
                               max_iter=500, seed_K=SEED_K, seed_L=SEED_L,
                               X0=None, data_dir=None,
                               tol=0, tol_rho=1e-5, alpha_init=1e-1,
                               min_step=1e-7, min_norm=1e-6,
                               armijo_alpha=1e-4, armijo_beta=0.8,
                               progress=False, return_state=False):
    """
    Maximise the sensitivity ``S(rho) = sum_i 1/lambda_i(N(rho))`` over the
    probe, unknown post-change channel.  Port of
    ``von_neumann_pgd_back_probing_unknown`` (cell 109).

    The measurement never enters the objective: driving the smallest eigenvalue
    of the pre-change output towards zero is what makes *any* post-change
    channel detectable, so the objective diverges as ``N(rho)`` becomes rank
    deficient.  The loop therefore stops as soon as
    ``min eig(N(rho)) < tol_rho``, which is what produces the plateau in
    Fig. 14c rather than a blow-up.

    Each step takes a normalised-gradient ascent direction with an Armijo
    backtracking line search (steps that would push an eigenvalue to
    ``<= 1e-12`` score zero and are rejected).

    The probe starts from the stored ``X_{N_a}_rho.npy`` (cell 109's
    ``generate_X_single(N_a, load_exist=True)``) unless ``X0`` is given -- this
    is what makes the run deterministic without a seed.

    On exit ``p``/``q`` are reported in the eigenbasis of ``N(rho)`` obtained
    from ``th.linalg.eig`` (not ``eigh``), exactly as the notebook does.

    Returns
    -------
    ``(history, p, q)``, or ``(history, p, q, Q, X)`` when
    ``return_state=True``.
    """
    if m2 is None:
        m2 = m1

    th_K, th_L, _, _ = kraus_pair(N_a, N_b, m1, m2, seed_K, seed_L)

    if X0 is None:
        X_rho = load_X_rho(N_a, data_dir=data_dir)
        X = th.tensor(X_rho, dtype=th.cfloat)
    else:
        X = X0.clone()

    obj_history = []

    with th.no_grad():
        X = X.detach()
        channel_N = apply_channel(th_K, m1, X, N_b)

    for t in tqdm(range(max_iter), disable=not progress):
        X = X.detach().requires_grad_(True)

        channel_N = apply_channel(th_K, m1, X, N_b)
        eigvals = th.linalg.eigvals(channel_N).real
        cost = th.sum(1 / eigvals)
        cost.backward()

        with th.no_grad():
            G_X = X.grad
            norm_G_X = th.norm(G_X, p=2).item()

            if norm_G_X >= min_norm:
                cost_old = cost.item()

                def eval_step(step):
                    X_new = X + step * G_X / norm_G_X
                    c_N = apply_channel(th_K, m1, X_new, N_b)
                    eigvals_new = th.linalg.eigvals(c_N).real
                    if th.any(eigvals_new <= 1e-12):
                        cost_new = th.tensor(0, device=X.device)
                    else:
                        cost_new = th.sum(1 / eigvals_new)
                    return cost_new, X_new, c_N

                success, outputs = _backtracking_line_search(
                    eval_step, cost_old, norm_G_X,
                    alpha_init, min_step, armijo_alpha, armijo_beta
                )

                if success:
                    _, X, channel_N = outputs

            X.grad = None

        eigvals, _ = th.linalg.eigh(channel_N)   # computed, then discarded

        with th.no_grad():
            # the notebook rebinds `eigvals` here, so the break test below
            # uses these values rather than the eigh ones
            eigvals = th.linalg.eigvals(channel_N).real
            cost_track = th.sum(1 / eigvals)
            obj_history.append(cost_track.item())

        if t >= 1:
            rel_change = (abs(obj_history[-1] - obj_history[-2]) /
                          max(abs(obj_history[-1]), 1e-12))
            if rel_change < tol:
                break

        if eigvals.min() < tol_rho:
            break

    with th.no_grad():
        channel_M = apply_channel(th_L, m2, X, N_b)
        _, U = th.linalg.eig(channel_N)
        _, Q, p, q = compute_cost(U, channel_N, channel_M)

    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()

    if return_state:
        return obj_history, p, q, Q.detach(), X.detach()
    return obj_history, p, q


# --------------------------------------------------------------------------
# Fig. 15 wrappers: the (p, q) pair actually fed to the CUSUM detector
# --------------------------------------------------------------------------
def _rank_min_branch(m1, N_a, N_b, m_list):
    """Rank-minimisation position of ``m1`` in the notebook's seeded sweep."""
    if m1 in m_list:
        return m_list.index(m1)
    return 0


def design_known(m1, N_a=N_A, N_b=N_B, m2=None, m_list=None,
                 max_iter=2000, max_iter_U=1000, joint_max_iter=200,
                 joint_max_iter_U=5, progress=False, **kwargs):
    """
    Measurement design for a **known** post-change channel -- the "Max-KL"
    curves of Fig. 15.

    Reproduces cell 118's branch test ``m1 < N_a + N_b - 1``:

    * ``m1 < 13`` ("Rank Min only"): seeded log-det rank minimisation
      (cell 97) followed by ``alg1`` on the rank-``R*`` support;
    * ``m1 >= 13`` ("Rank Min & Joint Opt"): the alternating joint
      optimisation of cell 84.

    The rank-minimisation branch replays ``th.manual_seed(10)`` and skips
    ``m_list.index(m1)`` draws so that a single ``m1`` reproduces the value it
    had inside the notebook's loop (default ``m_list = [3, 4, 6, 12, 13]``).

    Returns
    -------
    dict with keys ``p``, ``q``, ``history``, ``rank``, ``method``
    (``'rank_min'`` or ``'joint'``) and ``m1``.  ``history`` is the log-det
    history for the rank-min branch and the KL history for the joint branch.
    """
    if m_list is None:
        m_list = M_LIST_RANK_MIN
    if m2 is None:
        m2 = m1

    if m1 < N_a + N_b - 1:
        psi, history, rank = rank_minimization(
            N_a=N_a, N_b=N_b, m1=m1, m2=m2,
            skip_draws=_rank_min_branch(m1, N_a, N_b, m_list),
            max_iter=max_iter, progress=progress, **kwargs)
        p, q, _ = measurement_known(psi, rank, N_a=N_a, N_b=N_b, m1=m1, m2=m2,
                                    max_iter_U=max_iter_U)
        return {'p': p, 'q': q, 'history': history, 'rank': rank,
                'method': 'rank_min', 'm1': m1}

    history, p, q = joint_optimization_known(
        N_a=N_a, N_b=N_b, m1=m1, m2=m2, max_iter=joint_max_iter,
        max_iter_U=joint_max_iter_U, progress=progress, **kwargs)
    return {'p': p, 'q': q, 'history': history, 'rank': len(p),
            'method': 'joint', 'm1': m1}


def design_unknown(m1, N_a=N_A, N_b=N_B, m2=None, m_list=None,
                   max_iter=2000, joint_max_iter=500, progress=False,
                   **kwargs):
    """
    Measurement design for an **unknown** post-change channel -- the
    "Max-sens" curves of Fig. 15.

    Same branch test as :func:`design_known`:

    * ``m1 < 13``: the identical seeded rank minimisation (cell 105 reuses
      cell 97's optimisation), but the measurement is the eigenbasis of
      ``N(rho)`` and ``p, q`` are truncated to their last ``R*`` entries;
    * ``m1 >= 13``: the sensitivity joint optimisation of cell 109
      (``max_iter=500``).

    Returns the same dict as :func:`design_known`; for the joint branch
    ``history`` is the sensitivity history plotted in Fig. 14c.
    """
    if m_list is None:
        m_list = M_LIST_RANK_MIN
    if m2 is None:
        m2 = m1

    if m1 < N_a + N_b - 1:
        psi, history, rank = rank_minimization(
            N_a=N_a, N_b=N_b, m1=m1, m2=m2,
            skip_draws=_rank_min_branch(m1, N_a, N_b, m_list),
            max_iter=max_iter, progress=progress, **kwargs)
        p, q, _ = measurement_unknown(psi, rank, N_a=N_a, N_b=N_b,
                                      m1=m1, m2=m2)
        return {'p': p, 'q': q, 'history': history, 'rank': rank,
                'method': 'rank_min', 'm1': m1}

    history, p, q = joint_optimization_unknown(
        N_a=N_a, N_b=N_b, m1=m1, m2=m2, max_iter=joint_max_iter,
        progress=progress, **kwargs)
    return {'p': p, 'q': q, 'history': history, 'rank': len(p),
            'method': 'joint', 'm1': m1}


# --------------------------------------------------------------------------
# Self-test: reproduces the numbers quoted for Fig. 14
# --------------------------------------------------------------------------
def _self_test():
    ok = True

    print("Fig. 14a -- log-det rank minimisation, N_a = 8, N_b = 6")
    print(f"{'m1':>4} {'R*':>4} {'logdet[0]':>12} {'logdet[-1]':>12}")
    sweep = rank_minimization_sweep()
    ranks = []
    for m1 in M_LIST_RANK_MIN:
        _, history, rank = sweep[m1]
        ranks.append(rank)
        print(f"{m1:>4} {rank:>4} {history[0]:>12.3f} {history[-1]:>12.3f}")
    expected = [2, 3, 4, 5, 6]
    print(f"  ranks {ranks} (expected {expected})")
    ok &= ranks == expected
    ok &= -58.0 < sweep[3][1][-1] < -55.0        # m1 = 3 plateau near -57
    ok &= -21.0 < sweep[13][1][-1] < -18.0       # m1 = 13 plateau near -19.7

    print("\nFig. 14b -- joint optimisation, known post-change")
    kl = {}
    for m1 in M_LIST_JOINT:
        history, p, q = joint_optimization_known(m1=m1)
        kl[m1] = history[-1]
        print(f"  m1 = {m1:>2}: KL {history[0]:.4f} -> {history[-1]:.4f} "
              f"({len(history)} iterations), sum(p) = {p.sum():.4f}, "
              f"sum(q) = {q.sum():.4f}")
    ok &= abs(kl[13] - 1.852) < 0.02       # notebook cell 84's p, q -> 1.8519
    ok &= abs(kl[25] - 0.783) < 0.02       # notebook cell 84's p, q -> 0.7828

    print("\nFig. 14c -- joint optimisation, unknown post-change")
    sens = {}
    for m1 in M_LIST_JOINT:
        history, p, q = joint_optimization_unknown(m1=m1)
        sens[m1] = history[-1]
        print(f"  m1 = {m1:>2}: sensitivity {history[0]:.2f} -> "
              f"{history[-1]:.2f} ({len(history)} iterations)")
    ok &= abs(sens[13] - 1080.0) / 1080.0 < 0.02
    ok &= abs(sens[25] - 112.0) / 112.0 < 0.05

    print("\nFig. 15 designs (m1 = 3, 6, 13)")
    for m1 in (3, 6, 13):
        dk = design_known(m1)
        du = design_unknown(m1)
        print(f"  m1 = {m1:>2} [{dk['method']}] R* = {dk['rank']}")
        print(f"      Max-KL   p = {np.round(dk['p'], 5)}")
        print(f"               q = {np.round(dk['q'], 5)}  sum(q) = {dk['q'].sum():.4f}")
        print(f"      Max-sens p = {np.round(du['p'], 5)}")
        print(f"               q = {np.round(du['q'], 5)}  sum(q) = {du['q'].sum():.4f}")
        ok &= abs(dk['p'].sum() - 1.0) < 1e-3

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(_self_test())
