"""
Mathematical utility functions for quantum anomaly detection.

Includes polar decomposition, Stiefel manifold projections, KL divergence,
POVM operations, and measurement distribution functions.
"""

import numpy as np
import torch as th


def polar(A, mode):
    """
    Compute polar decomposition of a matrix.

    The polar decomposition factorizes A = U * P where U is unitary
    and P is positive-definite Hermitian.

    Parameters:
    -----------
    A : torch tensor
        Input matrix
    mode : str
        'exact': Use SVD-based exact computation
        'approx': Use iterative QR-based approximation

    Returns:
    --------
    U : torch tensor
        Unitary part of polar decomposition
    """
    if mode == 'exact':
        U, S, Vh = th.linalg.svd(A, full_matrices=False)
        return U@Vh

    elif mode == 'approx':
        Q, X = th.linalg.qr(A)

        for _ in range(20):
            X = 0.5*(X + th.linalg.inv(X).T.conj())

        return Q@X


def proj_stiefel(S, mode):
    """
    Project onto the Stiefel manifold (orthonormal frames).

    The Stiefel manifold St(n,p) consists of n×p matrices with
    orthonormal columns.

    Parameters:
    -----------
    S : list of torch tensors
        Collection of matrices to concatenate and project
    mode : str
        'exact' or 'approx' - passed to polar decomposition

    Returns:
    --------
    Stilde : torch tensor
        Orthonormal projection of concatenated S matrices
    """
    X = th.cat([*S], dim=0)
    Stilde = polar(X, mode)
    return Stilde


def kl_divergence(p, q):
    """
    Compute Kullback-Leibler divergence using PyTorch.

    KL(p||q) = sum_i p_i * (log(p_i) - log(q_i))

    Parameters:
    -----------
    p : torch tensor
        First probability distribution
    q : torch tensor
        Second probability distribution

    Returns:
    --------
    div : torch tensor
        KL divergence value (scalar)
    """
    return p @ (th.log(p) - th.log(q))


def kl_divergence_np(p, q):
    """
    Compute Kullback-Leibler divergence using NumPy.

    KL(p||q) = sum_i p_i * (log(p_i) - log(q_i))

    Parameters:
    -----------
    p : ndarray
        First probability distribution
    q : ndarray
        Second probability distribution

    Returns:
    --------
    div : float
        KL divergence value
    """
    return p @ (np.log(p) - np.log(q))


def check_complex(p, q, tol):
    """
    Verify that probability distributions are real-valued.

    Checks that imaginary parts of p and q are below tolerance.
    If they are sufficiently small, strips them and returns real parts.

    Parameters:
    -----------
    p : ndarray
        First distribution (possibly complex)
    q : ndarray
        Second distribution (possibly complex)
    tol : float
        Tolerance threshold for imaginary parts

    Returns:
    --------
    p_real : ndarray
        Real part of p
    q_real : ndarray
        Real part of q

    Raises:
    -------
    Exception
        If imaginary parts exceed tolerance
    """
    p = np.where(np.abs(p.imag) < tol, p.real, p)
    q = np.where(np.abs(q.imag) < tol, q.real, q)

    if np.allclose(np.abs(p.imag), 0, atol=tol) == False:
        if np.allclose(np.abs(q.imag,), 0, atol=tol) == False:
            raise Exception(f"Both p & q dist have imaginary part bigger than {tol}")
        else:
            raise Exception(f"p dist has imaginary part bigger than {tol}")
    elif np.allclose(np.abs(q.imag), 0, atol=tol) == False:
        raise Exception(f"q dist has imaginary part bigger than {tol}")

    return p.real, q.real


def povm_positive_eigen(rho, eps):
    """
    Construct POVM from positive eigenspaces of a density matrix.

    Creates a Positive-Operator-Valued-Measure (POVM) from the eigenspaces
    of rho corresponding to eigenvalues above eps. Each POVM element is
    an outer product of an eigenvector.

    Parameters:
    -----------
    rho : ndarray
        Input density matrix (Hermitian, positive-definite)
    eps : float
        Eigenvalue threshold for support

    Returns:
    --------
    Q : ndarray
        (k, n, n) array where k is number of eigenvalues > eps
        Each Q[i] is a rank-1 projector |psi_i><psi_i|
    """
    evals, evecs = np.linalg.eigh(rho)
    support_mask = evals.real > eps
    Psi = evecs[:, support_mask]
    Q = np.stack([np.outer(psi, psi.conj()) for psi in Psi.T])
    return Q


def distribution_computational_basis_measurement(n, rho, sigma):
    """
    Compute measurement outcome distributions in computational basis.

    For a given dimension n, computes the probability distributions
    for measuring each computational basis state |i><i| on rho and sigma.

    Parameters:
    -----------
    n : int
        Dimension of quantum system
    rho : ndarray
        Pre-change density matrix
    sigma : ndarray
        Post-change density matrix

    Returns:
    --------
    p_dist : ndarray
        Measurement probabilities on rho
    q_dist : ndarray
        Measurement probabilities on sigma
    """
    I = np.matrix(np.eye(n))

    p_dist = []
    q_dist = []
    for i in range(n):
        Q_i = I[:,i].dot(I[i,:])
        p_dist.append(np.trace(Q_i.dot(rho)))
        q_dist.append(np.trace(Q_i.dot(sigma)))

    p_dist, q_dist = check_complex(p=np.array(p_dist), q=np.array(q_dist), tol=1e-8)

    return p_dist, q_dist


def depolarize(rho, nu):
    """
    Depolarized state (1 - nu) * rho + nu * I / N.

    Parameters:
    -----------
    rho : ndarray
        N x N density matrix
    nu : float
        Depolarizing noise level in [0, 1]

    Returns:
    --------
    rho_hat : ndarray
        Depolarized density matrix (trace one)
    """
    N = rho.shape[0]
    return (1 - nu) * rho + nu * np.eye(N) / N


def measurement_distribution(Q, state, append_null=True):
    """
    Outcome distribution tr(Q_i state) of a POVM on a state.

    Parameters:
    -----------
    Q : ndarray
        (k, N, N) stack of POVM elements (e.g. from povm_positive_eigen)
    state : ndarray
        N x N density matrix
    append_null : bool, optional
        If True (default) and k < N, the remaining mass 1 - sum_i tr(Q_i state),
        i.e. the probability of the null-space outcome (outcome 0 in the paper;
        stored LAST to match the convention of the CUSUM routines), is appended
        so that the returned vector sums to one.

    Returns:
    --------
    p : ndarray
        Length-k (or k + 1) real probability vector
    """
    p = np.einsum('kij,ji->k', Q, state).real
    if append_null and Q.shape[0] < state.shape[0]:
        p = np.append(p, max(0.0, 1.0 - p.sum()))
    return p
