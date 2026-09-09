"""
Quantum state generation utilities.

Functions to generate random positive-definite density matrices (quantum states)
with various rank constraints and parameters.
"""

import numpy as np
import torch as th


def generate_mixed_state(n):
    """
    Generate a random mixed quantum state (density matrix).

    Creates an n×n Hermitian positive-definite matrix with trace 1
    by generating random eigenvalues and a random unitary.

    Parameters:
    -----------
    n : int
        Dimension of the quantum state

    Returns:
    --------
    rho : torch tensor
        n×n Hermitian positive-definite density matrix
    """
    U = th.linalg.qr(th.randn((n, n), dtype=th.cfloat))[0]
    p = th.rand(n,).to(th.cfloat)
    p = p/p.sum()
    return U@th.diag_embed(p)@U.T.conj()


def generate_pd_matrix(X, n):
    """
    Generate a positive-definite matrix with specified dimension.

    Uses QR decomposition of a random matrix X to construct orthonormal basis,
    then forms a PD matrix with random eigenvalues.

    Parameters:
    -----------
    X : array-like
        Seed matrix for QR decomposition
    n : int
        Dimension of output matrix

    Returns:
    --------
    rho : ndarray
        n×n Hermitian positive-definite matrix (as numpy array)
    """
    X = th.tensor(X, dtype=th.cfloat)
    U = th.linalg.qr(X)[0]
    p = th.rand(n,).to(th.cfloat)
    p = p/p.sum()

    return (U@th.diag_embed(p)@U.T.conj()).detach().numpy()


def generate_pd_matrix_rank(X, n, r):
    """
    Generate a positive-definite matrix with specified rank.

    Creates an n×n matrix where the rank (number of nonzero eigenvalues)
    is exactly r. The nonzero eigenvalues sum to 1 (trace normalization).

    Parameters:
    -----------
    X : array-like
        Seed matrix for QR decomposition
    n : int
        Dimension of output matrix
    r : int
        Desired rank (number of nonzero eigenvalues)

    Returns:
    --------
    rho : ndarray
        n×n Hermitian rank-r positive-definite matrix (as numpy array)
    """
    X = th.tensor(X, dtype=th.cfloat)
    U = th.linalg.qr(X)[0]

    # --- build eigenvalues p ---
    p = th.zeros(n, dtype=th.float32)          # n-r zeros already present
    p[:r] = th.rand(r)                         # r positive numbers
    p[:r] /= p[:r].sum()                       # normalize to trace = 1
    p = p[th.randperm(n)]                      # randomise their positions
    p = p.to(th.cfloat)                        # match dtype used above
    # --------------------------------

    return (U @ th.diag_embed(p) @ U.T.conj()).detach().numpy()


def generate_X_single(n, load_exist):
    """
    Generate or load random seed matrices for state generation.

    Parameters:
    -----------
    n : int
        Dimension of seed matrices
    load_exist : bool
        If True, load from disk; if False, generate new and save

    Returns:
    --------
    X_rho : ndarray
        Seed matrix for rho generation
    X_sigma : ndarray
        Seed matrix for sigma generation
    """
    if load_exist == True:
        X_rho = np.load(f'data/X/X_{n}_rho.npy')
        X_sigma = np.load(f'data/X/X_{n}_sigma.npy')
    else:
        X_rho = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        X_sigma = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        np.save(f'data/X/X_{n}_rho.npy', X_rho)
        np.save(f'data/X/X_{n}_sigma.npy', X_sigma)
    return X_rho, X_sigma


def generate_sigma(X_sigma, N, load_exist):
    """
    Generate or load a full-rank sigma (reference) state.

    Parameters:
    -----------
    X_sigma : ndarray
        Seed matrix for sigma generation
    N : int
        Dimension of quantum state
    load_exist : bool
        If True, load from disk; if False, generate new and save

    Returns:
    --------
    sigma : ndarray
        N×N reference quantum state (full rank)
    """
    if load_exist == True:
        sigma = np.load(f'data/adap_grad_jeremy/sigma_{N}.npy')
    else:
        sigma = generate_pd_matrix(X_sigma, N)
        np.save(f'data/adap_grad_jeremy/sigma_{N}.npy', sigma)
    return sigma


def generate_rho_set_rank(X_rho, N, R, set_num, load_exist):
    """
    Generate or load a set of rank-R quantum states.

    Creates set_num different rank-R density matrices, useful for
    ensemble averaging in anomaly detection experiments.

    Parameters:
    -----------
    X_rho : ndarray
        Seed matrix for rho generation
    N : int
        Dimension of quantum states
    R : int
        Rank (number of nonzero eigenvalues)
    set_num : int
        Number of states to generate
    load_exist : bool
        If True, load from disk; if False, generate new and save

    Returns:
    --------
    rho_set : ndarray
        (set_num, N, N) array of rank-R quantum states
    """
    if load_exist == True:
        rho_set = np.load(f'data/adap_grad_jeremy/rho_{N}_rank_{R}_set_{set_num}.npy')
    else:
        rho_set = np.zeros((set_num, N, N), dtype=complex)
        for i in range(set_num):
            rho = generate_pd_matrix_rank(X_rho, N, R)
            rho_set[i,:,:] = rho
        np.save(f'data/adap_grad_jeremy/rho_{N}_rank_{R}_set_{set_num}.npy', rho_set)
    return rho_set
