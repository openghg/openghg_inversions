from typing import Optional

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla


def gen_least_sq(
    y: np.array,
    A: np.array,
    W: Optional[np.array] = None,
    B: Optional[np.array] = None,
    run_checks: bool = False,
):
    """
    Paige's algorithm [1, 2] for the least-squares solution to

    y = Ax + w,

    where w ~ N(0, W).

    Args:
        y: m-dimensional vector
        A: m x n matrix
        W: m x m symmetric positive definite matrix
        B: m x m lower triangular matrix -- the Cholesky
           decomposition of W.

    One of W or B must be provided.

    Returns:


    Notes:

    The least squares solution for x is the minimizer of

    J(x) = (y - Ax)^T * W^-1 * (y - Ax)

    The solution via the normal equations is:

    x = (A^T W^-1 A)^-1 * A^T * W^-1 * y

    and the covariance matrix for x is

    C = (A^T W^-1 A)^-1

    If W is not diagonal then solving via the normal equations
    might have poor numerical performance.

    Paige's algorithm starts by using the Cholesky decomposition W = B*B^T
    to rewrite the problem as finding the values of x and v that minimize
    v^T * v subject to y = Ax + Bv. (So v ~ N(0, I).)

    The algorithm solves for v and x without inverting B, which provides a
    numerically stable solution. (The naive method of replacing y and A by
    B^-1 * y and B^-1 * A might be problematic if B is ill-conditioned.)

    NOTE: we assume that A is full rank. This is not strictly necessary, but
    the algorithm is more complicated if A is singular. (It is also problematic
    if A is ill-conditioned, so this should be improved before running this
    algorithm.)

    [1] C. C. Paige, Computer Solution and Perturbation Analysis of Generalized
    Linear Least Squares Problems, Mathematics of Computation, Jan 1979

    [2] Golub and Van Loan, Matrix Computations, 4th ed., section 6.1.2
    """
    # step 1: get Cholesky decomposition of W
    if B is None:
        if W is None:
            raise ValueError("One of the arguments W or B must be provided.")
        B = nla.cholesky(W)

    # step 2: QR decomposition of A
    Q, R = nla.qr(A, mode="complete")

    # get R_1 = non-zero rows of R, and decompose columns of Q accordingly
    mask = np.ma.any(R != 0, axis=1)
    R_1 = R[mask]
    Q_1 = Q[:, mask]
    Q_2 = Q[:, ~mask]

    # check decomposition
    if run_checks:
        assert np.allclose(Q_1.T @ A, R_1)
        assert np.allclose(Q_2.T @ A, np.zeros_like(Q_2.T @ A))

    # step 3: decompose Q_2.T @ B into (triangular) x (orthogonal)
    # specifically, Q_2.T @ B @ Z = S, with Z orthogonal, and S upper triangular
    _, k = Q_2.shape
    _, p = B.shape
    flip_k = sla.hankel([0] * (k - 1) + [1])
    flip_p = sla.hankel([0] * (p - 1) + [1])

    # QR decomp of (Q_2.T @ B).T with columns reversed
    Z_temp, S_temp = nla.qr(B.T @ Q_2 @ flip_k, mode="complete")

    # The result is:
    #
    # Z_temp.T @ (B.T @ Q_2 @ flip_k) = S_temp
    #
    # hence
    #
    # flip_k @ Q_2.T @ B @ Z_temp = S_temp.T
    #
    # hence
    #
    # Q_2.T @ B @ Z_temp @ flip_p = flip_k @ S_temp.T @ flip_p
    #
    # Thus Q_2.T @ B @ Z = S requires:
    Z = Z_temp @ flip_p
    S = flip_k @ S_temp.T @ flip_p

    if run_checks:
        assert np.allclose(Q_2.T @ B @ Z, S)

    # get S_2 = non-zero columns of S, and decompose columns of Z accordingly
    mask = np.ma.any(S != 0, axis=0)
    S_2 = S[:, mask]
    Z_1 = Z[:, ~mask]
    Z_2 = Z[:, mask]

    if run_checks:
        assert np.allclose(Q_2.T @ B, S_2 @ Z_2.T)

    # Assuming that R_1 and S_2 are invertible (which is true if A has full column rank)
    # we can solve for the solution x-hat
    G = Q_1.T @ (np.eye(p) - B @ Z_2 @ nla.inv(S_2) @ Q_2.T)
    w = G @ y
    R_1_inv = nla.inv(R_1)
    x_hat = R_1_inv @ w

    # NOTE: rather than invert S_2 and R_1, we might want to use
    # a (possibly truncated) pseudo-inverse

    # Covariance matrix C = H @ H^T for x_hat
    # Note: this computation isn't proved to be numerically stable...
    H = R_1_inv @ Q_1.T @ B @ Z_1
    C = H @ H.T

    # temp return for checking
    return x_hat, H, C


def tikhonov_reg_solver(y: np.array, A: np.array, W: np.array, B: np.array):
    """
    Find x minimizing

    J(x) = (y - Ax)^T @ W^-1 @ (y - Ax) + x^T @ B^-1 @ x

    Args:
        y: observation vector
        A: model/design matrix
        W: covariance for y
        B: inverse of Tikhonov matrix (covariance for regularisation term)
    """
    W_stack = sla.block_diag(W, B)

    n = A.shape[1]
    A_stack = np.vstack([A, np.eye(n)])
    y_stack = np.concatenate([y, np.zeros(n)])

    return gen_least_sq(y_stack, A_stack, W=W_stack)


def bayesian_least_squres(
    y: np.array, A: np.array, W: np.array, x_prior: np.array, B: np.array
):
    """
    Find maximum a posteriori estimate for x for model with prior N(x_prior, B)
    and likelihood y ~ N(Ax, W).

    This reduces to the Tikhonov problem by setting z = x - x_prior and
    w = y - A @ x_prior, and solving for z.
    """
    w = y - A @ x_prior
    x_hat, H, C = tikhonov_reg_solver(w, A, W, B)
    x_map = x_hat + x_prior

    return x_map, H, C
