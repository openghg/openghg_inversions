from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as ssp
import xarray as xr


def prior_covariance_matrix_2d(
    flux: xr.DataArray,
    grid_cell_error: float = 0.5,
    dims: list[str] = ["lat", "lon"],
    max_dist: int = 0,
    total_uncertainty: Optional[float] = None,
):
    """
    Create prior covariance matrix for 2D gridded data.

    For flattened flux vector (f1, f2, f3, ...), with grid cell error 0.5 and
    max_dist = 0 gives covariance matrix C = diag(0.25 * f1^2, 0.25 * f2^2, ...)

    If max_dist = 1, then the term 0.125 * fi * fj is added if grid cells i and j (in flattened coordinates)
    are taxi cab distance one apart.

    If max_dist = 2, then 0.0625 * fi * fj is added for grid cells i and j of taxi cab distance 2 apart.

    Taxi cab (i.e. L1) distance is used for convenience. In 2D, it differs from Euclidean distance by a multiplicative
    factor between 1 and sqrt(2).

    Args:
        flux: DataArray with 2D flux or boundary condition
        grid_cell_error: amount of uncertainty to give to each grid cell. Default = 0.5
        dims: list of dimensions in flux DataArray. Default = ["lat", "lon"]. Use ["lat", "height"]
            or ["lon", "height"] for boundary conditions.
        max_dist: maximum taxicab distance between grid cells before covariance set to 0.
        total_uncertainty: rescale covariance matrix so that the standard deviation of the sum of
            the flux is the specified value times the total flux. For instance `total_uncertainty=0.8`
            specifies 80% uncertainty on the total flux.

    Returns:
        Sparse covariance matrix.

    WARNING: for the EUROPE domain, the covariance matrix will occupy about 98 gigabytes if
    converted to a dense array. Reduce dimensions using sparse routines first.
    """
    if len(dims) != 2:
        raise ValueError("Can only make covariance matrix for 2D data.")

    m = len(flux[dims[0]])
    n = len(flux[dims[1]])
    size = m * n

    dim_tup = tuple(dims)
    v0 = flux.transpose(*dim_tup).stack(loc=dim_tup).values  # row major flattened matrix
    v = ssp.diags(v0, shape=(size, size))

    if max_dist == 0:
        return grid_cell_error**2 * v**2

    # incidence matrix for cells distance 1 apart in first
    # dimension is block diagonal with a n x n block for each row
    # the block has 1's on the 1st off-diagonal
    data = [np.ones(n)] * 2
    r1 = ssp.spdiags(data, [-1, 1], n, n)
    R1 = ssp.block_diag(tuple([r1] * m))

    # incidence matrix for cells distance 1 apart in second
    # dimension has 1's on the n-th off-diagonal
    data = [np.ones(size)] * 2
    C1 = ssp.spdiags(data, [-n, n], m=size, n=size)

    L0 = ssp.eye(size)
    L1 = R1 + C1

    levels = [L0, L1]

    # build incidence matrix for each taxicab distance
    while len(levels) <= max_dist:
        L = (levels[1] @ levels[-1] > 0).astype(int) - levels[-2]
        levels.append(L)

    # combine incidence matrices for each distance, with decaying weights
    C0 = sum(grid_cell_error ** (2 + i) * L for i, L in enumerate(levels))

    # add terms f_i * f_j, where (f_1, f_2, ...) is the flattened flux vector
    C = v @ C0 @ v

    # rescale if total uncertainty specified
    if total_uncertainty:
        C = C * (total_uncertainty * np.sum(v0)) ** 2 / np.sum(C)

    return C
