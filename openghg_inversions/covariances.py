from typing import Optional

import numpy as np
import scipy.linalg as sla
import xarray as xr


def prior_covariance_matrix_2d(
    flux: xr.DataArray,
    grid_cell_error: float = 0.5,
    dims: list[str] = ["lat", "lon"],
    off_diagonal: int = 0,
    corner_weight: float = 0.75,
    total_uncertainty: Optional[float] = None,
):
    """
    Create prior covariance matrix for 2D gridded data.

    For flattened flux vector (f1, f2, f3, ...), with grid cell error 0.5 and
    off_diagonal = 0 gives covariance matrix C = diag(0.25 * f1^2, 0.25 * f2^2, ...)

    If off_diagonal = 1, then the term 0.125 * fi * fj is added if grid cells i and j (in flattened coordinates)
    share an edge, and the term 0.75 * 0.125 * fi * fj is added if grid cells i and j share a
    corner. (The .75 comes from (.5)^sqrt(2) approx 0.75 * 0.5.)

    It's possible to continue this pattern, where the weight of off-diagonal terms is decreased according to the distance
    between centers of grid cells, but using Euclidean distance between centers (instead of taxi cab distance) is hard to
    implement.
    """
    if len(dims) != 2:
        raise ValueError("Can only make covariance matrix for 2D data.")

    if off_diagonal > 1:
        raise ValueError(f"`off_diagonal` can only be 0 or 1; {off_diagonal} given.")

    m = len(flux[dims[0]])
    n = len(flux[dims[1]])

    dim_tup = tuple(dims)
    v = flux.transpose(*dim_tup).stack(loc=dim_tup)  # row major flattened matrix

    # incidence matrix for cells distance 1 apart
    # is block diagonal with a n x n block for each row
    # the block has 1's on the 1st off-diagonal
    r1 = sla.toeplitz([0, 1] + [0] * (n - 2))
    R1 = sla.block_diag(*tuple([r1] * m))

    # incidence matrix for cells distance 1 apart
    # is block diagonal with a n x n block for each row
    # the block has 1's on the 1st off-diagonal
    C1 = sla.toeplitz([0] * n + [1] + [0] * (m * n - n - 1))

    # we need R2 and C2 to calculate the incidence matrix for
    # grids whose corners touch
    r2 = sla.toeplitz([0, 0, 1] + [0] * (n - 3))
    R2 = sla.block_diag(*tuple([r2] * m))
    C2 = sla.toeplitz([0] * 2 * n + [1] + [0] * (m * n - 2 * n - 1))
    I = np.eye(m * n)

    corners = ((R1 + C1) @ (R1 + C1) > 0).astype(int) - I - R2 - C2

    b = grid_cell_error
    C = (
        v.values.reshape((-1, 1))
        @ v.values.reshape((1, -1))
        * b**2
        * (I + b * R1 + b * R2 + corner_weight * b * corners)
    )

    return C
