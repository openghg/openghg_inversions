import pandas as pd
import sparse
import xarray as xr

def test_transpose():
    dums = pd.get_dummies([0]*3 + [1]*4 + [2]*5, dtype=int, sparse=True)
    sparse_dums = sparse.COO.from_scipy_sparse(dums.sparse.to_coo())
    da = xr.DataArray(sparse_dums)
    da.transpose()
