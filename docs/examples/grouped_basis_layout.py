# start-grid
import numpy as np
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.layout import BasisLayout, BasisPartition


coords = {"lat": [51.0, 52.0], "lon": [-2.0, -1.0]}
inner_labels = xr.DataArray(
    [[1, 2], [0, 0]],
    dims=("lat", "lon"),
    coords=coords,
    name="inner_labels",
)
remainder_labels = xr.DataArray(
    [[0, 0], [1, 1]],
    dims=("lat", "lon"),
    coords=coords,
    name="remainder_labels",
)

inner = BasisPartition(name="generated_inner", labels=inner_labels, group="inner")
remainder = BasisPartition(
    name="explicit_remainder",
    labels=remainder_labels,
    group="outer",
)
layout = BasisLayout(partitions=(inner, remainder), state_dim="region")
result = layout.to_flat_basis()
# end-grid

# start-uncovered
try:
    BasisLayout(partitions=(inner,), state_dim="region").to_flat_basis()
except ValueError as error:
    uncovered_error = str(error)
# end-uncovered

# start-operator
flux = xr.DataArray(
    np.ones((2, 2, 1)),
    dims=("lat", "lon", "time"),
    coords={**coords, "time": ["2020-01-01"]},
    name="flux",
)
basis_functions = BasisFunctions.from_flat_basis(
    result.basis_flat,
    flux,
    region_labels="range0",
    operator_kwargs={
        "state_dim": "region",
        "state_metadata": result.state_metadata,
    },
)
matrix = basis_functions.operator.basis_matrix
# end-operator

# start-selection
state = xr.DataArray(
    [0.9, 1.1, 1.0],
    dims="region",
    coords={
        "region": matrix.region.values,
        "basis_group": ("region", matrix.basis_group.values),
        "basis_partition": ("region", matrix.basis_partition.values),
        "region_in_partition": ("region", matrix.region_in_partition.values),
    },
    name="scaling",
)
inner_state = state.where(state.basis_group == "inner", drop=True)
outer_state = state.where(state.basis_group == "outer", drop=True)
# end-selection

# start-roundtrip
restored = BasisFunctions.from_datatree(basis_functions.to_datatree())
restored_matrix = restored.operator.basis_matrix
# end-roundtrip
