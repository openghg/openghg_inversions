Grouped basis and state metadata
================================

Basis functions can be constructed separately for different parts of the full
domain—for example, a detailed inner region and a coarser outer region—and
then combined into a single basis array.  The combined array remembers which
group and partition each basis region came from, so states can still be
selected and analysed by group after they have been assembled.

``BasisLayout`` provides this assembly step.  It combines disjoint,
partition-local label maps into the one flat basis consumed by a bucket
operator.  Alongside that flat map it produces the stable state metadata
``basis_group``, ``basis_partition``, and ``region_in_partition``.

This is currently a lower-level Python construction API.  Import it from
``openghg_inversions.basis.layout`` when an application has already generated
or loaded the partition label arrays.  The implementation is eager and
in-memory, and it does not infer a remainder partition.

The cells on this page are executed when the documentation is built.  You can
:jupyter-download-notebook:`download them as a Jupyter notebook <grouped_basis_layout>`
to modify and run locally.

.. jupyter-kernel:: python3
   :id: grouped_basis_layout

Build an inner/outer layout
---------------------------

Each partition uses positive integers for its own local region labels.  Zero,
negative, and NaN values mean that a cell is outside that partition.  The two
arrays below are disjoint, and the explicitly named remainder covers every
cell outside the inner partition.

.. jupyter-execute::

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
   result.basis_flat.values

The combined map has raw ``basis_label`` values 1, 2, and 3.  The metadata is
keyed by those raw labels before an operator applies its final state-label
policy.  Selecting only the small state-aligned variables keeps the spatial
coordinates out of the display.

.. jupyter-execute::

   result.state_metadata[["basis_group", "basis_partition", "region_in_partition"]]

The exact byte counts in an xarray display can vary by platform; the important
contract is the coordinate values and their order.

Remainders are explicit
-----------------------

Leaving out the remainder is an error rather than a request for
``BasisLayout`` to invent one.

.. jupyter-execute::

   try:
       BasisLayout(partitions=(inner,), state_dim="region").to_flat_basis()
   except ValueError as error:
       uncovered_error = str(error)
       print(uncovered_error)

Attach the metadata to an operator
----------------------------------

Pass both outputs to ``BasisFunctions.from_flat_basis``.  The
``state_metadata`` option is forwarded to ``BucketBasisOperator``.

.. jupyter-execute::

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
   print("Raw basis labels:", result.state_metadata.basis_label.values)
   print("Operator state labels:", matrix.region.values)
   matrix.coords.to_dataset()[
       ["basis_group", "basis_partition", "region_in_partition"]
   ].compute()

Here the raw basis labels are ``[1, 2, 3]``, while ``region_labels="range0"``
makes the final operator state coordinate ``[0, 1, 2]``.  Metadata follows the
states through that relabelling.

Select states by group
----------------------

Because the grouping fields are coordinates on the state dimension, normal
xarray operations can select an inner or outer subset.

.. jupyter-execute::

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
   print("Inner states:", inner_state.region.values)
   print("Outer states:", outer_state.region.values)
   print("Outer partition:", outer_state.basis_partition.item())

Round-trip preservation
-----------------------

``BasisFunctions`` preserves the state metadata in its versioned DataTree
representation.  ``save`` and ``load`` use the same representation for NetCDF
or Zarr artifacts.

.. jupyter-execute::

   restored = BasisFunctions.from_datatree(basis_functions.to_datatree())
   restored_matrix = restored.operator.basis_matrix
   {
       "region": restored_matrix.region.equals(matrix.region),
       "basis_group": restored_matrix.basis_group.equals(matrix.basis_group),
       "region_in_partition": restored_matrix.region_in_partition.equals(
           matrix.region_in_partition
       ),
   }

Stability boundary
------------------

This page documents the implemented layout artifact, state metadata, operator
attachment, and serialization contract.  ``BasisLayout`` remains an eager,
in-memory, lower-level Python API and does not infer a remainder.

Separate group-specific weight or sensitivity construction, grouped priors,
public RHIME configuration, and a full extension API are outside this contract.
Those broader changes remain tracked by `GitHub issue #456
<https://github.com/openghg/openghg_inversions/issues/456>`_ and `Linear
OPE-25
<https://linear.app/openghg-inversions/issue/OPE-25/pressure-test-grouped-innerouter-state-layouts>`_.
