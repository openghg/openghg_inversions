Grouped basis and state metadata
================================

Basis functions can be constructed separately for different parts of the full
domain—for example, many small basis regions in one area and a few larger basis
regions in the surrounding area—and then combined into a single basis array.
Here ``inner`` and ``outer`` are ordinary basis-group labels. They describe the
number and spatial extent of inferred basis regions, not nested atmospheric
model domains or grid resolution such as a 6 km inner transport domain. The
combined array remembers which group and partition each basis region came from,
so states can still be selected and analysed by group after assembly.

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

Build a grouped basis layout
----------------------------

Each partition uses positive integers for its own local region labels. Zero,
negative, and NaN values mean that a cell is outside that partition. In this
small example, one partition contains two single-cell basis regions and the
surrounding remainder contains one larger two-cell basis region. The two arrays
are disjoint, and the explicitly named remainder covers every other cell.

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
xarray operations can select a subset. The example retains ``inner`` and
``outer`` as application-chosen group names; they do not imply separate model
domains.

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

.. _grouped-basis-label-sources:

Where the labels come from
--------------------------

State labels are written when the basis is built, because that is where the
split into groups is decided. Nothing later in preparation adds them.

``BasisLayout``
   The Python construction shown above. Each partition names its own group.

Fixed outer regions
   ``fix_outer_regions = true`` in ``[prepared_inputs.basis]`` (CO2), or
   ``fix_basis_outer_regions`` in a standard RHIME configuration, fits a
   generated basis inside the inner region of an outer-region map and keeps
   every other map region as one fixed region.
   :func:`~openghg_inversions.basis.fixed_outer_regions_basis_from_data`
   builds this as a ``BasisLayout`` with two partitions, the fixed outer map
   and the generated inner labels, and
   :func:`~openghg_inversions.basis.prepare_basis` passes the layout's
   metadata to the operator. Basis labels are numbered outer regions first,
   then inner regions, so the label minus one is the state's position:

   .. list-table::
      :header-rows: 1

      * - Coordinate
        - Outer states
        - Inner states
      * - ``basis_group``
        - ``"outer"``
        - ``"inner"``
      * - ``basis_partition``
        - ``"fixed_outer"``
        - ``"inner"``
      * - ``region_in_partition``
        - outer-region map label plus one
        - generated inner label

   A basis saved with ``BasisFunctions.save`` keeps these labels, so a run
   that reuses it through ``basis.path`` gets them too.

Per-source bases
   With per-source native priors the retained state index has the levels
   ``source`` and ``region_in_source``. The ``source`` level is the state's
   source label. A generated basis, with or without fixed outer regions, is
   repeated for every source together with its labels, so each source has
   its own ``outer`` and ``inner`` states.
   ``MultiSourceBucketBasisOperator`` takes per-source ``state_metadata`` for
   this and saves it with the basis.

A generated basis without fixed outer regions, and a basis read from a
partition file (``partition_path``), have no groups and get no
``basis_group`` coordinate. A default label would add coordinates to every
run's outputs without selecting anything, and for a partition file it could
be wrong: the file does not say which of its regions are outer. To label a
partition, build it with ``BasisLayout`` and save the basis.

How the labels reach the prepared prior
---------------------------------------

The labels are coordinates on the state axis of the basis matrix ``U``.
Everything derived from that axis keeps them:

1. The projected sensitivity ``H`` has the basis state axis.
2. In the coherent reduction, the bucket restriction (``restriction =
   "bucket"``, the default) is ``Pi_U = (U.T B^-1 U)^-1 U.T B^-1``. Its state
   axis is ``U``'s, labels included, so the retained mean ``Pi m``, retained
   covariance and reduced ``H`` carry them. The absolute-flux-weighted
   restriction (per-source priors) copies the labels of the states it
   retains.
3. Prepared CO2 inputs hold these as ``alpha_prior_mean``,
   ``alpha_prior_covariance`` and ``H``, and save and reload them with the
   prepared inputs. The affine flux map saved beside them does not store
   the labels; it takes them from the prepared inputs it is bound to.

Output products are not affected: PARIS flux and concentration files are
the same with and without state labels. ``basic`` and ``legacy`` outputs
include the flat basis map, whose values follow the numbering above.

What selects by the labels
--------------------------

- :class:`~openghg_inversions.models.StateActivity` with ``fixed_groups``
  fixes the states of the named ``basis_group`` values.
- The outer-flux option selects its outer states by ``basis_group``
  (``outer_groups``) and the outer states held at their prior by ``source``
  (``fixed_sources``). It reads both from the prepared ``alpha_prior_mean``,
  so a preparation built with fixed outer regions can use
  ``outer_groups = ["outer"]`` directly. :doc:`atmospheric_baseline` explains
  the option, and the outer-flux section of :doc:`configured_co2_priors`
  configures it.

Stability boundary
------------------

This page documents the implemented layout artifact, state metadata, operator
attachment, and serialization contract.  ``BasisLayout`` remains an eager,
in-memory, lower-level Python API and does not infer a remainder.

Fixed outer regions are the only configured route that labels groups.
Separate group-specific weight or sensitivity construction, grouped priors,
configured ``BasisLayout`` partitions and a full extension API are outside
this contract.
