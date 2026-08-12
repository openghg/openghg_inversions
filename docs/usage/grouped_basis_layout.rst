Grouped basis and state metadata
================================

``BasisLayout`` assembles disjoint, partition-local label maps into the one
flat basis consumed by a bucket operator.  Alongside that flat map it produces
stable semantic metadata for every state: ``basis_group``,
``basis_partition``, and ``region_in_partition``.

This is currently a lower-level Python construction API.  Import it from
``openghg_inversions.basis.layout`` when an application has already generated
or loaded the partition label arrays.  The implementation is eager and
in-memory, and it does not infer a remainder partition.

Build an inner/outer layout
---------------------------

Each partition uses positive integers for its own local region labels.  Zero,
negative, and NaN values mean that a cell is outside that partition.  The two
arrays below are disjoint, and the explicitly named remainder covers every
cell outside the inner partition.

.. literalinclude:: ../examples/grouped_basis_layout.py
   :language: python
   :start-after: # start-grid
   :end-before: # end-grid

The combined map has raw ``basis_label`` values 1, 2, and 3::

   >>> result.basis_flat.values
   array([[1, 2],
          [3, 3]])

The metadata is keyed by those raw labels before an operator applies its final
state-label policy.  Selecting only the small state-aligned variables keeps
the spatial coordinates out of the display::

   >>> result.state_metadata[["basis_group", "basis_partition", "region_in_partition"]]
   <xarray.Dataset> Size: 96B
   Dimensions:              (basis_label: 3)
   Coordinates:
     * basis_label          (basis_label) int64 24B 1 2 3
   Data variables:
       basis_group          (basis_label) object 24B 'inner' 'inner' 'outer'
       basis_partition      (basis_label) object 24B 'generated_inner' ...
       region_in_partition  (basis_label) int64 24B 1 2 1
   Attributes:
       state_dim:  region

The exact byte counts in an xarray display can vary by platform; the important
contract is the coordinate values and their order.

Remainders are explicit
-----------------------

Leaving out the remainder is an error rather than a request for
``BasisLayout`` to invent one:

.. literalinclude:: ../examples/grouped_basis_layout.py
   :language: python
   :start-after: # start-uncovered
   :end-before: # end-uncovered

::

   >>> uncovered_error
   'BasisLayout partitions leave 2 grid cells unmapped.'

Attach the metadata to an operator
----------------------------------

Pass both outputs to ``BasisFunctions.from_flat_basis``.  The
``state_metadata`` option is forwarded to ``BucketBasisOperator``:

.. literalinclude:: ../examples/grouped_basis_layout.py
   :language: python
   :start-after: # start-operator
   :end-before: # end-operator

Here the raw basis labels are ``[1, 2, 3]``, while ``region_labels="range0"``
makes the final operator state coordinate ``[0, 1, 2]``.  Metadata follows the
states through that relabelling::

   >>> result.state_metadata.basis_label.values
   array([1, 2, 3])
   >>> matrix.region.values
   array([0, 1, 2])
   >>> matrix.coords.to_dataset()[["basis_group", "basis_partition", "region_in_partition"]].compute()
   <xarray.Dataset> Size: 96B
   Dimensions:              (region: 3)
   Coordinates:
       basis_group          (region) object 24B 'inner' 'inner' 'outer'
       basis_partition      (region) object 24B 'generated_inner' ...
       region_in_partition  (region) int64 24B 1 2 1
     * region               (region) int64 24B 0 1 2
   Data variables:
       *empty*

Select states by group
----------------------

Because the grouping fields are coordinates on the state dimension, normal
xarray operations can select an inner or outer subset:

.. literalinclude:: ../examples/grouped_basis_layout.py
   :language: python
   :start-after: # start-selection
   :end-before: # end-selection

::

   >>> inner_state.region.values
   array([0, 1])
   >>> outer_state.region.values
   array([2])
   >>> outer_state.basis_partition.item()
   'explicit_remainder'

Round-trip preservation
-----------------------

``BasisFunctions`` preserves the state metadata in its versioned DataTree
representation.  ``save`` and ``load`` use the same representation for NetCDF
or Zarr artifacts.

.. literalinclude:: ../examples/grouped_basis_layout.py
   :language: python
   :start-after: # start-roundtrip
   :end-before: # end-roundtrip

::

   >>> restored_matrix.region.equals(matrix.region)
   True
   >>> restored_matrix.basis_group.equals(matrix.basis_group)
   True
   >>> restored_matrix.region_in_partition.equals(matrix.region_in_partition)
   True

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
