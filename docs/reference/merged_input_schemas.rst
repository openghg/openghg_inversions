Merged-input schemas
====================

These references are generated from the live ``xarray-validate`` schema factories
used by ``MergedData.validate_scientific_inputs``. They describe selected
preparation variants, rather than prescribing a single layout for every recipe.
Other combinations use the same factories with different keyword arguments.
See :doc:`/usage/customising_rhime` for the preparation workflow and unit policy.

Data types describe decoded, in-memory xarray data, not the on-disk
representation. For example, a file may store numeric time offsets with CF
``units`` and ``calendar`` attributes; xarray decodes these before validation.
The time coordinates in these schemas must be ``datetime64`` after decoding.
Dates decoded to ``cftime`` objects do not satisfy that dtype requirement.

Each table distinguishes required fields, optional pattern matches, and allowed
extras. Dimensions are ordered unless the table says otherwise; an empty dimension
list means a scalar. Dtypes such as ``integer`` and ``floating`` denote whole
NumPy dtype families. Relationship descriptions come from the validation callbacks'
docstrings; those callbacks are not executed to build this page.

This reference describes metadata validation. It does not establish finite
numerical values or replace the checks owned by projection and graph construction.
The generated description is for documentation, not executable schema restoration.

Standard observations with a supplied basis and no BCs
------------------------------------------------------

.. schema-reference:: openghg_inversions.inversion_data._merged_schema:merged_site_schema
   :kwargs: {"split_by_sectors": false, "require_footprints": false, "use_bc": false}

Multisector observations with a generated spatial basis and ordinary BCs
------------------------------------------------------------------------

.. schema-reference:: openghg_inversions.inversion_data._merged_schema:merged_site_schema
   :kwargs: {"split_by_sectors": true, "require_footprints": true, "use_bc": true}

Standard observations with a generated spatial basis and centered BCs
---------------------------------------------------------------------

.. schema-reference:: openghg_inversions.inversion_data._merged_schema:merged_site_schema
   :kwargs: {"split_by_sectors": false, "require_footprints": true, "use_bc": true, "boundary_representation": "centered-v1"}

Static native flux with canonical numerical scale
-------------------------------------------------

.. schema-reference:: openghg_inversions.inversion_data._merged_schema:merged_flux_schema
   :kwargs: {"timed": false}

Time-dependent native flux on the explicit surface route
--------------------------------------------------------

Compatible scaled units are accepted here; ``resolve_runtime_flux`` owns conversion.

.. schema-reference:: openghg_inversions.inversion_data._merged_schema:merged_flux_schema
   :kwargs: {"timed": true, "allow_scaled_units": true}
