Nested-domain RHIME tutorial
============================

Nested-domain RHIME combines a coarse outer model domain with a finer inner
domain without counting emissions in the overlap twice. The modern workflow:

#. retrieves the outer and inner footprint/flux scenarios independently;
#. keeps only sites available on both domains, filters the canonical outer
   observations once, and mirrors those retained times onto the inner data;
#. sets the outer ``fp``, ``fp_x_flux``, and retained prior flux to zero over
   the inner latitude/longitude extent;
#. builds a native basis and sensitivity matrix on each grid;
#. aligns ``H_inner`` to the canonical ``(site, time)`` observation index; and
#. samples ``H @ x_outer + H_inner @ x_inner`` with the ordinary RHIME
   boundary-condition, model-error, and likelihood components.

The two grids are not RHIME emissions sectors. A sector separates different
flux sources on one spatial layout; a nested domain separates two resolutions
of the same flux source and must retain two independent basis operators.

Surface-site example
--------------------

Use :func:`openghg_inversions.rhime.run_rhime_nested` with the ordinary modern
RHIME arguments plus the inner-domain options:

.. code-block:: python

   from openghg_inversions.rhime import run_rhime_nested

   result = run_rhime_nested(
       species="ch4",
       sites=["TAC", "MHD"],
       averaging_period=["1h", "1h"],
       domain="EUROPE",
       inner_domain="6km",  # retrieves OpenGHG domain EUROPE-6km
       start_date="2019-01-01",
       end_date="2019-02-01",
       flux_sources=["total-ukghg-edgar7"],
       obs_store="user",
       footprint_store="outer-footprints",
       emissions_store="outer-emissions",
       inner_footprint_store="inner-footprints",
       inner_emissions_store="inner-emissions",
       basis_algorithm="weighted",
       nbasis=100,
       fix_basis_outer_regions=True,
       outer_regions_path="/path/to/outer_region_definition_EUHROB.nc",
       # Quadtree is the safe default for a grid without a matching land/sea file.
       inner_basis_algorithm="quadtree",
       inner_nbasis=80,
       use_bc=True,
       output_name="europe_nested_6km",
       output_format="none",
       save_inversion_output=False,
       draws=1000,
       tune=1000,
       chains=4,
   )

``inner_domain`` may instead contain the complete OpenGHG domain name, such as
``"EUROPE-EUHROB"``. A suffix is prefixed with the outer domain exactly once.
The outer boundary-condition contribution is retained; an independent inner
boundary condition is not added. ``outer_regions_path`` is optional; use it
when the fixed outer-region map for the nested footprint extent differs from
the ordinary packaged map for the outer domain. Migrated legacy configs may
still use ``outer_region_definition_file``; RHIME warns and normalizes it to
``outer_regions_path``.

When both bases are generated and ``inner_nbasis`` is omitted, ``nbasis`` is
treated as the total outer-plus-inner budget. RHIME splits it using the
square-root ratio of the retained absolute ``fp_x_flux`` sensitivities, with
the inner share bounded between 35 and 60 percent. Set ``inner_nbasis``
explicitly when ``nbasis`` should remain the independent outer target.

The result exposes the ordinary RHIME objects and both native preparations:

.. code-block:: python

   result.idata.posterior[["x_outer", "x_inner"]]
   result.inv_inputs[["H", "H_inner"]]
   result.outer_basis_functions
   result.inner_basis_functions

INI configuration
-----------------

Nested options can be added to a copy of ``rhime_template.ini`` and run with
the installed CLI. The remaining sections use the normal RHIME schema.

.. code-block:: ini

   [INPUT.NESTED_DOMAIN]
   inner_domain = "6km"
   inner_footprint_store = "inner-footprints"
   inner_emissions_store = "inner-emissions"
   inner_basis_algorithm = "quadtree"
   inner_nbasis = 80
   inner_time_tolerance = None

   [RHIME.OUTPUT]
   output_name = "europe_nested_6km"
   output_format = "none"
   save_inversion_output = False

.. code-block:: bash

   openghg-inversions run-rhime-nested -c nested_rhime.ini

Timestamp alignment
-------------------

Exact inner and outer observation timestamps are required by default. If an
inner footprint product uses a known small timestamp offset, set an explicit
tolerance:

.. code-block:: python

   result = run_rhime_nested(
       config_file="nested_rhime.ini",
       inner_time_tolerance="30min",
   )

Nearest matching is performed separately for each site. Any observation with
no inner timestamp inside the tolerance raises an error. RHIME does not
silently duplicate the nearest footprint or replace a missing inner response
with zero.

Satellite example
-----------------

Satellite data uses the same nested workflow. Keep all site-aligned selectors
as lists, even for one synthetic site, so they retain the modern schema shape:

.. code-block:: python

   satellite = run_rhime_nested(
       species="ch4",
       sites=["GOSAT-BRAZIL"],
       averaging_period=["1D"],
       platform=["satellite"],
       inlet=["column"],
       fp_height=["column"],
       max_level=[3],
       domain="BRAZIL",
       inner_domain="6km",
       start_date="2019-01-01",
       end_date="2019-02-01",
       flux_sources=["total-inventory"],
       bc_store="satellite-store",
       obs_store="satellite-store",
       footprint_store="outer-satellite-store",
       emissions_store="outer-emissions-store",
       inner_footprint_store="inner-satellite-store",
       inner_emissions_store="inner-emissions-store",
       output_name="gosat_brazil_nested",
       output_format="none",
       save_inversion_output=False,
   )

The existing satellite column boundary-condition scaling is applied once to
the outer ``H_bc`` during normal RHIME assembly. ``H_inner`` contains only the
fine-grid emissions response.

Prepared-input workflow
-----------------------

For cached or externally prepared data, combine two ordinary
``RhimePreparedInputs`` objects and run from that explicit boundary:

.. code-block:: python

   from openghg_inversions.rhime import (
       combine_nested_rhime_inputs,
       run_rhime_nested_from_prepared_inputs,
   )

   nested = combine_nested_rhime_inputs(
       outer_prepared,
       inner_prepared,
       time_tolerance="30min",
   )
   result = run_rhime_nested_from_prepared_inputs(
       prepared_inputs=nested,
       run_spec=run_spec_with_output_format_none,
       sampler=sampler,
   )

Current output boundary
-----------------------

Nested runs currently require ``output_format="none"``. The returned
``InferenceData`` and both retained basis objects are complete for analysis,
but the existing ``InversionOutput``, basic, PARIS, and legacy writers each
assume one output grid. Rejecting those formats prevents the inner posterior
from being discarded or written on the outer grid. A future dual-grid output
schema can add those formats without changing the nested model contract.
