.. _nested-domain-rhime-tutorial:

Run a nested-domain RHIME inversion
===================================

Nested-domain RHIME combines a surrounding outer transport domain with a
higher-resolution inner transport domain without counting emissions in the
overlap twice. The workflow:

#. retrieves the outer and inner footprint/flux scenarios independently;
#. keeps only sites available on both domains, filters the canonical outer
   observations once, and mirrors those retained times onto the inner data;
#. sets the outer ``fp``, ``fp_x_flux``, and retained prior flux to zero over
   the inner latitude/longitude extent;
#. builds a native basis and sensitivity matrix on each grid;
#. aligns ``H_inner`` to the canonical ``(site, time)`` observation index; and
#. samples ``H @ x_outer + H_inner @ x_inner`` with the ordinary RHIME
   boundary-condition, model-error, and likelihood components.

All retained sites currently use the same outer domain and the same inner
domain. Both domains must be available for each retained site. Preparation
uses OpenGHG's domain definitions to normalize footprint coordinates and
matching runtime flux coordinates before masking and basis generation. For an
unregistered domain, it warns and uses the first retained site's grid as the
reference. Supplied basis partitions are aligned to their runtime flux grid
before constructing the basis. Incompatible site grids are rejected before
masking or basis projection.
The shared inner rectangle is applied separately to each actual target
footprint or flux grid.
Masking borrows the original arrays and preserves Dask laziness.

The two grids are not RHIME emissions sectors or grouped basis regions. A
sector separates different flux sources on one spatial layout; a nested domain
separates two transport grids for the same flux source and must retain two
independent basis operators. Native runtime flux is resolved separately for
each grid, after the outer scientific mask is applied. Each basis binds its
current native prior; generation uses the first requested source for design
weights. Reusing a saved operator does not replace either current prior with
the artifact's saved flux. Observation-aligned response arrays remain separate
inputs to sensitivity construction. See the :doc:`family support summary
<nested_domain_model_family>` before using this guide.

Model and state semantics
-------------------------

The outer sensitivity ``H`` has state dimension ``region`` and scales with
``x_outer``. The inner sensitivity ``H_inner`` has the independent state
dimension ``inner_region`` and scales with ``x_inner``. They meet only in the
shared observation dimension:

.. math::

   \mu_{\mathrm{outer},t}
   = \sum_r H_{r,t} x_{\mathrm{outer},r},
   \qquad
   \mu_{\mathrm{inner},t}
   = \sum_q H_{\mathrm{inner},q,t} x_{\mathrm{inner},q}.

The model records these contributions as ``mu_outer`` and ``mu_inner``. Its
``mu`` deterministic is their pollution sum:

.. math::

   \mu_t = \mu_{\mathrm{outer},t} + \mu_{\mathrm{inner},t}.

When ``inner_x_prior`` is omitted, ``x_inner`` uses a copy of the outer
scaling prior; pass ``inner_x_prior`` to choose it independently. Any enabled
outer boundary-condition or offset contribution is added to this pollution sum
when constructing the mean passed to the shared likelihood, which applies the
configured observation error, aggregation error, and model-data mismatch
treatment.

Both state vectors are sampled jointly. Separating their posterior influence
therefore depends on the outer and inner footprints producing sufficiently
distinct observation-space sensitivities.

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
       flux_store="outer-emissions",
       inner_footprint_store="inner-footprints",
       inner_flux_store="inner-emissions",
       basis_algorithm="weighted",
       nbasis=100,
       fix_basis_outer_regions=True,
       outer_regions_path="/data/maps/intem_region_definition_EUHROB.nc",
       # Quadtree is the safe default for a grid without a matching land/sea file.
       inner_basis_algorithm="quadtree",
       inner_nbasis=80,
       use_bc=True,
       mismatch_model="pollution_event",
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
boundary condition is not added. ``outer_regions_path`` selects an exact support
file. The bundled
``intem_region_definition_EUHROB.nc`` for the EUROPE 6 km nest marks label 6
as the inner rectangle. To select it explicitly from Python, use
``packaged_basis_support_path("outer_regions", "EUHROB")`` from
``openghg_inversions.basis``. Relative paths name caller files; they do not
implicitly search packaged assets. Migrated legacy configs may still use
``outer_region_definition_file``; RHIME warns and normalizes it to
``outer_regions_path``.

When a store contains fine-grid footprints but only a coarser flux product,
set ``inner_flux_domain`` to that flux product's domain. RHIME then
interpolates the flux density with nearest neighbours onto the inner footprint
grid explicitly. For example, the PARIS 6 km footprints use
``inner_flux_domain="EUROPE"`` with the shared EUROPE emissions product.

The EUHROB asset is a coarse-grid fixed-region map: its marked rectangle tells
the outer-basis algorithm where the nested area lies, while all other labels
remain fixed outer regions. It does not replace the independent basis built on
the native 6 km grid.

Because the marked rectangle covers the same extent as the real inner domain,
nested-domain outer masking (below) always zeroes the outer footprint response
and prior flux there before the outer basis is built -- there is nothing left
for the outer algorithm to subdivide inside that label. ``prepare_nested_rhime_inputs``
handles this by passing ``allow_empty_inner_region=True`` to
:func:`~openghg_inversions.basis.fixed_outer_regions_basis_from_data`
for the outer domain only: the marked label is
kept as a single fixed region (like every other outer label) instead of
raising. Outside nested preparation, an empty marked region still raises,
since it normally signals a mismatched map or missing footprint/flux data.

When both bases are generated and ``inner_nbasis`` is omitted, ``nbasis`` is
treated as the total outer-plus-inner budget. RHIME splits it using the
square-root ratio of the retained absolute ``fp_x_flux`` sensitivities, with
the inner share bounded between 35 and 60 percent. Set ``inner_nbasis``
explicitly when ``nbasis`` should remain the independent outer target.

Use saved basis files
---------------------

Saved outer and inner bases remain separate because each belongs to its own
native grid. Configure the exact outer and inner artifacts independently with
``basis_path`` and ``inner_basis_path``:

.. code-block:: python

   result = run_rhime_nested(
       config_file="nested_rhime.ini",
       basis_path="/data/basis/outer.nc",
       inner_basis_path="/data/basis/inner-6km.zarr",
   )

Do not create one basis artifact spanning both grids. When generating a basis
instead, ``basis_output_path`` and ``inner_basis_output_path`` save the outer
and inner artifacts at their exact destinations independently.
These exports also run with ``run_nested(config=config, prepared=nested,
inner_basis_output_path=...)``. Each returned native basis and its saved
``InversionOutput`` retain the export destination as provenance; the borrowed
prepared inputs remain unchanged.
Without an export destination, the validated preparations are reused directly.
``landsea_path`` and ``inner_landsea_path`` select each grid's class map when
weighted generation needs one; support-map selection fails for missing files
or unsupported domains before acquisition. If no inner basis file or algorithm
is specified, the inner grid uses the quadtree algorithm.

The returned ``NestedOutput`` exposes an outer and an inner ``InversionOutput``
view. They borrow one sampled trace and retain their separate native bases:

.. code-block:: python

   result.outer.trace.posterior[["x_outer", "x_inner"]]
   result.outer.inv_inputs["H"]
   result.outer.basis_functions
   result.inner.basis_functions

INI configuration
-----------------

Nested options can be added to a copy of ``rhime_template.ini`` and run with
the installed CLI. Add only the new ``[INPUT.NESTED_DOMAIN]`` section below;
then edit the options under the template's existing ``[RHIME.OUTPUT]`` section
rather than appending a duplicate section. The current INI loader flattens
section names, so section placement is organizational rather than a namespace
and option names must not be repeated between sections.

.. code-block:: ini

   [INPUT.NESTED_DOMAIN]
   inner_domain = "6km"
   inner_footprint_store = "inner-footprints"
   inner_flux_store = "inner-emissions"
   inner_flux_domain = None
   inner_basis_algorithm = "quadtree"
   inner_nbasis = 80
   inner_time_tolerance = None

   ; Edit these values in the existing [RHIME.OUTPUT] section:
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
       flux_store="outer-emissions-store",
       inner_footprint_store="inner-satellite-store",
       inner_flux_store="inner-emissions-store",
       mismatch_model="pollution_event",
       output_name="gosat_brazil_nested",
       output_format="none",
       save_inversion_output=False,
   )

The existing satellite column boundary-condition scaling is applied once to
the outer ``H_bc`` during normal RHIME assembly. ``H_inner`` contains only the
fine-grid emissions response.

Prepared-input workflow
-----------------------

For cached or externally prepared data, the outer preparation must be built
from merged data whose inner-domain footprint response and flux were already
masked before basis projection. That spatial contribution cannot be removed
from an already projected ``H``. Combine the two preparations with the
explicit acknowledgement shown below and run from that boundary:

.. code-block:: python

   from openghg_inversions.rhime import (
       RhimeConfig,
       combine_nested_rhime_inputs,
       run_nested,
   )

   nested = combine_nested_rhime_inputs(
       outer_prepared,
       inner_prepared,
       time_tolerance="30min",
       outer_overlap_masked=True,
   )
   config = RhimeConfig.from_params({
       "species": "ch4",
       "sites": ["TAC", "MHD"],
       "averaging_period": ["1h", "1h"],
       "domain": "EUROPE",
       "start_date": "2019-01-01",
       "end_date": "2019-02-01",
       "flux_sources": ["total-ukghg-edgar7"],
       "mismatch_model": "pollution_event",
       "output_name": "prepared_nested",
       "output_format": "none",
   }, multisector=False)
   result = run_nested(config=config, prepared=nested)

This uses the same materialization, model construction, inference and output
operations as a complete run. Supplied preparation is validated at entry;
acquisition and scientific preparation are skipped. No staged manifest is
required. Prepared-input request-window semantics are being considered across
workflows in `issue 913 <https://github.com/openghg/openghg_inversions/issues/913>`_.

For direct graph inspection, call ``materialize_nested_inputs`` on
``nested.combined`` and ``config.model``, then pass its returned Dataset and
``priors`` to ``build_nested_rhime_model`` with the same model specification.

To prepare the two domains yourself, first call
``inner_data_config(config.data, inner_domain="6km", ...)`` from
``openghg_inversions.rhime``. This is the same configuration step used by the
full runner: it inherits the outer stores, resolves independent inner basis
paths, defaults to quadtree, and disables the inner boundary component. Pass
the outer and inner configurations to ``resolve_basis_inputs`` and
``acquire_merged_data``, then pass the acquired data and resolved basis inputs
to ``prepare_nested_rhime_inputs``. Keep ``inner_nbasis=None`` at preparation
when requesting automatic total-budget allocation.

Omitting ``outer_overlap_masked=True`` raises rather than silently combining
two ordinary overlapping sensitivity matrices. The retrieval-backed
``run_rhime_nested`` path performs the mask before preparation and supplies
this acknowledgement internally.

Current output boundary
-----------------------

Nested runs support ``output_format="none"`` or ``output_format="paris"``.
The existing single-grid ``InversionOutput``, basic, and legacy writers each
assume one output grid, so those formats are still rejected: using them would
either discard the inner posterior or mis-grid it onto the outer domain.

PARIS output does not merge the two grids either. Instead, the nested RHIME
recipe builds two ordinary, single-grid ``InversionOutput`` *views* of the
shared trace -- one per domain, each with its variable roles pointed at that
domain's tagged posterior (``x_outer``/``hx_outer`` or ``x_inner``/``hx_inner``)
-- and passes them to the existing single-grid PARIS/flux/country
postprocessing. The selected inner trace is renamed to the retained inner
basis operator's state dimension at this view boundary; the shared sampled
trace is unchanged. This produces three products:

- an outer PARIS flux file, matching the ordinary single-grid schema exactly,
  with zero emissions inside the inner extent (the outer prior flux and
  footprint response are already masked there by
  ``mask_outer_merged_for_inner_domain``), so it never double-counts against
  the inner file;
- an inner PARIS flux file at the inner domain's own native resolution,
  including its own country totals (a coarse country-definition file is
  nearest-neighbour resampled onto the inner grid and cached below the run's
  output path, since a fine domain rarely has a matching country file at its
  native resolution; the cache uses the inner domain and source country file,
  reuses compatible grids across small coordinate differences, and regenerates
  a map when the actual target grid changes);
- one shared PARIS concentration file, since the combined forward model
  (``mu = mu_outer + mu_inner + baseline``) already reports one concentration
  per observation -- there is no separate "inner concentration" product.

``run_rhime_nested`` requires ``output_path`` for PARIS and writes all three, using the
inner domain's label (e.g. ``EUROPE-6km``) in the inner flux file's name.
Before acquisition or inference, shared PARIS option resolution validates the
postprocessing option names, template version, time point, explicit flux
frequency and any explicit
``inner_country_file``. These checks create no output or country cache. They
also apply when calling ``make_nested_paris_products`` directly; options that
depend on the sampled data are resolved during product construction.
The returned ``NestedOutput.products`` retains ``paris_flux``,
``paris_flux_inner`` and ``paris_concentration`` so Python callers can inspect
the products without reconstructing them. ``NestedOutput.paths`` records the
corresponding ``paris_*_path`` destinations and any written
``inversion_output_path``, ``inversion_output_inner_path`` and ``trace_path``.
When inversion-output artifacts are requested, each domain view also retains
its actual outer/inner identity, coordinate-grid sizes and extents,
the explicit trace-to-basis state-dimension mapping, and the outer overlap
mask policy. These fields survive an ``InversionOutput`` save/load round trip.
The applied mask bounds are retained from the inner footprint grid. For
externally pre-masked inputs, pass ``outer_overlap_mask_extent`` as
``(lat_min, lat_max, lon_min, lon_max)`` to ``combine_nested_rhime_inputs`` when
known; omission records an unknown extent rather than deriving it from the
inner flux grid.
The inner view records its own state-activity policy.
Use ``make_nested_paris_products(result.outer, result.inner,
inner_domain_label="EUROPE-6km", country_file=...)`` for explicit product
construction from these views. The scientific result contains no live PyMC
model, sampler or writing policy. See :doc:`legacy_and_migration` for the
removed result and prepared-runner interfaces.
