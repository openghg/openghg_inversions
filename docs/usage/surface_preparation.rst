Surface CO₂/O₂ preparation pilot
================================

The experimental ``prepare_surface_site`` producer calculates continuous-surface
CO₂ and O₂ responses without constructing ``ModelScenario``. Existing acquisition
defaults remain available. The pilot supports O₂ mole fractions; it does not
convert per-meg or APO observations. Column, satellite and flask preparation
continue to use their existing owners.

Dependency and precision
------------------------

Ordinary boundary sensitivities, observation alignment and flux convolution reuse
OpenGHG helpers. The normal
installation temporarily pins OpenGHG devel commit
``b9a9d1f21f22d2f8dcca714283eb936c22338c57``. This pilot calls its xarray
integrated and time-resolved convolution operations. Both pip and uv installs receive that revision; VG
callers need no separate dependency override. Replace this pin with a suitable
released minimum before publishing OGI to PyPI. Install the pinned provider with:

.. code-block:: bash

   uv sync --locked

The required surface tests exercise the installed provider. Missing required
operations fail explicitly. Private provider imports may migrate to public
operations later; that transition does not block this workflow.

Explicit source responses
-------------------------

Observations accept ``mf`` or the species name (``co2`` or ``o2``) as the primary
mole fraction. The four ancillary roles use the suffixes ``_error``,
``_repeatability``, ``_variability`` and ``_number_of_observations`` with either
prefix. Each role is resolved independently, so mixed names such as ``co2``
with ``mf_error`` are valid. Duplicate aliases for one role fail with the role
and variable names. Explicit errors remain available for subsequent preparation.

For supplied observations with another primary name, set
``observation_variable="measured_concentration"``. Ancillary aliases keep the
same fixed names; a custom primary does not introduce custom error aliases.
The selector must identify the only primary candidate. Unrelated extensions,
such as ``co2_quality_flag``, keep their names. Prepared concentration fields use ``mf`` and its documented ancillary names.

With existing OpenGHG data wrappers or xarray datasets:

.. code-block:: python

   from openghg_inversions.inversion_data import prepare_surface_site

   site = prepare_surface_site(
       observations,
       footprint,
       {"biosphere": biosphere_flux, "fossil": fossil_flux},
       species="co2",
       transport={"biosphere": "time_resolved", "fossil": "integrated"},
       boundary_data=boundary_curtains,
       output_units="ppm",
   )

Boundary preparation produces ordinary absolute ``bc_n/e/s/w`` contributions
using OpenGHG baseline sensitivities with climatology disabled.

Each source and boundary contribution is calculated at native footprint release
times, then averaged to the chosen observation cadence. Each source is convolved
independently. ``time_resolved`` includes both its
resolved history and its monthly residual; adding an integrated response for
that same physical contribution would double count it. Modes describe distinct
physical contributions, not inference state groups. ``fp_x_flux_sectoral`` retains
source and grid labels; ``fp_x_flux`` is its explicit source sum. Signed fluxes
are retained. Fluxes need regular declared validity, complete backward history
and complete calendar months containing the releases used by the resolved residual.
Resolved lag coordinates must be regular, nonnegative whole hours, with spacing
that divides 24 hours. Release and flux intervals remain whole hours; their
absolute timestamp phases may differ. The release/lag transport grid must fit
the provider rolling window and selected lag labels. Coarser flux intervals,
including two-hour flux on an hourly transport grid, are supported when their
cadence fits the resolved history window. A cadence longer than that window is
rejected because the provider would otherwise substitute monthly flux for the
resolved calculation. Gaps and inadequate coverage fail explicitly.

For distinct source footprints, pass a source-to-footprint mapping as the second
argument and supply ``site_footprint_data=...`` explicitly. That separate footprint
owns observation alignment, shared filter fields and boundary transport. Each
source response must match the resulting observations and spatial grid. A source
label or mapping order never chooses the site's boundary transport.

Inputs are borrowed. The producer owns shallow metadata containers around
provider calls; payload calculations remain lazy where supported. Provider
observation alignment can evaluate observation-validity masks to select rows.
Preparation and model materialization remain explicit execution boundaries.

Retrieval and retained native flux
----------------------------------

``MergedData.from_options`` adds an opt-in beside the existing scenario route:

.. code-block:: python

   from openghg_inversions.inversion_data import MergedData, SiteOptions

   selectors = SiteOptions.from_inputs(
       sites=["MHD"], averaging_period="1h", inlet="10m", fp_height="10m",
       time_resolved=True,
   )
   merged = MergedData.from_options(
       species="co2", domain="EUROPE",
       start_date="2020-01-01", end_date="2020-02-01",
       site_options=selectors, flux_sources=["biosphere", "fossil"],
       surface_transport={"biosphere": "time_resolved", "fossil": "integrated"},
       use_bc=True,
   )

Pass the appropriate store and dataset selectors for your installation. This
route delegates to ``acquire_surface_data`` and shares concrete observation and
footprint retrieval with the existing scenario route. It retains observation
averaging, calibration, inlet/height selection, missing-site handling, explicit
total observation errors and flux sanitation.

Acquisition retrieves retained observations and footprints before flux. It uses
the same observation alignment and selected native release bins as preparation,
then combines the actual backward history and complete release-month residual
support across retained sites and sources. A 24-hour history is the usual case;
120-hour histories also use their actual lag coordinates. No fixed history cap
or preceding-month assumption is imposed. Legacy ``fp_HiTRes`` excludes its final
residual entry when deriving resolved history. Unused releases and dropped sites
do not enlarge the request. A history that crosses a month boundary requires its
actual preceding hours, rather than the full preceding month.

All sources are queried over the common required span so mixed integrated and
resolved modes can retain compatible native axes. Retrieval includes the
preceding declared fixed interval for shifted flux labels; when period metadata
are absent, a returned cadence can trigger one bounded earlier request.
The full retrieved native flux support is retained. Older-inventory relabelling
and climatological year reuse are disabled.

Selected identities are captured at retrieval in owned wrapper metadata while
borrowing the datasets. Preparation combines those facts with transport metadata
and serializes provenance once. Combined observation versions remain unknown
when retrieval did not preserve them; combined footprint identifiers retain their
order.

Fresh acquisition requires compatible native source axes for retained-flux
reporting. Standard/multisector scientific preparation of a supplied surface
``MergedData`` applies the same exact-axis policy, converting units before
stacking or summing. Mixed-cadence observation-space responses can be calculated
with ``prepare_surface_site``; they do not yet define a common native reporting
axis or an affine flux companion. Supply an independently justified native
representation before using such products. This pilot does not construct the
ordinary CO₂ coherent reduction automatically; see :doc:`coherent_reduction` and
:doc:`co2_model_family`.
