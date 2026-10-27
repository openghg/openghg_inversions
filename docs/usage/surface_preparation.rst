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
``b9a9d1f21f22d2f8dcca714283eb936c22338c57``, which includes the Numba
``fp_x_flux`` algorithm. Both pip and uv installs receive that revision; VG
callers need no separate dependency override. Replace this pin with a suitable
released minimum before publishing OGI to PyPI. Install the pinned provider with:

.. code-block:: bash

   uv sync --locked

The required surface tests exercise the installed provider. Missing required
operations fail explicitly. Private provider imports may migrate to public
operations later; that transition does not block this workflow.

Explicit source responses
-------------------------

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
and complete calendar months for the resolved residual. Unsupported cadences,
gaps or coverage fail instead of selecting another transport calculation.

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
route reuses getters and retains their observation averaging, calibration,
inlet/height selection, missing-site handling, flux sanitation and provenance.
It requests additional flux history for resolved transport and disables the
older-inventory fallback that would relabel historical flux as current data.
Other getter improvements remain separately tracked.

Fresh acquisition requires compatible native source axes for retained-flux
reporting. Standard/multisector scientific preparation of a supplied surface
``MergedData`` applies the same exact-axis policy, converting units before
stacking or summing. Mixed-cadence observation-space responses can be calculated
with ``prepare_surface_site``; they do not yet define a common native reporting
axis or an affine flux companion. Supply an independently justified native
representation before using such products. This pilot does not construct the
ordinary CO₂ coherent reduction automatically; see :doc:`coherent_reduction` and
:doc:`co2_model_family`.
