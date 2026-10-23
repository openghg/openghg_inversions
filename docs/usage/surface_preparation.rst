Surface CO₂/O₂ preparation pilot
================================

The experimental ``prepare_surface_site`` producer calculates continuous-surface
CO₂ and O₂ responses without constructing ``ModelScenario``. Existing acquisition
defaults remain available. The pilot supports O₂ mole fractions; it does not
convert per-meg or APO observations. Column, satellite and flask preparation
continue to use their existing owners.

Dependency and precision
------------------------

Centered boundary arithmetic is implemented privately in OGI. Integrated-footprint
preparation, including centered boundaries, works with the released OpenGHG
dependency installed by ``uv sync --locked``.

Time-resolved transport additionally requires OpenGHG's fix for monthly residual
fluxes in windows beginning mid-month. The tested development revision is
``28600028f23a694dbac299d93ff5f229d37aaef6``. Install it explicitly after
synchronizing the normal dependencies:

.. code-block:: bash

   uv sync --locked
   uv pip install "openghg @ git+https://github.com/openghg/openghg.git@28600028f23a694dbac299d93ff5f229d37aaef6"

A later ``uv sync --locked`` restores the release in the lockfile. The optional
``py312-surfacePilot`` tox environment tests this same pinned revision. Before
resolved preparation, OGI checks the monthly behavior using small synthetic data;
providers missing the fix fail explicitly. This check does not compute user data.
OpenGHG
still supplies observation alignment and flux convolution through private helpers;
those imports can migrate to public operations in a subsequent OpenGHG release.
The local boundary operation computes anomaly and mass operands directly from
the curtains, without using rounded legacy BC products.

Boundary provenance records ``boundary_operator_owner="openghg_inversions"``;
``boundary_provider_policy_revision`` retains the arithmetic policy identifier.

OGI normalizes all four stable boundary curtains together, using an
explicit tolerance for their total raw mass versus unity. Choose that tolerance
from the footprint's justified numerical error. Nonfinite or negative weights,
zero mass, and mass outside the tolerance fail validation. Boundary coverage must
be declared by ``time_period`` interval starts. Concentrations and references are
converted to common units before subtraction. Arithmetic and the centered PyMC
binding use float64; this cannot recover precision already lost in stored input.
For configured runs, add ``floatX=float64`` to ``PYTENSOR_FLAGS`` before starting
Python, retaining any other settings. The offline example selects float64 explicitly.

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
       boundary_reference=400.0,
       boundary_reference_units="ppm",
       boundary_mass_tolerance=justified_mass_tolerance,
       output_units="ppm",
   )

Each source is convolved independently. ``time_resolved`` includes both its
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
       use_bc=True, reference=400.0, reference_units="ppm",
       mass_tolerance=justified_mass_tolerance,
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

.. _surface-centered-boundary-contract:

Centering through preparation and inference
-------------------------------------------

Derive observation errors before filtering, using the existing
``prepare_observation_errors`` owner. Project the native ``bc_anomaly_*`` and
``bc_mass_*`` arrays through the same chosen BC basis using ``bc_sensitivity`` or
``build_sensitivities``. Their outputs are ``H_bc_anomaly`` (A), ``H_bc_mass`` (M)
and scalar ``bc_reference`` (c). They retain spatial BC labels through gathering
and prepared save/reopen. The reference must agree across sites within a tracer.

The source-neutral adapter accepts these projected operands (choose the summed
response for a shared source-scaling field, or a compatible source-resolved
flux basis for separate fields):

.. code-block:: python

   from openghg_inversions.basis._helpers import bc_sensitivity
   from openghg_inversions.inversion_data import prepare_rhime_inputs_from_xarray
   from openghg_inversions.rhime.preparation import prepare_observation_errors
   from openghg_inversions.rhime.co2 import prepare_co2_inputs

   with_errors = prepare_observation_errors({"MHD": site}, averaging_error=True)
   projected = bc_sensitivity(with_errors, chosen_boundary_basis)
   canonical = prepare_rhime_inputs_from_xarray(
       {name: data.drop_vars("fp_x_flux_sectoral") for name, data in projected.items()},
       basis_functions=chosen_flux_basis, averaging_period="1h",
   )
   # The reduction must describe these same observations, states and sources.
   prepared = prepare_co2_inputs(canonical, coherent_reduction)
   prepared.save("centered-co2.nc")

Selection filters preserve the operands. ``six_hr_mean`` uses common valid science
rows; missing optional diagnostics do not choose those rows. ``daily_median`` is
rejected for centered data. Absolute historical prepared files remain readable;
centered files use schema version 2 and reject incomplete or already period-expanded
operands. Old ``H_bc`` alone cannot be converted into accurate centered operands.

The ordinary CO₂ model and ordinary linked CO₂/O₂ model bind the raw operands to
BC periods at model construction. Direct builders accept ``boundary_operands``
(in linked models, a mapping keyed by tracer), ``bc_freq`` and ``bc_anchor_time``.
Configured ordinary and saved routes select the same choices in their boundary
configuration, for example:

.. code-block:: toml

   [model.boundary]
   enabled = true
   frequency = "1D"
   anchor_time = "2020-01-01"

Period choices apply to raw centered operands; already-expanded absolute
sensitivities cannot be reinterpreted this way. Linked channels use one shared
period/anchor choice and must agree when both specify it. Their tables are
``channels.co2.boundary`` and ``channels.o2.boundary``. Anchors apply to fixed
durations and must be omitted for calendar-month periods.

Use the ordinary additive-Gaussian recipe. Cached, fixed-OU and custom likelihood
routes reject centered inputs until separately validated. Standard and multisector
model builders also reject centered inputs; their preparation operations can
still produce the reusable canonical handoff shown above.

For the original BC coefficients b, the centered likelihood uses
``q = c * M @ ones``, ``y_centered = y - q`` and
``mu_centered = flux + A @ b + c * M @ (b - ones) + offset``.
This preserves the original coefficients, priors, fixed states, errors and
covariance, including constant curtains and bases that do not partition unity.
BC and total concentration deterministics stay absolute. The recipe's trace
annotation restores observed and predictive ``y`` to absolute concentrations
once, using retained reference shifts; replay does not need to rebuild the model.
Ordinary recipe runners perform this annotation automatically. Callers composing
PyMC operations directly must restore sample coordinates, then call
:func:`~openghg_inversions.rhime.co2.centered_boundary.restore_absolute_predictives`
before serialization or postprocessing.

A small offline example creates its own inputs, performs a coherent reduction,
saves/reopens preparation and restores absolute prior predictions:

.. code-block:: bash

   python -m examples.surface_centered_pilot /tmp/surface-pilot

The pilot is tracked in `issue 957
<https://github.com/openghg/openghg_inversions/issues/957>`_. Wider adoption,
automatic coherent preparation, public provider APIs, alternative O₂ observables
and native mixed-cadence reporting remain separate work.
