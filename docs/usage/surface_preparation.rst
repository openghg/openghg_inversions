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
OpenGHG helpers. Accurate centered boundary normalization and anomaly/mass
arithmetic are owned by OGI. The normal
installation temporarily pins OpenGHG devel commit
``b9a9d1f21f22d2f8dcca714283eb936c22338c57``. This pilot calls its xarray
integrated and time-resolved convolution operations. Both pip and uv installs receive that revision; VG
callers need no separate dependency override. Replace this pin with a suitable
released minimum before publishing OGI to PyPI. The local curtain kernel does
not depend on an unmerged upstream change:

.. code-block:: bash

   uv sync --locked

The required surface tests exercise the installed provider. Missing required
operations fail explicitly. Private provider imports may migrate to public
operations later; that transition does not block this workflow.

Centered boundary provenance records
``boundary_operator_owner="openghg_inversions"``.
The operation uses native boundary/transport inputs, never rounded absolute
sensitivities, and records its arithmetic policy revision.

For centered preparation, OGI normalizes all four stable boundary curtains together, using an
explicit tolerance for their total raw mass versus unity. Choose that tolerance
from the footprint's justified numerical error. Nonfinite or negative weights,
zero mass, and mass outside the tolerance fail validation. Boundary coverage must
be declared by ``time_period`` interval starts. Concentrations and references are
converted to common units before subtraction. Centered preparation arithmetic uses float64; this cannot recover precision already lost in stored input.

Input validation
----------------

Owned views of raw observations, site transport, each source footprint, fluxes
and boundary conditions are checked against the corresponding OpenGHG structural
schemas at the preparation boundary. Standalone ``prepare_surface_site`` checks
all raw input roles before provider operations. Fresh acquisition checks raw
observations and transport per retained site before resolving observation bins,
then checks retrieved fluxes and boundaries once before calculating responses.
A shared dataset can serve more than one of these roles. Schema selection follows
the integrated, split resolved or legacy resolved fields being consumed. Valid
integer spatial labels remain accepted. Validation preserves borrowed payloads,
chunks and lazy auxiliary coordinates. Xarray and Pint own alignment and unit
conversion; scientific steps own coverage and weight policies.

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
such as ``co2_quality_flag``, keep their names. Only a temporary validation view
renames the primary for the provider schema; prepared concentration fields use
``mf`` and its documented ancillary names.

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
       boundary_representation="centered",
       boundary_reference=400.0,
       boundary_reference_units="ppm",
       boundary_mass_tolerance=justified_mass_tolerance,
       output_units="ppm",
   )

Ordinary preparation is the default. Omit ``boundary_representation`` and the
reference arguments to obtain absolute ``bc_n/e/s/w`` contributions. Supply
``boundary_representation="centered"`` plus the reference to retain accurate
anomaly/mass operands instead. A reference alone never selects centered preparation;
Centered preparation requires a justified mass tolerance; ordinary preparation
uses OpenGHG’s existing baseline arithmetic without joint mass normalization.

Each source and boundary contribution is calculated at native footprint release
times, then averaged to the chosen observation cadence. Each source is convolved
independently. ``time_resolved`` includes both its
resolved history and its monthly residual; adding an integrated response for
that same physical contribution would double count it. Modes describe distinct
physical contributions, not inference state groups. ``fp_x_flux_sectoral`` retains
source and grid labels; ``fp_x_flux`` is its explicit source sum. Explicit surface
source labels are sorted. Response concentrations use observation units or the
explicit ``output_units``. Spatial axis order follows labelled inputs, with
``time`` placed last. Signed fluxes
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
       use_bc=True, boundary_representation="centered",
       reference=400.0, reference_units="ppm",
       mass_tolerance=justified_mass_tolerance,
   )

Pass the appropriate store and dataset selectors for your installation. This
route delegates to ``acquire_surface_data`` and shares concrete observation and
footprint retrieval with the existing scenario route. It retains observation
averaging, calibration, inlet/height selection, missing-site handling, explicit
total observation errors and flux sanitation.

Acquisition retrieves retained observations and footprints before flux. It uses
one observation-role resolution, unit conversion and coarsest alignment per
retained site, reusing those arrays and native release bins for preparation.
It then combines the actual backward history and complete release-month residual
support across retained sites and sources. A 24-hour history is the usual case;
120-hour histories also use their actual lag coordinates. No fixed history cap
or preceding-month assumption is imposed. Legacy ``fp_HiTRes`` excludes its final
residual entry when deriving resolved history. Unused releases and dropped sites
do not enlarge the request. A history that crosses a month boundary requires its
actual preceding hours, rather than the full preceding month.

All sources are queried over the common required span so mixed integrated and
resolved modes can retain compatible native axes. Retrieval includes the
preceding declared intervals, including calendar labels offset from month or year
starts. The getter resolves one common effective span for all sources; when
period metadata are absent, a returned cadence can trigger one bounded common
earlier request without applying source padding again.
The full retrieved native flux support is retained. Explicit surface native
prior-flux reporting sorts source labels and converts to the first sorted
source's flux units. Default acquisition retains source insertion order.
Older-inventory relabelling and climatological year reuse are disabled.

When a boundary period is declared, both acquisition routes include the interval
governing a mid-period start and preserve its native timestamp.

Selected identities are captured at retrieval in owned wrapper metadata while
borrowing the datasets. Preparation combines those facts with transport metadata
and serializes provenance once. Combined observation versions remain unknown
when retrieval did not preserve them; combined footprint identifiers retain their
order.

The surface preparation policy revision records the scientific ordering: form
native source products before averaging onto observation bins. It is independent
of prepared-input schema versions and manifest format versions, which describe
serialized artifacts.

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

Centered preparation handoff
----------------------------

Apply supported observation selection first, then derive missing observation
errors with ``prepare_observation_errors`` on the retained population. Explicit
existing errors remain unchanged. Project the native ``bc_anomaly_*`` and
``bc_mass_*`` arrays through the same chosen BC basis using ``bc_sensitivity`` or
``build_sensitivities``. Their outputs are ``H_bc_anomaly`` (A), ``H_bc_mass`` (M)
and scalar ``bc_reference`` (c). They retain spatial BC labels through gathering.
The reference must agree across sites within a tracer.

The source-neutral adapter accepts these projected operands (choose the summed
response for a shared source-scaling field, or a compatible source-resolved
flux basis for separate fields):

.. code-block:: python

   from openghg_inversions.basis._helpers import bc_sensitivity
   from openghg_inversions.inversion_data import prepare_rhime_inputs_from_xarray
   from openghg_inversions.rhime.co2 import prepare_co2_inputs
   from openghg_inversions.rhime.preparation import filter_observations, prepare_observation_errors

   selected, retained_selectors = filter_observations(
       {"MHD": site}, selectors, filters=["daytime9to5"],
   )
   with_errors = prepare_observation_errors(selected, averaging_error=True)
   projected = bc_sensitivity(with_errors, chosen_boundary_basis)
   canonical = prepare_rhime_inputs_from_xarray(
       {name: data.drop_vars("fp_x_flux_sectoral") for name, data in projected.items()},
       basis_functions=chosen_flux_basis, averaging_period="1h",
   )
   # The reduction must describe these same observations, states and sources.
   prepared = prepare_co2_inputs(canonical, coherent_reduction)
   prepared.save("centered-co2.nc")

Selection filters preserve aligned operands and scalar references. Preparation
owns spatial ``bc_basis_algorithm="NESW"`` or explicit ``bc_basis_paths``;
frequency and anchor belong to the model. Both representations are written in
generic format version 3 with an explicit representation field. Known ordinary
raw-v2 files remain readable; retired baked-v1 and ambiguous centered-v2 files
must be regenerated. See :doc:`prepared_format_support` for the exact map.
Old rounded ``H_bc`` alone cannot recover accurate centered operands.

Standalone CO₂ and linked CO₂/O₂ prepared inputs support saving and reopening
both ordinary boundaries and centered operands. Linked centered operands use the
version-2 dedicated prepared-input schema; ordinary linked inputs retain
version-1 compatibility. See :doc:`prepared_format_support` for supported formats.

The additive-Gaussian and cached fixed-OU CO₂ and linked CO₂/O₂ routes bind raw
operands to BC periods at model construction. Their model-facing runners and
direct builders accept ``bc_freq`` and ``bc_anchor_time``. Direct builders
consume raw ``boundary_operands`` (in linked models, a mapping keyed by tracer).
Configured and supported saved routes select the same choices in their
boundary configuration, for example:

.. code-block:: toml

   [model.boundary]
   enabled = true
   frequency = "1D"
   anchor_time = "2020-01-01"

Period choices apply to both raw ordinary sensitivities and centered operands,
using the same selection and period masks. A saved raw preparation can be reused
with different supported periods. Already-expanded sensitivities cannot be
reinterpreted with a period override. Linked channels use one shared
period/anchor choice and must agree when both specify it. Their tables are
``channels.co2.boundary`` and ``channels.o2.boundary``. Anchors apply to fixed
durations and must be omitted for calendar-month periods.

Native observation labels select boundary rows. When observations carry a
``time`` coordinate, it determines period membership and replaces auxiliary
boundary timestamps. Boundary times that also serve as indexed row labels must
agree with the observation timestamps; conflicting values raise an error.

Centered inference supports the additive-Gaussian recipe, ordinary fixed-OU
likelihoods with fixed or inferred amplitudes, and inferred-amplitude fixed OU
using the matched cached sampler. Both multiplicative and additive BC priors
can use numerical centering; a linked model can retain multiplicative CO2
boundaries while fitting additive O2 corrections. Configure PyTensor
``floatX=float64`` before model construction; these routes reject an explicit
float32 choice.

The concrete CO2 builder passes centered observations and means to a custom
likelihood callback with unchanged error operands. The prepared-input runner
restricts centered replay to its supported built-in likelihoods because its
fixed observed-variable roles and predictive-restoration contract have not been
established for arbitrary callbacks. Direct custom-model callers own that
contract. Standard and multisector model builders still reject centered inputs;
their preparation operations can produce the reusable canonical handoff shown
above.

For the original BC coefficients b, the centered likelihood uses
``q = c * M @ ones``, ``y_centered = y - q`` and
``mu_centered = flux + A @ b + c * M @ (b - ones) + offset``.
This preserves the original coefficients, priors, fixed states, errors and
covariance, including constant curtains and bases that do not partition unity.
BC and total concentration deterministics stay absolute. The recipe's trace
annotation restores observed and predictive ``y`` to absolute concentrations
once, using retained reference shifts; replay does not need to rebuild the model.
The supported recipe runners perform this annotation automatically. Callers composing
PyMC operations directly must restore sample coordinates, then call
:func:`~openghg_inversions.rhime.co2.centered_boundary.restore_absolute_predictives`
before serialization or postprocessing.

The cached sampler preserves the physical BC variables and their priors. Only
its private quadratic uses active coefficients ``b_active - 1`` and design
``(A + c * M)[:, active]``. Its fixed BC term is
``A @ b_star + c * M @ (b_star - ones)``, where ``b_star`` is one for active
states and the physical fixed value otherwise. This term uses the full operands
before removing zero design columns. Sampled flux and offset designs retain
their usual coefficients. Both amplitude and state updates use the centered
residual; the OU and aggregation covariance are unchanged.

A small offline example creates its own inputs, performs a coherent reduction,
saves/reopens preparation and restores absolute prior predictions:

.. code-block:: bash

   python -m examples.surface_centered_pilot /tmp/surface-pilot

The pilot is tracked in `issue 957
<https://github.com/openghg/openghg_inversions/issues/957>`_. Wider adoption,
automatic coherent preparation, public provider APIs, alternative O₂ observables
and native mixed-cadence reporting remain separate work.
