Customising RHIME
=================

RHIME is built from reusable functions for data preparation, basis
construction, model construction, sampling, and output. This is a deliberate
design choice: when the standard workflow does not express the science you
need, you should be able to start from a working example and replace the
smallest relevant part.

The examples on this page use a procedural style. Data is passed from one
function to the next in the order that the inversion runs, so readers do not
need to design a new class hierarchy or framework before changing a model.
Some library functions return structured objects that group related values,
but the examples show how to use those objects directly.

Reusing the components does not make every custom workflow automatic. A
replacement function is responsible for the scientific choices it introduces,
and a deeper change may also need different data preparation. The remaining
RHIME stages can only validate the inputs and assumptions covered by their
documented interfaces.

Choose the smallest starting point that fits the change:

* To change only the likelihood, pass a Python function to ``run_standard``.
* To run the matched fixed-tau Ornstein--Uhlenbeck (OU) CO₂ model with inferred
  site amplitudes, use :ref:`the package-supported cached-sigma CO₂ recipe
  <co2-cached-sigma-recipe>` in
  the :doc:`CO₂ model family <co2_model_family>`.
* To resume from externally supplied merged observations, footprints, and
  fluxes, pass a borrowed ``RhimeMergedData`` object as ``merged_data``.
* To change a preparation stage such as basis construction, copy the visible
  runner and replace that stage.
* To start from alternative or cached prepared data, use
  ``run_standard(prepared=prepared, config=config)`` or
  ``run_multisector(prepared=prepared, config=config)``. To replace the complete
  standard or multisector graph, construct and sample a PyMC model directly;
  see :ref:`rhime-complete-model-composition` for the explicit multisector
  materialization, construction, sampling and output operations. Both ordinary
  runners return ``InversionOutput``.

.. _cached-sigma-co2-recipe:
.. _run-the-production-cached-sigma-co2-recipe:

The cached-sigma CO₂ recipe documentation moved to the
:ref:`package-supported cached-sigma recipe <co2-cached-sigma-recipe>`.

Resume from external scientific data
----------------------------------------------

``run_standard`` and ``run_rhime_multisector`` accept ``merged_data`` as a
Python-only handoff. It bypasses OpenGHG acquisition,
checks the recorded scientific facts and single- or multi-sector layout, and
then re-enters the visible recipe at observation selection and retained-population
error preparation::

   result = run_standard(
       config_file="config.ini",
       merged_data=my_merged_data,
   )

The supplied ``RhimeMergedData`` and its xarray or Dask arrays remain borrowed.
Filtering returns retained per-site datasets and aligned ``SiteOptions``; basis
construction and labelled assembly consume those values without mutating the
external object.

Before preparation, ``RhimeMergedData.validate_for_preparation`` checks known
species, domain and acquisition date bounds against the requested run. A
conflict raises before filtering or basis construction. Changed windows,
including narrower ones, are unsupported: acquire data for the new window
instead. Dates are compared as UTC instants; timezone-naive dates mean UTC,
so equivalent timezone spellings are accepted. Reuse keeps the recorded
selectors even when the configuration requests different retrieval selectors;
later priors and sampling choices do not determine acquisition compatibility.
Missing historical facts remain unknown and are not filled from the new
request. Validation neither selects observations nor executes lazy arrays.

Preparation steps pass per-site datasets between them, not ``MergedData``:
``MergedData`` holds acquired data only. When a custom step drops sites, keep
the selectors aligned with ``site_options.retain_sites(tuple(site_data), context=...)``,
as ``filter_observations`` does, and pass the retained ``SiteOptions`` to
``assemble_inputs``. Numerical arrays stay borrowed; the caller owns any files
backing supplied datasets.

Follow the modern preparation spine
------------------------------------

The ordinary runner resolves ``basis_path``, support maps and selected BC basis
files before acquisition. It selects observations, prepares their errors,
then calls the canonical load-or-generate operation with exact paths and
labelled arrays:

.. code-block:: python

   from openghg_inversions.basis import prepare_basis
   from openghg_inversions.basis.support import load_bc_basis
   from openghg_inversions.native_flux import resolve_runtime_flux
   from openghg_inversions.rhime.preparation import (
       acquire_merged_data, filter_observations, prepare_observation_errors,
       resolve_basis_inputs,
   )
   from openghg_inversions.rhime import validate_output_resources

   # config was constructed from external options with RhimeConfig.from_params.
   # For a complete run that will produce outputs:
   validate_output_resources(config.output)
   basis_inputs = resolve_basis_inputs(config.data)
   bc_basis = (
       load_bc_basis(config.data.bc_basis_paths)
       if config.data.use_bc and config.data.bc_basis_paths is not None else None
   )
   merged = acquire_merged_data(config.data)
   filtered, site_options = filter_observations(
       merged.site_data, merged.site_options, filters=config.data.filters,
   )
   filtered = prepare_observation_errors(
       filtered, averaging_error=config.data.averaging_error,
   )
   runtime_flux = resolve_runtime_flux(
       merged.flux_data, split_by_sectors=merged.split_by_sectors,
   )
   weighting_source = config.data.flux_sources[0] if config.data.flux_sources else next(iter(merged.flux_data))
   weighting_flux = merged.flux_data[weighting_source]["flux"]
   basis = prepare_basis(
       site_data=filtered,
       runtime_flux=runtime_flux,
       weighting_flux=weighting_flux,
       domain=config.data.domain,
       start_date=config.data.start_date,
       nbasis=config.data.nbasis,
       basis_algorithm=config.data.basis_algorithm,
       fix_outer_regions=config.data.fix_basis_outer_regions,
       **basis_inputs,
   )
   if config.data.basis_output_path is not None:
       basis.save(config.data.basis_output_path)

A selected ``basis_path`` reuses saved geometry with current merged flux;
otherwise preparation generates a partition from the filtered observations.
The runner optionally saves the returned basis at ``basis_output_path`` and
preserves the existing basis value and its original source provenance. The
requested export path belongs to configuration, not a rewritten scientific value. ``prepare_basis``
accepts no export destination and performs no export write. Sensitivity construction
and input assembly consume the returned ``BasisFunctions``. Pass ``bc_basis`` to
``build_sensitivities``; ``None`` uses NESW without any BC basis-file I/O.
Historical names,
case/glob discovery and filename construction belong only to the temporary
HBMCMC entry route described in :doc:`rhime_configuration`. Modern custom runners
accept explicit selections and forward already resolved paths. These scientific
steps receive resolved values.

``runtime_flux`` is native prior flux, resolved once for this domain/grid
context. It retains all sources and their signed values. ``weighting_flux``
selects only the first requested source for basis design. Non-sectoral source
combination follows the established OpenGHG time alignment and forward-fill
policy; sectoral native flux retains its time/grid compatibility checks.
Observation-aligned ``fp_x_flux`` and ``fp_x_flux_sectoral`` remain in the site
datasets used by ``build_sensitivities``. They are response arrays, not substitutes
for the native prior used in reconstruction. Resolving the native flux once
establishes ownership; it does not imply fewer executed Dask tasks.

Acquire a merged handoff
-------------------------

``SiteOptions`` keeps all observation and footprint selectors aligned to one
site order. Normalize scalar or per-site inputs once, then acquire an in-memory
merged handoff::

   from openghg_inversions.inversion_data import RhimeMergedData, SiteOptions
   from openghg_inversions.rhime import run_standard

   selectors = SiteOptions.from_inputs(
       sites=["TAC", "MHD"], averaging_period="4h", inlet=["185m", "10m"],
   )
   merged = RhimeMergedData.from_options(
       site_options=selectors,
       species="ch4", domain="EUROPE",
       start_date="2020-01-01", end_date="2020-02-01",
       flux_sources=["inventory"],
       obs_store="user", footprint_store="user", flux_store="user",
       bc_store="user",
   )

Use your configured object stores and flux source names. Fresh acquisition may
drop unavailable sites; the returned ``merged.site_options`` retains every
selector in the resulting site order.
Fresh retrieval constructs this dataset handoff directly; the older
tuple-returning API derives its ``fp_all`` mapping from the handoff.

``RhimeMergedData`` stores per-site xarray datasets in ``site_data``,
source-labelled flux datasets in ``flux_data``, optional boundary conditions
in ``boundary_data``, complete ``SiteOptions``, acquisition facts and selected
provenance. Its arrays remain borrowed and may be Dask-backed. It has no
``fp_all`` property. ``AcquisitionFacts`` records known species, domain and
window bounds alongside other retrieval choices. For example,
``AcquisitionFacts(species="ch4", domain="EUROPE")`` makes no claim about dates.
Explicit ``from_legacy_fp_all`` and ``to_legacy_fp_all`` utility functions
in ``openghg_inversions.hbmcmc.legacy_data`` support legacy callers in 0.8.
The entire HBMCMC subpackage, including these utilities, is removed in 0.9;
see :doc:`legacy_and_migration` for replacements and older-release guidance.
They are separate from the modern data model.

Acquisition stays in memory. For a reusable file, prepare the data with the
named stages below and call ``RhimePreparedInputs.save``; the staged
``prepare`` command writes this same durable contract. See :doc:`rhime` for
prepared-input persistence and reuse. Selected provenance
includes each input's available store, UUID and dataversion, plus the
acquisition OpenGHG version and commit. Unavailable identities remain
``unknown``. When several footprint inlets contribute, identifier tuples
retain the contributing input order.

In Python, ``merged.provenance`` is a ``MergedDataProvenance`` value with
``observations``, ``footprints`` and ``flux`` mappings and optional ``boundary``
identity. Each ``InputProvenance`` exposes ``store``, ``uuid`` and
``dataversion``; for example, ``merged.provenance.footprints["TAC"].uuid``.
The software fields are ``openghg_version`` and ``openghg_commit``.
Both classes are available from ``openghg_inversions.inversion_data``.
They describe selected inputs; scientific compatibility remains the
responsibility of ``validate_for_preparation``. Empty or all-unknown provenance
is false in boolean tests; that does not establish data validity. Missing
input identities default to unknown without replacing supplied identities.

Selected ``MergedDataProvenance`` remains on the in-memory merged handoff. It
is not included in saved ``RhimePreparedInputs`` or the staged preparation
manifest. Retain it separately when an audit requires selected input identities;
persistence of acquisition provenance is deferred to
`issue #829 <https://github.com/openghg/openghg_inversions/issues/829>`_.

For the existing tuple-returning retrieval API, use
``openghg_inversions.hbmcmc.legacy_data.retrieve_inversion_data``. The former
``data_processing_surface_notracer`` name forwards the same arguments and
return value and emits ``DeprecationWarning``.
The tuple's flux and boundary wrappers are reconstructed from the modern
datasets and selected source identities. Their metadata no longer includes
arbitrary OpenGHG catalogue fields; scientific dataset attributes remain
available on the datasets.

Change the likelihood with a Python function
--------------------------------------------

For an ordinary likelihood variation, keep RHIME's complete
acquisition-to-output workflow and pass one Python function to ``run_standard``.
The function is deliberately unavailable to INI files: it is
not imported from configuration or stored in a run or model specification.

The complete integration is one named argument:

.. code-block:: python

   from my_project.likelihoods import likelihood_builder
   from openghg_inversions.rhime import run_standard

   result = run_standard(
       config_file="config.ini",
       mismatch_model=None,
       likelihood_builder=likelihood_builder,
   )

RHIME calls the function with explicit keyword arguments while constructing
the PyMC model: the prepared observations, completed forward-model mean,
reported observation error, selected aggregation error, and output dimension.
Pollution-event-only terms remain inside the built-in pollution-event
component. The function adds ``epsilon`` and the canonical observed variable
``y`` to the active model and returns ``y``. There is no framework context or
likelihood-result record to construct.

Options owned only by a custom likelihood can be supplied separately with
``likelihood_kwargs``. RHIME expands that mapping into the callable without
hiding the common scientific arrays in an opaque object::

   result = run_standard(
       config_file="config.ini",
       mismatch_model=None,
       likelihood_builder=likelihood_builder,
       likelihood_kwargs={"degrees_of_freedom": 4.0},
   )

``mismatch_model=None`` explicitly opts out of the built-in selected by the
configuration template. Passing both a custom callable and either built-in
selection is rejected as ambiguous.

RHIME passes these options directly and records a JSON-compatible
representation beside the likelihood identity in result metadata and any
saved inversion output. NumPy and Python dates and times become ISO strings;
timedeltas become strings. A non-empty mapping is rejected when no
``likelihood_builder`` is active.

Editable likelihood
~~~~~~~~~~~~~~~~~~~

The following example replaces RHIME's Gaussian observation distribution with
a Student-t distribution while reusing RHIME's current mean and error
construction:

.. literalinclude:: ../../examples/rhime_customisation/likelihoods.py
   :language: python
   :linenos:

RHIME records the function's module and name, together with its explicit
``likelihood_kwargs``, in the result and in saved output.
It does not copy the Python source code into the output, so a project should
keep the source and environment used for an inversion. Ordinary likelihood
builders retain the canonical ``y`` and ``epsilon`` names used by sampling and
output code.

The example rejects dense and low-rank aggregation covariance because it uses
an independent Student-t distribution. Supporting those aggregation-error
modes would require a multivariate likelihood.

The package-supported scalar-sigma CO2 likelihood is a concrete CO2 model
recipe rather than a Standard RHIME customization. See the :ref:`global
scalar-sigma CO2 workflow
<co2-scalar-sigma-recipe>` for its preparation, cache, units, and runner
contract.

Built-in mismatch equations are direct model components, not examples of this
custom-callback contract. In particular,
``models.additive_sigma.add_additive_sigma_likelihood`` adds an absolute
concentration-scale variance, while ``models.pollution_event`` owns the
pollution-enhancement-scaled equation. Select the absolute additive equation
directly with ``mismatch_model="additive_sigma"`` in ``run_standard`` or
``run_rhime_multisector``; ``sigma_prior``, ``sigma_freq``, and
``sigma_per_site`` then configure its scale. Additive sigma does not calculate a
``min_error`` floor unless
``use_minimum_error_floor=True`` is selected explicitly. A custom likelihood
loads only ``mf`` and ``mf_error`` from the universal observation inputs and
does not require ``min_error``. Built-in floor choices are model settings;
named residual and percentile estimates use observations retained after invalid
rows are dropped. Preparation retains the raw operands for compatible model reuse.

Legacy ``run_hbmcmc`` additive configuration is translated at that script's
entry point into the same explicit likelihood settings. Parameter resolution
stores built-in mismatch science in ``RhimeModelSpec`` before the standard or
multisector recipe is called; the recipes themselves select no default.

The model-owned
``models.fixed_ou.add_fixed_ou_gaussian_likelihood`` adds a fixed-timescale,
within-site Ornstein--Uhlenbeck mismatch covariance. For example::

   from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
   from openghg_inversions.rhime import run_standard

   result = run_standard(
       ...,
       likelihood_builder=add_fixed_ou_gaussian_likelihood,
       likelihood_kwargs={
           "tau_hours": 5.0,
           "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.75},
       },
   )

The residual covariance is
``R = F F^T + diag(d) + direct_sum_s(a_s^2 T_s(tau_s))``, where
``T_s[i, j] = exp(-abs(t_i - t_j) / tau_s)``. The direct sum gives zero
cross-site OU covariance. ``F F^T + diag(d)`` is the selected fixed
aggregation covariance plus reported observation-error variance; this
component does not apply the historical ``min_error`` floor.

``tau_hours`` may instead be an exact mapping from retained site labels to
fixed positive timescales. Pass ``fixed_site_amplitudes`` as a scalar or exact
site mapping to use known amplitudes instead of the inferred prior. An inferred
amplitude requires an explicit ``site_amplitude_prior`` in the same
concentration units as the observations; no fixed default is assumed. Tau has
units of hours. Observation rows may be interleaved or nonmonotonic in time;
the component preserves their order and sets cross-site OU covariance exactly
to zero.

The component evaluates low-rank aggregation covariance with a fixed-OU
generalized-eigen and Woodbury method without forming a dense observation
covariance. Dense aggregation input is a full-rank fallback. It currently
requires every generalized base-plus-OU mode variance to be strictly positive.
An exact zero mode is rejected before applying the low-rank factor, even when
that factor would make a materialized dense covariance positive definite; a
positive OU amplitude can lift a zero base mode. The component currently
requires PyMC's native No-U-Turn Sampler (NUTS) backend. Sampled tau is a
separate extension; the package-supported cached sampler is the matched recipe
described in :ref:`co2-cached-sigma-recipe`.

Optional project CLI
~~~~~~~~~~~~~~~~~~~~

The short wrapper below packages the same one-call integration as a reusable
Python function and command-line entry point:

.. literalinclude:: ../../examples/rhime_customisation/run_with_likelihood.py
   :language: python
   :linenos:

Run it with a normal RHIME configuration and optional JSON overrides::

   python -m package_name.run_with_likelihood config.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

``run_multisector`` accepts the same Python-only likelihood-builder contract,
including with ``prepared=...``. Saved sampling through
``multisector_stages.sample`` uses that contract too. The multisector model
retains sector flux components and roles, then
passes their combined observation mean to the likelihood, so no special case
or semantic compromise is required.

Use the seam from a generated project
-------------------------------------

Create a normal downstream package with the current `OpenGHG project
cookiecutter <https://github.com/openghg/openghg-project-cookiecutter>`_::

   uvx cookiecutter gh:openghg/openghg-project-cookiecutter

The template already declares ``openghg`` and ``openghg_inversions`` as
dependencies and creates a ``src`` layout. Add the project-owned files beside
the generated package's existing modules, without copying any OpenGHG
Inversions implementation::

   src/my_inversion/
     __init__.py
     likelihoods.py
     runner.py
   tests/
     test_runner.py

Put only the scientific change in ``likelihoods.py``:

.. literalinclude:: ../../examples/rhime_cookiecutter/my_inversion/likelihoods.py
   :language: python
   :linenos:

Keep project-level invocation in ``runner.py``. Its one library call owns no
retrieval, filtering, basis, input assembly, sampling, predictive selection,
or output implementation:

.. literalinclude:: ../../examples/rhime_cookiecutter/my_inversion/runner.py
   :language: python
   :linenos:

After ``uv sync --extra dev``, run the module with a normal RHIME INI file::

   uv run python -m my_inversion.runner inversion.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

An optional console command can point directly at the same ``main`` function.
Add this table to the generated ``pyproject.toml``::

   [project.scripts]
   my-inversion = "my_inversion.runner:main"

Then the equivalent command is::

   uv run my-inversion inversion.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

The generated runner uses documented names from ``openghg_inversions.rhime``.
Its likelihood module imports reusable components from their documented owner
modules: ``models.pollution_event``, ``observation_error``, and ``sigma``.
The dependency direction is therefore the generated project to OpenGHG
Inversions; OpenGHG Inversions does not import the consumer package. Pin the
release or commit used for a scientific run in the downstream project's
lockfile. An optional RHIME recipe or generated-project CI in the generic
cookiecutter would be a separate cross-repository change and is not required
for this workflow.

Copy the complete runner
------------------------

The copied runner below is an advanced, version-coupled escape hatch for
scientific changes beyond the likelihood seam. It makes the major stages
visible while importing their implementations from the supported public RHIME
API. Prefer ``run_standard(prepared=prepared, config=config)`` when replaying
prepared inputs or starting from a different preparation graph while retaining
the standard model operations.

The deliberate change is the likelihood passed to
``build_standard_model`` after explicit numerical materialization: the example
selects the same project-owned
Student-t builder as the preferred form. Acquisition, filtering, basis
construction, labelled input assembly, conversion of delayed arrays for PyMC,
sampling, predictive selection, filenames, and output handling remain
library-owned.

In a project created with the `OpenGHG project cookiecutter
<https://github.com/openghg/openghg-project-cookiecutter>`_, copy the preferred
modules to ``src/<package_name>/likelihoods.py`` and
``src/<package_name>/run_with_likelihood.py``. Copy the complete module below
to ``src/<package_name>/rhime_runner.py`` only when the project needs to own a
deeper orchestration change. Import scientific stage implementations from
``openghg_inversions.rhime`` rather than copying those implementations.

Run it with a normal RHIME configuration and optional overrides::

   python -m package_name.rhime_runner config.ini \
       --start-date 2020-01-01 --end-date 2020-02-01 \
       --output-path outputs --draws 1000 --tune 1000 --chains 4

Less common Python/config options can be supplied as a JSON object::

   python -m package_name.rhime_runner config.ini \
       --kwargs '{"output_format": "inv_out"}'

The test suite imports all three sources, exercises both runners, and validates
the likelihood contract directly, so the documentation and runnable examples
cannot drift apart.

.. literalinclude:: ../../examples/rhime_customisation/runner.py
   :language: python
   :linenos:

Compose a custom basis stage
----------------------------

The second complete runner replaces one call in the preparation spine:
``build_project_basis`` replaces ``openghg_inversions.basis.prepare_basis``. The project function
composes public basis primitives instead of selecting a built-in basis
algorithm. It:

* derives and normalises a two-dimensional weight field with
  ``basis_weights_from_data(site_data, weighting_flux)``;
* receives a class map resolved before acquisition from an exact ``landsea_path``
  or explicitly selected packaged support map, then reduces positive codes to
  ``land`` and the remaining cells to ``ocean``;
* creates physical north-south/east-west coordinates with
  ``LatLonGridGeometry``;
* uses balanced inertial splits, decomposes every proposed child into
  four-neighbour connected components, and rejects splits whose children
  exceed the configured PCA eccentricity guard;
* generates class-safe labels with ``region_constrained_basis``; and
* wraps the flat labels and current run flux in retained ``BasisFunctions``
  with ``basis_functions_from_flat_basis(flux=runtime_flux, ...)``.

The weight field uses the first requested source. The retained basis attaches
all runtime flux sources, combining them for an ordinary inversion or stacking
them in mapping order when ``split_by_sectors=True``. Neither handoff needs the
legacy merged dictionary.

The nested split strategy follows the latest selected-country guarded-basis
variant in the ``verification-games`` project. That variant gives the UK,
Ireland, France, Germany, Italy, Belgium, and the Netherlands separate classes,
while remaining land and ocean form two more. This smaller example deliberately
uses the land/ocean class variant so the composition stays readable. To adopt
the selected-country policy, replace ``_land_ocean_classes`` with a project
function that maps the loaded integer country codes or country names to those
classes. Keep this classification outside OpenGHG Inversions unless it becomes
a broadly supported policy.

The weighting is deliberately not identical to that later verification-games
preparation step. Verification-games sums absolute cached ``fp_x_flux`` after
those sensitivities exist. At this earlier visible-runner basis boundary they
have not been constructed yet, so the example uses the public
``basis_weights_from_data`` field while preserving the guarded split strategy
and class composition.

The flat labels and retained object record namespaced provenance for the class
policy, normalised weight source, connected balanced-inertial strategy,
connectivity, and eccentricity threshold. Those fields travel with a saved
``BasisFunctions`` artifact and make the project choice inspectable later.

Acquisition, filtering, sensitivity construction, labelled input assembly,
the standard likelihood and model, conversion of delayed arrays for PyMC,
sampling, predictive selection, filenames, and output handling remain
library-owned.

The project owns the scientific validity of its classification, coverage,
region count, eccentricity threshold, and split policy. ``BasisFunctions`` and
the unchanged downstream stages validate the grid, coordinates, sources, site
alignment, state layout, and model inputs they consume; they cannot certify
that the project's scientific partition is appropriate. Malformed artifacts
fail while loading; structurally valid artifacts fail later if they violate a
downstream alignment contract.

The example exposes ``project_basis_path`` separately from standard RHIME
options. Point it at a self-contained ``.nc`` or ``.zarr`` artifact written by
``BasisFunctions.save`` to bypass fitting::

   python -m package_name.custom_basis_runner config.ini \
       --project-basis-path cache/project-basis.zarr

The eccentricity guard is also a visible project-owned option, rather than an
opaque library default. It defaults to ``10`` and is removed before standard
RHIME option resolution::

   python -m package_name.custom_basis_runner config.ini \
       --max-child-pca-eccentricity 10

``BasisFunctions.reuse_operator(project_basis_path, flux=runtime_flux)``
loads the saved operator and metadata, selects any source-specific operator in
current source order, and binds the current native prior. It closes the artifact
without loading the saved flux payload. The standard ``prepare_basis`` operation
uses the same rule for ``basis_path``. For reconstruction of a saved result,
``BasisFunctions.load`` instead restores the complete retained basis, including
its saved flux, without acquisition.

Without an artifact, the basis-building function computes the arrays needed by
the splitting algorithm. Before that point, the filtered xarray objects may
still defer their calculations with Dask. The function derives and normalises
the two-dimensional weights, uses the class map resolved before acquisition,
constructs the
geometry, and creates the region labels. It then combines the labels with the
current run's flux in a ``BasisFunctions`` object. That object flows unchanged
through sensitivity construction and labelled assembly. The concrete recipe's
adjacent ``standard_model_input_names`` declaration selects the arrays it
needs. ``materialize_standard_inputs`` computes those arrays and enabled priors
together, then resolves model-specific BC periods and the likelihood's floor
before graph construction. Project-owned lazy extensions in
``RhimePreparedInputs`` remain untouched. The generic ``materialize_pymc_inputs``
only executes selected arrays; it does not resolve those scientific choices.

The customisation is concentrated in ``_guarded_basis``. The runner's one
deliberate substitution is marked by an inline comment where
``build_project_basis`` replaces the standard ``openghg_inversions.basis.prepare_basis`` call. The
complete source remains below because the test suite executes the same file
that the documentation displays.

Copy ``examples/rhime_customisation/custom_basis_runner.py`` to
``src/<package_name>/custom_basis_runner.py`` and keep the project basis rule
beside the copied orchestration spine. This source is imported and executed by
integration tests and rendered here, so the documentation and runnable example
cannot drift apart.

This guarded composition is intentionally project-specific and is not yet a
common, stable strategy that warrants another lower-ceremony ``run_standard``
option.

.. literalinclude:: ../../examples/rhime_customisation/custom_basis_runner.py
   :language: python
   :linenos:
