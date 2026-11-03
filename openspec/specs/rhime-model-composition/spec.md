# RHIME model composition

> Status: Current supported behavior at `42a32f62`; documentation baseline,
> no runtime change. This specification covers the named interfaces below.

## Purpose

Define the current labelled PyMC composition boundary and the standard and
multisector custom-likelihood seam. Scientific covariance support remains a
property of each concrete likelihood and recipe.

## Requirements

### Requirement: Registered scientific coordinates

`registered_model()` SHALL retain scientific coordinates in a `CoordRegistry`
and supply PyMC range coordinates. `add_coords()` SHALL require a registry,
accept identical coordinates subject to the missing-value limitation below,
and reject conflicting values or lengths.
Auxiliaries SHALL remain metadata through `registered_model(coords=xr.Coordinates)`
or `add_coords(..., model_dims=(...))`. Without `model_dims`, `add_coords()`
considers all supplied coordinate mapping keys as model dimensions.

Known implementation limitation: repeating an unchanged auxiliary coordinate
containing `NaN` can raise `ValueError` because the registry's array comparison
treats missing values as unequal. This false conflict is tracked in
[#856](https://github.com/openghg/openghg_inversions/issues/856); accepting
unchanged missing auxiliary metadata remains follow-up work, not current
supported behavior.

#### Scenario: Register a measurement MultiIndex

- **WHEN** `registered_model(coords=...)` receives xarray coordinates containing
  a measurement MultiIndex and its site/time level coordinates
- **THEN** PyMC receives a range coordinate for the measurement dimension
- **AND** the registry retains the original MultiIndex and its auxiliary
  coordinates without introducing site/time model dimensions

#### Scenario: Register auxiliary coordinates with selected model dimensions

- **WHEN** a registered model receives those measurement coordinates through
  `add_coords(coords, model_dims=("nmeasure",))`
- **THEN** the registry retains the measurement index and site/time auxiliaries
- **AND** site/time are not registered as independent PyMC dimensions

#### Scenario: Reject an unregistered or conflicting composition

- **WHEN** `add_coords()` is called in a model without a registry
- **THEN** it raises a `RuntimeError` identifying `registered_model()`
- **AND WHEN** an existing registry receives conflicting coordinates
- **THEN** registration raises a `ValueError`

### Requirement: Explicit custom-likelihood scientific inputs

The standard and multisector custom-likelihood seams SHALL call the supplied
callable with `mean`, `observations`, `observation_error`, and `output_dim`.
The mean SHALL include the recipe's applicable pollution and
baseline contributions. The seam SHALL keep built-in-specific inputs such as
`minimum_error`, `pollution_mean`, and `pollution_event_baseline` outside this
common argument set.

#### Scenario: Replace the likelihood in a concrete recipe

- **WHEN** a standard or multisector model is built with a custom likelihood
- **THEN** the callable receives `mean`, `observations`, `observation_error`,
  and `output_dim` exactly once, with the completed forward mean
- **AND** it receives additional options only through explicit
  `likelihood_kwargs`

### Requirement: Canonical custom-likelihood result

The callable SHALL return a PyTensor `TensorVariable` named `y` and create the
canonical model variables `y` and `epsilon`. The seam SHALL reject a non-tensor
return with `TypeError`, or a wrong return name or missing canonical variables
with `ValueError`, before sampling. These checks do not establish that the
returned tensor is the registered observed variable, or validate the scientific
meaning and dimensions of `epsilon`. The custom callable owns those guarantees.

#### Scenario: Reject a noncanonical custom likelihood

- **WHEN** a custom likelihood returns a value other than a PyTensor variable
  named `y`, or leaves `y` or `epsilon` absent from the model
- **THEN** the seam raises the corresponding type or value error before sampling

### Requirement: One owner of model and likelihood construction

Standard and multisector requests SHALL reject a custom `likelihood_builder`
combined with configured built-in mismatch/likelihood settings before acquisition
in ordinary runners or materialization in prepared-input execution. Their eager
builders `build_standard_model` and `build_multisector_model` SHALL take numerical
inputs, `RhimeModelSpec` and materialized prior parameters without output
configuration and SHALL NOT accept `model_builder`.

Materialization SHALL be an explicit runner/caller operation. Selected related
inputs and enabled priors SHALL preserve shared Dask execution at that named
boundary. Construction SHALL NOT compute lazy payloads or require a separate
lazy/eager builder variant or saved-workflow phase. The optional `sigma_alignment`
argument SHALL remain available on the canonical eager builders.

`run_standard(prepared=..., config=...)` and
`run_multisector(prepared=..., config=...)` SHALL use the same scientific
operations as acquired-data and saved execution without a complete-model-builder
hook. Whole-model customization SHALL directly compose scientific functions with
explicit inputs, materialization, sampling and compatible output operations.

#### Scenario: Reject competing likelihood selections

- **WHEN** an ordinary or prepared-input standard/multisector run supplies a custom likelihood alongside a configured built-in selection
- **THEN** it raises `ValueError` before acquisition or model-input materialization

#### Scenario: Reject a complete-model runner hook

- **WHEN** a standard or multisector runner receives `model_builder`, with or without `likelihood_builder`
- **THEN** it rejects that unsupported option before constructing a model or invoking a likelihood
- **AND** whole-model callers can directly compose the supported scientific operations

#### Scenario: Construct an eager multisector model

- **WHEN** a caller supplies eager numerical inputs, resolved `RhimeModelSpec` and materialized priors to `build_multisector_model`
- **THEN** it constructs the scientific model without output configuration or an intermediate compatibility builder
- **AND** supported explicit `sigma_alignment` is preserved

### Requirement: Explicit custom options and recorded provenance

The standard and multisector runners SHALL reject nonempty `likelihood_kwargs`
without a `likelihood_builder` before configuration processing. Supported
custom-likelihood result construction SHALL record module/qualified-name
metadata and the supplied options as structured metadata. The prepared-input
runner's saved inversion output SHALL retain those options.

The metadata describes the runtime customization; it is not a reconstructable
callable identity. Callable instances fall back to their class's module and
qualified name. A `functools.partial` records `functools.partial`, without its
wrapped function or bound arguments. Only explicitly supplied `likelihood_kwargs`
are recorded as options.

#### Scenario: Supply options without their callable owner

- **WHEN** an ordinary standard or multisector request supplies nonempty
  `likelihood_kwargs` without a custom callable
- **THEN** it raises a `ValueError` before reading configuration or acquiring data

#### Scenario: Persist a supported custom likelihood

- **WHEN** a prepared-input run uses a custom likelihood and JSON-compatible
  explicit options and saves its inversion output
- **THEN** the result and output metadata retain the callable's
  module/qualified-name metadata and explicit options
- **AND** reloading the output preserves the options

### Requirement: Declared likelihood and prior semantics

Modern recipes SHALL apply their declared scientific likelihood and prior
semantics. Standard offsets SHALL retain their full-baseline semantics.
Pollution-event likelihood construction SHALL NOT substitute boundary-only
historical behavior. Sigma priors SHALL participate in the selected model rather
than create disconnected variables; additive fixed-error construction SHALL NOT
introduce a minimum-error floor. Custom likelihoods SHALL retain their explicit
scientific ownership under the common composition seam.

#### Scenario: Construct the selected scientific model

- **WHEN** a modern recipe constructs a supported built-in likelihood and its priors
- **THEN** it follows the selected scientific equations and full-baseline offset semantics
- **AND** it does not add disconnected sigma variables or an additive fixed-error floor

## Source and test evidence

- `openghg_inversions/models/coords.py`: `registered_model`, `add_coords`, and
  `CoordRegistry`; `tests/models/test_coords.py`:
  `test_coord_registry_repeated_registration_and_conflict`,
  `test_add_coords_requires_registered_model`,
  `test_registered_model_keeps_xarray_auxiliary_coordinates_out_of_pymc_dims`, and
  `test_add_coords_preserves_auxiliary_coords_for_model_dims`.
- `openghg_inversions/rhime/_model_building.py`: `_call_custom_likelihood`;
  `tests/test_rhime.py`:
  `test_recipe_passes_completed_forward_mean_to_custom_likelihood`,
  `test_custom_likelihood_builders_use_the_common_contract`,
  `test_recipe_expands_only_explicit_custom_likelihood_options`, and
  `test_run_rhime_rejects_noncanonical_custom_likelihood_before_sampling`.
- `openghg_inversions/rhime/builders.py`: `callable_metadata`;
  `openghg_inversions/rhime/outputs.py`: `annotate_likelihood_trace`;
  `tests/test_rhime.py`:
  `test_ordinary_runners_reject_orphaned_likelihood_options_before_config` and
  `test_likelihood_builder_provenance_is_saved_with_result_metadata`.
- `openghg_inversions/rhime/standard.py` and `multisector.py`:
  extension-point validation; `tests/test_rhime.py`:
  `test_ordinary_runners_reject_custom_likelihood_with_configured_mismatch`,
  `test_prepared_runner_rejects_custom_likelihood_with_builtin_selection_before_materialization`,
  `test_standard_prepared_run_rejects_complete_model_builder`, and
  `test_multisector_builder_accepts_canonical_inputs_and_resolved_spec`.
