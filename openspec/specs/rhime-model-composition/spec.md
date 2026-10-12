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
accept identical coordinates, and reject conflicting values or lengths.
Auxiliaries SHALL remain metadata through `registered_model(coords=xr.Coordinates)`
or `add_coords(..., model_dims=(...))`. Without `model_dims`, `add_coords()`
considers all supplied coordinate mapping keys as model dimensions.

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
callable with `mean`, `observations`, `observation_error`, `aggregation_error`,
and `output_dim`. The mean SHALL include the recipe's applicable pollution and
baseline contributions. The seam SHALL keep built-in-specific inputs such as
`minimum_error`, `pollution_mean`, and `pollution_event_baseline` outside this
common argument set.

#### Scenario: Replace the likelihood in a concrete recipe

- **WHEN** a standard or multisector model is built with a custom likelihood
- **THEN** the callable receives the completed forward mean and the common
  observation arguments exactly once
- **AND** it receives additional options only through explicit
  `likelihood_kwargs`

### Requirement: Canonical custom-likelihood result

The callable SHALL return a PyTensor `TensorVariable` named `y` and create the
canonical model variables `y` and `epsilon`. The seam SHALL reject a non-tensor
return with `TypeError`, or a wrong return name or missing canonical variables
with `ValueError`, before sampling.

#### Scenario: Reject a noncanonical custom likelihood

- **WHEN** a custom likelihood returns a value other than a PyTensor variable
  named `y`, or leaves `y` or `epsilon` absent from the model
- **THEN** the seam raises the corresponding type or value error before sampling

### Requirement: One owner of model and likelihood construction

Standard and multisector requests SHALL reject a custom `likelihood_builder`
combined with configured built-in mismatch/likelihood settings before acquisition
in ordinary runners or materialization in the prepared-input runner. Their
public build stages SHALL reject simultaneous `model_builder` and
`likelihood_builder` inputs before either callable runs.

#### Scenario: Reject competing likelihood selections

- **WHEN** an ordinary or prepared-input standard/multisector run supplies a
  custom likelihood alongside a configured built-in selection
- **THEN** it raises `ValueError` before acquisition or model-input materialization

#### Scenario: Reject competing builder ownership

- **WHEN** a standard or multisector build stage receives both a complete-model
  builder and a likelihood builder
- **THEN** it raises `ValueError` before either builder executes

### Requirement: Explicit custom options and recorded provenance

The standard and multisector runners SHALL reject nonempty `likelihood_kwargs`
without a `likelihood_builder` before configuration processing. Supported
custom-likelihood result construction SHALL record the callable's module and
qualified name and the supplied options as structured metadata. The
prepared-input runner's saved inversion output SHALL retain those options.

#### Scenario: Supply options without their callable owner

- **WHEN** an ordinary standard or multisector request supplies nonempty
  `likelihood_kwargs` without a custom callable
- **THEN** it raises a `ValueError` before reading configuration or acquiring data

#### Scenario: Persist a supported custom likelihood

- **WHEN** a prepared-input run uses a custom likelihood and JSON-compatible
  explicit options and saves its inversion output
- **THEN** the result and output metadata identify the callable and options
- **AND** reloading the output preserves the options

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
- `openghg_inversions/rhime/standard.py`, `multisector.py`, and `prepared.py`:
  extension-point validation; `tests/test_rhime.py`:
  `test_ordinary_runners_reject_custom_likelihood_with_configured_mismatch`,
  `test_prepared_runner_rejects_custom_likelihood_with_builtin_selection_before_materialization`,
  and `test_public_build_stages_reject_simultaneous_model_and_likelihood_builders`.
