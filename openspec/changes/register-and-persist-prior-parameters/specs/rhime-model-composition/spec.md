# RHIME model composition delta

Registration and persistence work for `Interpolated` priors is outside this
amendment; its existing runtime handling is not removed.

## MODIFIED Requirements

### Requirement: Registered scientific coordinates

`registered_model()` SHALL retain scientific coordinates in a `CoordRegistry`
and supply PyMC range coordinates. `add_coords()` SHALL require a registry,
accept identical coordinates subject to the missing-value limitation below,
and reject conflicting values or lengths.
Auxiliaries SHALL remain metadata through `registered_model(coords=xr.Coordinates)`
or `add_coords(..., model_dims=(...))`. Without `model_dims`, `add_coords()`
considers all supplied coordinate mapping keys as model dimensions.

The registry SHALL own scientific dimension indexes, auxiliary-coordinate
information, the corresponding PyMC range-coordinate mapping, and restoration
into inference data. Registered model data SHALL own fixed numerical array input
values; constructed-prior metadata SHALL retain numerical scalar parameters
inline. The coordinate registry SHALL NOT become a second numerical-data store.
Scientific coordinates SHALL be restored on matching inference-data dimensions,
including those of registered prior parameters in `constant_data`.

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

#### Scenario: Restore coordinates of registered prior data

- **WHEN** sampling a registered model produces prior parameters in `constant_data`
- **THEN** their dimensions carry the corresponding scientific indexes and
  applicable auxiliary coordinates rather than PyMC range coordinates
- **AND** auxiliary coordinates do not become additional model dimensions

## ADDED Requirements

### Requirement: Fixed prior inputs retain their scientific meaning

Fixed numerical array prior parameters consumed by registered RHIME construction
SHALL enter through the existing model-data boundary. Numerical scalar parameters
SHALL remain ordinary scalar arguments and be retained inline in constructed-prior
metadata, without requiring named model-data variables. Shared prior construction
SHALL apply this policy consistently for ordinary and reparameterized distributions.
Random variables and derived symbolic expressions SHALL remain graph expressions,
including scalar-shaped expressions. Their symbolic nature SHALL be recognized
before scalar/array coercion; registration or metadata capture SHALL NOT evaluate
a stochastic parameter into fixed data.

Parameter registration SHALL preserve dimension and coordinate information
before labels are discarded. Labelled parameters SHALL align to their owning
component's canonical scientific coordinates, including supported active-state
selection. Invalid labels or incompatible dimensions SHALL fail at the owning
alignment boundary before distribution construction. Positional state arrays
SHALL retain their documented canonical-order interpretation.

Each parameter SHALL retain its own dimensions and scalar broadcasting
semantics. Numerical scalars and zero-dimensional numerical values SHALL be
distinguished from arrays with dimensions: a one-dimensional array of length one
SHALL remain an array. Scalars SHALL NOT be expanded into stored arrays matching
the random variable's shape. Parameter dimensions SHALL NOT be inferred solely
from the random variable's dimensions.

#### Scenario: Ordinary prior with scalar and labelled parameters

- **WHEN** a Normal state prior receives a labelled `mu` and scalar `sigma`
- **THEN** `mu` is registered array data in canonical state order
- **AND** `sigma` remains a scalar argument, broadcasts to the state variable's
  shape and is retained inline in constructed-prior metadata
- **AND** saving and reloading retains `mu`'s values and scientific coordinates
  with its metadata reference resolved correctly

#### Scenario: Scalar-only prior

- **WHEN** a constructed prior uses only numerical scalar parameters
- **THEN** its resolved parameters are retained inline in saved prior metadata
- **AND** no prior-parameter data variables are added to `constant_data` for those
  scalars

#### Scenario: Length-one array retains its dimension

- **WHEN** a prior parameter is a labelled one-dimensional array of length one
- **THEN** registration and persistence retain its axis and scientific coordinate
- **AND** the parameter is not squeezed into inline scalar metadata

#### Scenario: Align and select active state parameters

- **WHEN** a supported state prior receives parameters in reversed label order
  and only a subset of states is active
- **THEN** its active distribution uses the values associated with those active
  scientific labels in canonical order
- **AND** retained prior data and coordinates preserve the selection's meaning

#### Scenario: Reject incompatible labelled state parameters

- **WHEN** supplied state parameters have missing, extra or duplicate state
  labels, or dimensions incompatible with the owning component's contract
- **THEN** alignment reports the incompatibility before constructing the prior
  rather than silently using positional values or filling missing parameters

#### Scenario: Preserve a stochastic parameter

- **WHEN** a prior parameter is a random variable or an expression depending on
  one, including a scalar-shaped parameter
- **THEN** the distribution retains that symbolic dependency
- **AND** registration does not replace it with an evaluated numerical value

### Requirement: Prior records distinguish requested and effective parameterization

Saved prior metadata SHALL be derived only from priors actually used by the
constructed model, rather than all prior settings in the request. Unused priors,
including those configured for disabled components, SHALL produce neither saved
prior metadata nor parameter references. Their lazy numerical inputs SHALL NOT
be materialized for registration or metadata writing.

Prior metadata SHALL identify the distribution, supplied parameterization and
reparameterization policy. It SHALL capture resolved scalar values, including
applicable resolved defaults, and associate array parameters with registered data
at construction rather than later rediscovering them from the original request.
Derived effective parameters SHALL remain identifiable from the retained
inputs and the declared conversion, without requiring duplicate numerical
payloads or serialization of a general expression graph.

For lognormal arithmetic-moment input, numerical scalar `mean` and `stdev` SHALL
be retained inline and array moments SHALL be registered once. Effective `mu`
and `sigma` SHALL be derived through one shared symbolic conversion in model
construction, accepting scalar constants, registered arrays, symbolic parameters
and mixed inputs. Ordinary and reparameterized lognormal construction SHALL use
that same conversion. Scalar constants SHALL NOT require model-data registration
to participate in it. Directly supplied `mu` and `sigma` SHALL remain
distinguishable from arithmetic moments. Reparameterization SHALL retain the
scientific variable and its declared relationship to the latent variable.

#### Scenario: Arithmetic moments in either lognormal construction

- **WHEN** a lognormal prior receives fixed `mean` and `stdev`, with or without
  reparameterization
- **THEN** constructed-prior metadata retains scalar moments inline and references
  registered array moments with their scientific labels
- **AND** the effective distribution has those requested moments
- **AND** recording the conversion does not require storing duplicate `mu` and
  `sigma` arrays

#### Scenario: Direct lognormal parameters

- **WHEN** a lognormal prior receives `mu` and `sigma` directly
- **THEN** scalar parameters are retained inline and array parameters are
  registered under that parameterization
- **AND** metadata does not misdescribe them as arithmetic `mean` and `stdev`

#### Scenario: Disabled component with an unused lazy prior

- **WHEN** a model is constructed with `use_bc=False` and a configured lazy
  `bc_prior`, or with offsets disabled and a configured lazy `offset_prior`,
  then sampled and saved
- **THEN** saving succeeds without saved prior metadata or parameter references
  for the disabled component
- **AND** the unused prior's lazy graph is never executed

### Requirement: Constructed prior inputs survive persistence without recomputation

Caller arrays SHALL be borrowed and potentially lazy. Related lazy inputs,
including supported lazy zero-dimensional numerical parameters and required
auxiliary coordinates, SHALL preserve shared execution graphs until materialized
together at the explicit model-construction boundary. Enabled prior inputs MAY
join this computation before sensitivity evaluation
identifies exact-zero columns and fixes their states instead of sampling them,
even if that evaluation makes a prior unnecessary. Disabled components SHALL be
excluded before this boundary. Only actually constructed priors SHALL be
recorded. Alignment and materialization SHALL NOT mutate caller arrays or prior
mappings.

Sampling and the existing inference-data persistence path SHALL retain the actual
registered numerical prior arrays in `constant_data`, together with scientific
coordinates. Resolved scalar values SHALL be captured at construction and saved
inline in prior metadata. Model-record and output-metadata writing SHALL use
those captured scalars and registered array references without executing the
original caller inputs again. Numerical scalar metadata SHALL be permitted in
JSON; numerical array payloads SHALL NOT be embedded there. Dimensions, shape or
a lazy marker SHALL NOT substitute for retained array values.

The owning writer SHALL reject a record whose required prior-array references
cannot be satisfied before marking the saved handoff complete. The owning
reader SHALL resolve those references from saved inference data and report
missing required array values clearly before product writes, without rebuilding
a model or importing recorded callables. Scalar constants SHALL require no data
lookup, and unused configured priors SHALL generate no references. Full,
prepared-input and saved standard execution SHALL use the same corrected
model-construction and recording path.

#### Scenario: Sample, save and reload lazy labelled prior inputs

- **WHEN** a real registered model with Dask-backed labelled array-valued prior
  parameters is sampled, saved and reloaded
- **THEN** reloaded `constant_data` retains the numerical parameter values,
  scientific indexes and applicable auxiliary coordinates
- **AND** model metadata associates the saved values with the prior's
  parameterization and selection rather than containing shape-only placeholders

#### Scenario: Shared lazy inputs execute only at construction

- **WHEN** related prior parameters, including a supported lazy zero-dimensional
  numerical parameter, share a lazy dependency and the completed run's trace and
  metadata are subsequently saved
- **THEN** construction materializes the related parameters together, executing
  that shared dependency once
- **AND** metadata saving does not execute the caller's dependency again
- **AND** the lazy scalar's resolved value is retained inline without a model-data
  lookup
- **AND** caller arrays, labels, lazy backing and prior mappings remain unchanged

#### Scenario: Missing registered parameter values

- **WHEN** a saved model record references a required prior array absent from its trace
- **THEN** its owning writer refuses completion, or its owning reader reports
  the missing reference before product writes
- **AND** it does not recover values from the original lazy inputs, JSON array
  payloads or a rebuilt model

#### Scenario: Equivalent full, prepared-input and saved standard construction

- **WHEN** full, prepared-input and saved standard routes consume equivalent
  prepared inputs and prior choices
- **THEN** all construct and record their priors through the same shared path
- **AND** scalar metadata, aligned array inputs and prior densities agree within
  numerical tolerance
