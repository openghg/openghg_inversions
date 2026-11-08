# Explicit data contracts

## Purpose

Make the scientific guarantees expected of labelled data explicit in static
types and validation, while retaining ordinary xarray containers and operations.

## ADDED Requirements

### Requirement: Contract types distinguish established guarantees

An xarray data contract SHALL have a named static subtype of `DataArray`,
`Dataset` or `DataTree`, with its description and validation colocated.

#### Scenario: Check a typed handoff

- **WHEN** a function accepts a contract type
- **THEN** a type checker rejects a plain base xarray value passed to it
- **AND** accepts the validator's return value and a package producer's declared return value
- **AND** accepts the contract value wherever the base xarray type is expected

### Requirement: Contracts document structural and numerical guarantees

The contract type's docstring SHALL be the canonical description of the
guarantees consumers may assume: applicable variables, dimensions, coordinates,
units and relationships, and any numerical properties with their tolerance
policies. Producer and input-boundary documentation SHALL explain how their
operations establish those guarantees from their own preconditions. Each
documented guarantee SHALL be established before code relies on it.

Consumers SHALL be able to interpret the guarantees from the contract and the
value without access to historical validator arguments. Parameter-dependent
roles SHALL be recoverable through a fixed convention or information already
carried by the value.

#### Scenario: Interpret a parameterized validation result

- **WHEN** validation accepts a configurable observation-dimension name
- **THEN** the contract specifies how a consumer identifies that dimension from the value or a fixed convention
- **AND** the consumer does not need the earlier validation call's arguments to recover its meaning

### Requirement: Contract values retain ordinary runtime representations

Establishing a contract SHALL retain the ordinary xarray runtime object without
instantiating the contract subtype or changing the supplied object's class.
A contract SHALL provide class-level `validate` as a staticmethod or classmethod.
Other class-level operations MAY be provided; the type SHALL NOT promise new
instance methods or properties absent from the ordinary runtime container.

#### Scenario: Use validated data with xarray

- **WHEN** an ordinary xarray object acquires a contract annotation
- **THEN** its runtime class remains unchanged
- **AND** consumers use its existing xarray instance API and class-level contract operations

### Requirement: Validation preserves the supplied object

A contract validator SHALL check the structural guarantees of independently
supplied data and either return the identical object with the contract's static
annotation or report the violated requirement, identifying the affected variable, dimension, coordinate
or scientific relationship.

Validation SHALL NOT normalize, mutate, copy or wrap supplied data. Required
normalization SHALL be an explicit preparation operation.

#### Scenario: Validate independently supplied data

- **WHEN** supplied data satisfies the contract's structural guarantees
- **THEN** validation returns the identical object with a narrowed static type
- **AND** its values, coordinates, attributes and array backing are unchanged

#### Scenario: Reject an invalid layout

- **WHEN** supplied data violates a required structural relationship
- **THEN** the error names the affected elements and relationship
- **AND** validation does not silently repair the data

### Requirement: Trusted producers and transformations establish their outputs

A package function constructing data satisfying a contract SHALL declare that
contract in its return type. A transformation returning a contract type SHALL
preserve or establish its guarantees through validation or known construction.
Trusted construction SHALL have tests showing its outputs pass the validator.
Calling the standalone validator SHALL NOT be a required step in trusted
production construction, and routine reconstruction SHALL NOT require runtime
revalidation.

#### Scenario: Verify a package producer

- **WHEN** a package producer constructs contracted data
- **THEN** its conformance tests show that the output passes the contract validator

#### Scenario: Transform contracted data

- **WHEN** a package transformation returns a contract type
- **THEN** it validates its result or constructs a conforming result covered by a validator-based test
- **AND** review identifies that validation or construction as the basis of the guarantee

### Requirement: Consumers trust established contracts

Consumers SHALL use established guarantees without repeating their validation.
Callers SHALL preserve the contract when modifying a value. A contract type
SHALL NOT be taken as evidence that a value has not been modified since its
contract was established; it promises neither immutability nor detection of
invalidation.

#### Scenario: Consume a package producer's output

- **WHEN** a function receives a contract type from a package producer
- **THEN** it does not call that contract's validator or repeat checks of its established guarantees
- **AND** any additional composition or numerical-operation requirements are checked by their owner

### Requirement: Validation checks structure without computing

A contract validator SHALL check only structural guarantees. It SHALL preserve
lazy payloads and shared computation. It MAY inspect already-eager metadata and
indexed coordinates, but SHALL NOT materialize or inspect numerical or
auxiliary-coordinate payloads.

#### Scenario: Validate lazy structure

- **WHEN** a contract validator receives Dask-backed data
- **THEN** it leaves the payload graph unexecuted
- **AND** it returns the identical object or reports a structural violation

### Requirement: Producers establish numerical guarantees

Documented numerical guarantees SHALL be established by the code producing the
contract value, not by the contract validator. A trusted producer SHALL
establish each one either by construction that follows from its own input
preconditions, or by enforcing it with numerical work its algorithm already
requires. A boundary accepting independently supplied data SHALL establish them
with an explicit eager check that owns the materialization, before returning
the contract type. Introducing a contract SHALL NOT add computation repeating
that work or force earlier materialization merely to obtain the annotation.

#### Scenario: Construct a numerical guarantee

- **WHEN** a producer establishes a numerical property by construction
- **THEN** its tests check the property numerically on its outputs
- **AND** review confirms the property follows from the producer's input preconditions rather than from where the data came from

#### Scenario: Reject invalid supplied values

- **WHEN** independently supplied data passes structural validation but violates a documented numerical property
- **THEN** the accepting boundary's explicit eager check fails with the violated scientific requirement
- **AND** the boundary returns no contract type

#### Scenario: Reuse a required numerical decomposition

- **WHEN** preparation already performs a decomposition that can establish a required numerical property
- **THEN** it uses that result to enforce the property's policy and returns its trusted product
- **AND** introducing the contract adds no preliminary or subsequent decomposition solely to check the same property

### Requirement: Extensions preserve shared guarantees

A contract extension SHALL be a subtype of its shared contract and establish
both shared and additional guarantees. The shared contract SHALL permit the
additional data required by its supported extensions.

#### Scenario: Validate an extended contract

- **WHEN** supplied data satisfies the shared contract but violates an extension requirement
- **THEN** the extension validator rejects it
- **AND** successful extension validation produces a value accepted wherever the shared type is expected

### Requirement: Declared contracts have shared conformance coverage

The package SHALL provide one explicit declaration mechanism for xarray
contracts and mechanically discover contract modules for shared conformance
tests, without maintaining a per-module import list.
Declarations SHALL reject types outside the three supported xarray container
families or without their own class-level validator. Records holding contracted
fields SHALL remain ordinary records outside this declaration mechanism.

Each declared contract's tests SHALL supply valid examples, including lazy
payloads, and any required validator arguments for shared identity and
validation coverage.

#### Scenario: Check declaration coverage

- **WHEN** a new contract module is added in the designated location and shared conformance tests run
- **THEN** the module is discovered without adding it to an import list
- **AND** every declared contract is checked for its validator and identity-preserving validation
- **AND** validation leaves lazy example payloads unexecuted
- **AND** a declared contract missing its required examples fails the suite

### Requirement: Raw centered boundary transport has a contract

Prepared or loaded centered boundary transport SHALL be established as a raw
centered transport contract before binding. The contract SHALL require
`H_bc_anomaly` and `H_bc_mass` sharing the spatial `bc_region` dimension and
one raw observation dimension, `time` or `nmeasure`, each with unique indexed
labels and a plain (not period-expanded) state index; a scalar `bc_reference`;
anomaly and reference in the same concentration units and dimensionless mass;
and `boundary_representation='centered-v1'`. Optional additive centering
weights SHALL lie on `(bc_curtain, bc_period)` with unique curtain labels,
interval-start period labels (or the single `whole_window` label without
timestamps) and dimensionless units. Other variables SHALL be permitted.

The contract SHALL document finite anomaly and reference values, finite
non-negative mass that is not all zero, and finite strictly positive weights;
the boundary that first materializes those payloads SHALL establish them.

#### Scenario: Accept supplied centered transport

- **WHEN** independently supplied or loaded centered transport satisfies the raw layout in either operand dimension order
- **THEN** the accepting preparation or loading boundary returns it unchanged with the raw contract type
- **AND** no numerical payload or auxiliary coordinate is computed

#### Scenario: Reject malformed centered transport

- **WHEN** a required variable, attribute, dimension, index, unit or weight layout is missing or inconsistent
- **THEN** the accepting boundary reports the affected element before binding

### Requirement: Bound centered boundary transport has a shared contract

Binding raw centered transport to one fit SHALL produce a bound centered
transport contract consumed by every centered boundary family. It SHALL
contain `H_bc_anomaly` and `H_bc_mass` on `(observation, bc_region)` in that
order, where the observation dimension is the first dimension of
`H_bc_anomaly` and has unique indexed labels in the fit's row order, and
`bc_region` is a unique index with levels `(bc_curtain, bc_period)` covering
every curtain in every boundary period. It SHALL retain the scalar reference
and the raw units. Binding SHALL establish the contract by construction
without computing numerical payloads.

The additive extension SHALL additionally hold dimensionless centering weights
on `bc_region`, aligned exactly to the bound states. Only weighted additive
anomalies SHALL require it; other families SHALL consume the shared contract
without weights.

#### Scenario: Bind raw transport

- **WHEN** a centered family binds raw transport to observations and a boundary calendar
- **THEN** the result passes the bound contract validator
- **AND** payloads that were lazy remain lazy

#### Scenario: Bind weighted additive anomalies

- **WHEN** weighted additive anomalies bind raw transport whose weights label exactly the bound curtains and periods
- **THEN** the result passes the additive extension validator
- **AND** weights prepared with a different calendar are rejected at binding

### Requirement: Absolute multiplicative transport keeps its own layout

Absolute `H_bc` transport, accepted only by the multiplicative family, SHALL
keep its supported state layouts without centered curtain/period or reference
assumptions. It SHALL NOT require a distinct contract in this pilot.

#### Scenario: Bind absolute transport

- **WHEN** a multiplicative choice binds absolute `H_bc`
- **THEN** it binds without centered operands, reference or centering weights
