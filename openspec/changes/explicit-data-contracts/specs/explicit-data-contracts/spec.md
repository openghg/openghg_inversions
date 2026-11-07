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

### Requirement: Contracts own structural and numerical guarantees

A contract SHALL describe the structural and numerical guarantees consumers
may assume, including applicable variables, dimensions, coordinates, units,
relationships and numerical tolerances. Each promised guarantee SHALL be
established by validation or trusted construction before code relies on it.
Partial validation SHALL NOT claim guarantees it has not established.

#### Scenario: Check structure before numerical values

- **WHEN** supplied data passes structural checks but a promised numerical property remains unestablished
- **THEN** those checks do not return a type promising the numerical property
- **AND** the full contract is claimed only after validation or trusted construction establishes it

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

A contract validator SHALL check independently supplied data and either return
the identical object with the contract's static annotation or report the
violated requirement, identifying the affected variable, dimension, coordinate
or scientific relationship.

Validation SHALL NOT normalize, mutate, copy or wrap supplied data. Required
normalization SHALL be an explicit preparation operation.

#### Scenario: Validate independently supplied data

- **WHEN** supplied data satisfies the contract
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
Trusted construction SHALL have tests showing its outputs pass the validator,
with explicit materialization where numerical validation requires it. Routine
reconstruction SHALL NOT require complete runtime revalidation.

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

### Requirement: Validation respects execution boundaries

Structural validation SHALL preserve lazy payloads and shared computation.
It MAY inspect already-eager metadata and indexed coordinates, but SHALL NOT
materialize numerical or auxiliary-coordinate payloads.

Validation requiring concrete values SHALL occur at an explicit execution
boundary, with materialization owned by that boundary rather than hidden in
validation. Trusted construction MAY establish numerical guarantees for lazy
data without executing it merely to repeat their validation.

#### Scenario: Validate lazy structure

- **WHEN** structural validation receives Dask-backed data
- **THEN** it leaves the payload graph unexecuted
- **AND** it claims only the guarantees established at that stage

#### Scenario: Reject invalid numerical values

- **WHEN** independently supplied data passes structural checks but violates a required numerical property
- **AND** its owner explicitly materializes it for numerical validation
- **THEN** validation fails at that boundary with the violated scientific requirement
- **AND** no full contract type is returned

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
contracts and enumerate declared contracts for shared conformance tests.
Declarations SHALL reject types outside the three supported xarray container
families or without their own class-level validator. Records holding contracted
fields SHALL remain ordinary records outside this declaration mechanism.

Each declared contract SHALL supply valid examples for shared identity and
validation coverage. Structural validation examples SHALL include lazy payloads;
full numerical validation examples SHALL respect the explicit execution boundary.

#### Scenario: Check declaration coverage

- **WHEN** the intended contract modules have been loaded for shared conformance tests
- **THEN** every declared contract is checked for its validator and identity-preserving validation
- **AND** structural validation leaves lazy example payloads unexecuted
- **AND** a declared contract missing its required examples fails the suite
