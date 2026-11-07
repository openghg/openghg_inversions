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

The contract type's docstring SHALL be the canonical description of the
structural and numerical guarantees consumers may assume, including applicable
variables, dimensions, coordinates, units, relationships and numerical tolerance
policies. Producer and input-boundary documentation SHALL explain how their
operations establish those guarantees from their own preconditions. Each
promised guarantee SHALL be established before code relies on it; partial
validation SHALL NOT claim guarantees it has not established.

Consumers SHALL be able to interpret the guarantees from the contract and the
value without access to historical validator arguments. Parameter-dependent
roles SHALL be recoverable through a fixed convention or information already
carried by the value.

#### Scenario: Check structure before numerical values

- **WHEN** supplied data passes structural checks but a promised numerical property remains unestablished
- **THEN** those checks do not return a type promising the numerical property
- **AND** the full contract is claimed only after validation or trusted construction establishes it

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

Within its documented execution preconditions, a contract validator SHALL check
independently supplied data and either return the identical object with the
contract's static annotation or report the
violated requirement, identifying the affected variable, dimension, coordinate
or scientific relationship.

Validation SHALL NOT normalize, mutate, copy or wrap supplied data. Required
normalization SHALL be an explicit preparation operation.

#### Scenario: Validate independently supplied data

- **WHEN** supplied data satisfies the contract and the validator's execution preconditions
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
data without executing it merely to repeat their validation. A validator
requiring concrete payloads SHALL document that precondition and reject lazy
payloads requiring inspection without executing them.

Producers SHALL reuse necessary preparation or algorithmic work to establish
numerical guarantees where possible. Introducing a contract SHALL NOT add a
separate computation repeating that work or force earlier materialization merely
to obtain the annotation. Calling the standalone validator SHALL NOT be a
required step in trusted production construction.

#### Scenario: Validate lazy structure

- **WHEN** structural validation receives Dask-backed data
- **THEN** it leaves the payload graph unexecuted
- **AND** it claims only the guarantees established at that stage

#### Scenario: Reject invalid numerical values

- **WHEN** independently supplied data passes structural checks but violates a required numerical property
- **AND** its owner explicitly materializes it for numerical validation
- **THEN** validation fails at that boundary with the violated scientific requirement
- **AND** no full contract type is returned

#### Scenario: Reuse a required numerical decomposition

- **WHEN** preparation already performs a decomposition that can establish a required numerical property
- **THEN** it uses that result to enforce the property's policy and returns its trusted product
- **AND** introducing the contract adds no preliminary or subsequent decomposition solely to validate the same property

#### Scenario: Reject unsupported lazy numerical validation

- **WHEN** a full validator requires concrete values and receives lazy payloads it must inspect
- **THEN** it reports the execution precondition without running the payload graph
- **AND** it returns no full contract type

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

Each declared contract's tests SHALL supply valid examples and any required
validator arguments for shared identity and validation coverage. Examples SHALL
identify the validation operation and its execution requirements. Structural
validation examples SHALL include lazy payloads; validators requiring concrete
payloads SHALL have eager success examples and lazy rejection examples.

#### Scenario: Check declaration coverage

- **WHEN** a new contract module is added in the designated location and shared conformance tests run
- **THEN** the module is discovered without adding it to an import list
- **AND** every declared contract is checked for its validator and identity-preserving validation
- **AND** structural validation leaves lazy example payloads unexecuted
- **AND** a declared contract missing its required examples fails the suite
