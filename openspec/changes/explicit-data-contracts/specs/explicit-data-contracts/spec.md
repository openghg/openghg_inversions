# Explicit data contracts

## Purpose

Make the scientific structure expected of labelled data explicit in static
types and validation, while retaining ordinary xarray containers and operations.

## ADDED Requirements

### Requirement: Named contracts describe established guarantees

A data contract SHALL identify the structure and scientific conventions that
consumers may assume, including applicable variables, dimensions, coordinates,
units and relationships between arrays.

The contract SHALL have a named static type and colocated validation.
Its description SHALL distinguish guarantees established at that stage from
numerical conditions checked at a later execution boundary.

#### Scenario: Consume a prepared value

- **WHEN** a function accepts a named data contract
- **THEN** its annotation identifies the guarantees it relies on
- **AND** those guarantees and their validation are discoverable together

### Requirement: Validation preserves the supplied object

A contract validator SHALL check independently supplied data and either return
the same object with the contract's static annotation or report the violated
requirement.

Validation SHALL NOT normalize, mutate, copy or wrap the supplied data.
Required normalization SHALL be an explicit preparation operation.

#### Scenario: Validate independently supplied data

- **WHEN** supplied data satisfies the contract
- **THEN** validation returns the identical object with a narrowed static type
- **AND** its values, coordinates, attributes and array backing are unchanged

#### Scenario: Reject an invalid layout

- **WHEN** supplied data violates a required structural relationship
- **THEN** validation identifies the failed requirement
- **AND** it does not silently repair the data

### Requirement: Trusted producers and transformations own their guarantees

A package function that constructs data satisfying a contract SHALL declare
that contract in its return type. Consumers SHALL trust those established
guarantees without repeating validation.

A transformation returning a contract type SHALL preserve or establish its
guarantees. Routine reconstruction SHALL NOT require complete revalidation.

Callers SHALL preserve the contract when modifying a value. A static annotation
does not promise runtime immutability or automatic detection of invalidation.

#### Scenario: Construct and consume canonical data

- **WHEN** a trusted producer establishes the required layout
- **THEN** it returns the contract type without rerunning structural validation
- **AND** downstream consumers use the established layout directly

#### Scenario: Transform contracted data

- **WHEN** a package transformation returns a contract type
- **THEN** the transformation establishes that its result satisfies that contract
- **AND** an inherited xarray return annotation alone is not sufficient evidence

### Requirement: Validation respects execution boundaries

Structural validation SHALL preserve lazy payloads and shared computation.
It MAY inspect already-eager metadata and indexed coordinates, but SHALL NOT
materialize numerical or auxiliary-coordinate payloads.

Conditions requiring concrete values SHALL be checked at the explicit
execution boundary that owns them. A structural contract SHALL NOT imply
that deferred numerical checks have already passed.

#### Scenario: Validate lazy data before model construction

- **WHEN** structural validation receives Dask-backed data
- **THEN** it leaves the payload graph unexecuted
- **AND** numerical conditions remain assigned to their explicit execution boundary
