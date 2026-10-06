# Spec Delta

## Purpose

Preserve scientific consistency across equivalent recipe routes with phase-owned
choices, explicit numerical ownership, customization and established check behavior.

Saved preparation and inference follow [prepared-recipe-handoff](../prepared-recipe-handoff/spec.md)
and [sampled-recipe-handoff](../sampled-recipe-handoff/spec.md), respectively.

## ADDED Requirements

### Requirement: Canonical scientific execution across supported routes

Equivalent supported full, prepared-input and staged routes SHALL reuse their
family's scientific preparation, construction, inference and product operations.
Full execution SHALL sequence those operations in memory without mandatory
intermediate persistence. Preparation SHALL consistently derive retained labels
from retained observations and align applicable per-site choices, accepting valid
nonempty subsets without bypassing malformed-input or cache validation.
Direct scientific customization, signatures and numerical returns SHALL retain
their contracts; staged/setup APIs and owned CO2 configuration conventions SHALL
follow the declared reset instead of requiring compatibility aliases.
Families supporting partial workflows SHALL remain usable without implementing
unrelated stages or adopting a common inference algorithm or result format.

#### Scenario: Full execution without intermediate saves
- **WHEN** a full route runs with intermediate saves disabled
- **THEN** it completes through in-memory scientific handoffs and produces
  requested final products without writing or reloading intermediate checkpoints

#### Scenario: Scientific parity and retained subsets
- **WHEN** equivalent routes receive equivalent inputs and choices, including a
  valid retained-site subset after acquisition, cache reload or filtering
- **THEN** their retained arrays, labels, aligned metadata, scientific terms and
  products agree within justified numerical tolerances
- **AND** validation uses real science and an independent calculation or existing
  regression reference rather than requiring identical stochastic trajectories

#### Scenario: Direct customization and partial recipes
- **WHEN** a caller uses supported likelihood or complete-model customization,
  or a family supports only prepared construction and its own inference
- **THEN** its explicit scientific arguments, conflict rules and return contract
  remain available without requiring acquisition, MCMC staging or saved replay

### Requirement: Phase-owned choices and isolated sampler configuration

Preparation, model/inference and output choices SHALL have explicit phase owners.
Resolving or overriding choices SHALL NOT mutate the caller's choices or affect
later invocations, including through nested sampler keyword containers. Sampler
configuration round trips SHALL preserve supported values and container types,
including lists and tuples. Model coordinate and dimension registration SHALL
remain authoritative: inference-data keyword configuration SHALL reject `coords`
and `dims` rather than overriding the registered model. Unsupported or inapplicable
phase choices SHALL fail at their owning boundary before scientific artifact writes.

#### Scenario: Repeated sampling with an override
- **WHEN** one invocation overrides sampler options containing nested lists,
  tuples or mappings and a later invocation uses the original options
- **THEN** the original options and later behavior remain unchanged
- **AND** serializing and restoring supported options preserves their values and
  list-versus-tuple distinctions

#### Scenario: Attempted inference-data coordinate override
- **WHEN** inference-data keyword options contain `coords` or `dims`
- **THEN** option resolution rejects the conflicting keyword before sampling or
  scientific artifact writes, including when its supplied value is empty

### Requirement: Explicit numerical ownership and execution boundaries

Routes SHALL treat caller-supplied numerical inputs as borrowed and potentially
lazy. Choice resolution and access SHALL NOT mutate, copy, compute, persist,
densify or rechunk those inputs implicitly. Related arrays SHALL retain shared
lazy execution until an explicit model-input, serialization or eager-kernel
boundary materializes them together. Built-in construction SHALL own its model
materialization; a supplied complete-model builder SHALL receive potentially lazy
inputs and own the materialization required by its implementation.

#### Scenario: Borrowed lazy preparation and construction
- **WHEN** a route receives borrowed inputs with shared lazy arrays
- **THEN** resolution and preparation preserve input values and lazy execution
  until an applicable explicit materialization boundary
- **AND** built-in construction materializes related model inputs together at its
  declared boundary rather than through incidental configuration access

#### Scenario: Custom complete-model builder
- **WHEN** a caller supplies a supported complete-model builder
- **THEN** the route forwards its resolved scientific arguments and potentially
  lazy inputs without first imposing the built-in construction boundary

### Requirement: One consumed output policy per invocation

Each configuration-driven execution SHALL consume one resolved output policy
through its family's product operations. Derived result metadata SHALL NOT become
an independently resolved output policy. The configured CO2 route SHALL consume
its supported model, sampler and output choices and expose its scientific results
and requested products consistently with that family; raw scientific runners
SHALL retain their separate numerical return contracts. Equivalent live and saved
product routes SHALL retain labelled dimensions, units, chain/draw axes and
family-specific reconstruction meaning without broadening unsupported products.

#### Scenario: Configured CO2 products
- **WHEN** configured CO2 execution requests supported products and writing choices
- **THEN** those choices govern the resulting products and destinations rather
  than being ignored or requiring a separately resolved stage-only output setup

#### Scenario: Live and saved products
- **WHEN** live execution and authenticated replay use the same samples,
  scientific output context and equivalent product policy
- **THEN** their supported scientific products agree, including conditional or
  affine reconstruction where applicable
- **AND** changing reporting or writing choices does not redefine inference

### Requirement: Explicit readiness and independent diagnosis behavior

Readiness and diagnosis SHALL remain independently selectable checks with their
existing schemas, thresholds and pass/fail/unknown meanings. Returned empty or
non-finite predictive evidence SHALL yield a fail check and preserve applicable
predictive/report writes. Standard/multisector SHALL retain their scoped
construction/prediction `KeyError` and `ValueError` catch; CO2 execution errors
SHALL propagate before readiness destination creation. Loading, authentication
outside that catch and serialization errors SHALL propagate as errors. Strict
gates SHALL retain nonzero behavior for fail without newly treating unknown as
fail. Diagnosis SHALL require no original recipe configuration or producing graph
and SHALL declare its historical sample-record support independently of the
scientific-stage compatibility reset.

#### Scenario: Readiness execution and artifact failures
- **WHEN** prediction returns invalid evidence or raises an execution error
- **THEN** the family preserves the stated returned-evidence and exception behavior
- **AND** authentication outside the catch or serialization failure remains an
  error rather than being reported as a completed scientific check

#### Scenario: Independent diagnosis and unavailable metrics
- **WHEN** diagnosis receives a posterior, optional supported sample record and
  convergence choices without a recipe configuration
- **THEN** it assesses convergence without construction or sampling, retaining
  fail when any available metric fails and otherwise unknown when unassessable
- **AND** a historical record is accepted only under diagnosis's explicitly
  supported contract, independent of scientific-stage record support
