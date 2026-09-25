# Spec Delta

> Status: Final. Approved by the specification owner on 2026-09-25.

## Purpose

Defines how retained basis states reconstruct labelled native scaling and signed flux, with explicit source and execution semantics that align with affine reconstruction terminology.

## ADDED Requirements

### Requirement: Directional retained-state application
The retained basis API SHALL expose `state_to_native` for the linear native-scaling action \(x=U_{bucket}\alpha\) and `state_to_flux` for the signed flux action \(f=F x\). The basis operator SHALL own the unweighted native action; the flux-bearing basis value SHALL expose both operations using its retained reference flux. Documentation and docstrings SHALL define the underlying retained-to-native map on first use as a *coarse-to-fine map (prolongation)* and call the requested output operation *reconstruction*. The public API SHALL use one map term rather than retaining a second public adapter name. These operations SHALL NOT change the affine map's \(m+U^*(\alpha-\alpha_{ref})\) equation or infer an affine reference state for the linear basis action.

#### Scenario: Apply a single-source linear basis
- **WHEN** a labelled retained state is applied to a single-source basis and its retained signed flux
- **THEN** the native scaling equals \(U_{bucket}\alpha\), the flux equals \(F U_{bucket}\alpha\), and both retain non-state sample dimensions

#### Scenario: Keep linear and affine meanings distinct
- **WHEN** a caller uses the basis action and an affine reconstruction with non-unit native mean or reference state
- **THEN** each follows its own declared equation while the public operation names identify native scaling and flux consistently

### Requirement: Explicit native source accounting
For a source-specific basis, the directional operations SHALL retain one ordered source axis in the reconstructed result. The source MultiIndex level on the retained `state` dimension SHALL identify its regions without becoming a second independent source axis. Application SHALL avoid an xarray coordinate-name collision if both source roles appear in an intermediate. The operations SHALL support unequal region counts by source. For a shared basis with source-resolved retained flux, `state_to_native` SHALL represent the shared scaling and `state_to_flux` SHALL preserve the flux source axis by broadcasting that scaling. A total-grid product SHALL sum source-resolved flux explicitly at its consuming boundary.

#### Scenario: Reconstruct ragged source-specific states
- **WHEN** sources have unequal numbers of retained regions, nonlexicographic source order, and a gathered MultiIndex state
- **THEN** native scaling and flux preserve each source's labels, order, and values without introducing padded public states or summing sources

#### Scenario: Reconstruct a shared state with source-specific flux
- **WHEN** one retained state basis controls several labelled flux sources
- **THEN** `state_to_native` produces the shared scaling and `state_to_flux` retains separate signed flux values for each source

#### Scenario: Request a total flux grid
- **WHEN** a postprocessing consumer requests a total rather than source-resolved flux
- **THEN** that consumer explicitly sums the native source axis and obtains the same values as the historical weighted reconstruction for equivalent linear inputs

### Requirement: Retained flux is authoritative for normal postprocessing
Normal basis flux reconstruction SHALL use the flux retained with the basis value, including signed and source-specific values and its native time semantics. Output-specific time-axis naming SHALL use the same flux data without introducing an independent routine flux override or prescribing when xarray alignment requires disambiguation. Current output products SHALL retain their values and coordinate labels during this change; replacing the historical `flux_time` label with `time` is tracked separately in #728. When a saved basis is loaded for a new run, the load boundary SHALL bind the validated current-run flux before reconstruction.

#### Scenario: Present a flux time axis
- **WHEN** postprocessing presents a flux time axis alongside observation time data
- **THEN** the output retains the flux values, timestamps, period metadata, source order, and current product labels without unintended xarray alignment

#### Scenario: Reload a basis for current-run flux
- **WHEN** a saved basis geometry is loaded into a run with a validated current flux
- **THEN** reconstruction uses the retained value after that explicit current-run flux binding

### Requirement: Labelled lazy application and safe output conversion
Directional reconstruction SHALL preserve borrowed xarray inputs and Dask laziness unless a verified calculation or output constraint requires a named eager boundary. It SHALL NOT hide compute, persistence, or eager densification of sparse input chunks. It SHALL validate exact ordered state and native labels at their owning boundary, preserve compatible non-state axes, and avoid collisions between native and retained source labels. Dense conversion required by a chosen writer SHALL occur explicitly before serialization, so a sparse result cannot cause the requested output to be lost during writing.

#### Scenario: Apply borrowed lazy inputs
- **WHEN** a Dask-backed or sparse retained basis and flux are applied to chain and draw states
- **THEN** construction of the result does not mutate inputs or compute their payloads except at a named, tested eager boundary, and all sample axes remain labelled

#### Scenario: Reject mismatched labels
- **WHEN** retained state or flux native labels disagree with the authoritative basis labels or order
- **THEN** application fails before positional contraction or silent xarray alignment

#### Scenario: Save a sparse-backed product
- **WHEN** an output writer cannot serialize the reconstructed sparse payload directly
- **THEN** the output boundary converts it to a supported representation before writing and preserves the requested product

### Requirement: Migrate the overlapping interpolation API
In-tree reconstruction consumers SHALL use the directional operations and explicit source summation. During deprecation, `interpolate` SHALL identify its replacement and retain its historical summed and eager multisource behavior. Direct-use inventory SHALL inform removal timing; the deprecated method SHALL NOT become a second permanent reconstruction contract.

#### Scenario: Follow a legacy basis call
- **WHEN** a caller uses the deprecated `interpolate` method during the migration period
- **THEN** it receives a deprecation notice naming the directional replacement and the same source-summed values and conversion behavior as before

#### Scenario: Migrate an output consumer
- **WHEN** an in-tree output consumer needs a total flux field
- **THEN** it applies `state_to_flux`, explicitly sums sources where present, and converts at its established output boundary
