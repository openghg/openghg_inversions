# Configuration ownership amendment

## MODIFIED Requirements

### Requirement: INI decoding is independent of semantic construction

`read_rhime_ini(path)` SHALL return a dictionary of decoded file options. It
SHALL own INI syntax, section flattening and value decoding, and preserve the
existing first-occurrence rule for repeated bare keys. It SHALL NOT require
complete run settings, select a model type, accept overrides, translate legacy
names or resolve defaults/site shorthand. No new file format or section schema
is introduced by this change.

Standard, multisector, nested and custom runners SHALL combine decoded options
with winning overrides, extract their own recipe choices, and construct canonical
configuration once. Shorthand SHALL remain editable until that construction.
Shared semantic/site-alignment helpers SHALL remain independent of file syntax.

Historical fixedbasis translation SHALL be owned by HBMCMC entry points.
Supported RHIME spelling and output-format aliases, including deprecated
`emissions_*` flux selectors, SHALL be translated at runner/frontend boundaries
outside HBMCMC after winning overrides. Existing `run_rhime` behavior SHALL
remain supported on devel and in 0.8, with `DeprecationWarning` for deprecated
options, until their removal in 0.9. Canonical spelling precedence and the
separate flux old/new conflict errors SHALL remain unchanged.
`RhimeConfig.from_params` and canonical normalization SHALL know only canonical
names: they SHALL NOT import alias definitions, translate aliases, issue alias
warnings, or provide alias-specific replacement messages. Unsupported names
SHALL use ordinary unsupported-option validation. Canonical inputs SHALL pass
without alias warnings. Complete and phase-only frontends SHALL share the runner
compatibility owner; phase-only routes SHALL select applicable groups before
value validation/defaults without constructing an unrelated complete request.
HBMCMC-only resource selectors SHALL be translated by the legacy entry point,
or rejected with guidance by modern runner boundaries before construction.
Fixedbasis scientific policy SHALL remain in its separate compatibility adapter.
The `resolve_rhime_config` wrapper SHALL be removed.

During 0.8, HBMCMC's deprecated `params_from_config` SHALL return a dictionary
using the modern decoder and apply overrides before normalization. With
`normalise=False`, it SHALL preserve decoded values and historical names.
With `normalise=True`, it SHALL translate historical names and resource selections
into canonical keys and resolve declared filesystem values to absolute `Path`
objects, preserving file/override origins. This is an explicit return-value
migration from original path strings and historical selector keys; documentation
SHALL describe it. The adapter SHALL NOT be re-exported by RHIME or needed by
modern runners. Its removal belongs to the HBMCMC retirement contract.

#### Scenario: Read recipe-specific or incomplete options

- **WHEN** a caller reads an INI file with incomplete ordinary settings or
  project-specific choices
- **THEN** the reader returns decoded values without trying to construct a run
- **AND** the recipe can remove its own options and apply overrides before
  validating the remaining canonical request

#### Scenario: Override shorthand after reading

- **WHEN** a caller reads site shorthand and changes the requested sites or periods
- **THEN** semantic construction expands only the winning values
- **AND** the reader performs no premature expansion or date-dependent resolution

#### Scenario: Construction translates deprecated spellings once

- **WHEN** a caller supplies a supported deprecated spelling to `run_rhime`
- **THEN** its frontend preserves the canonical value and established precedence,
  emitting a deprecation warning before canonical construction
- **AND** canonical construction receives only canonical names and emits no alias warning

#### Scenario: Direct canonical factory receives an old name

- **WHEN** a caller supplies a deprecated spelling directly to `RhimeConfig.from_params`
- **THEN** it rejects the unsupported name through ordinary option validation
- **AND** it neither translates the name nor emits an alias warning

#### Scenario: Dictionary compatibility adapter

- **WHEN** a 0.8 caller uses HBMCMC's `params_from_config` with supported overrides and normalization controls
- **THEN** the deprecated adapter returns raw overridden values when normalization is disabled,
  or canonical keys and absolute path values when normalization is enabled
- **AND** modern runners can instead read, combine and resolve without using it
