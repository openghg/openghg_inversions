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
Supported modern RHIME aliases SHALL have owners outside HBMCMC;
`RhimeConfig.from_params` SHALL resolve those aliases before modern coercion,
validation and construction without importing or calling HBMCMC. Runners SHALL
NOT repeat that translation. HBMCMC-only selectors SHALL be translated by the
legacy entry point before construction, or rejected by modern entry points.
The translator SHALL emit `DeprecationWarning` when it replaces or removes a
deprecated option name or output-format value; canonical options SHALL pass
without a deprecation warning. For the relocated RHIME name/output aliases,
canonical spellings SHALL retain their existing precedence when both spellings
are supplied. Separately specified conflict rules, including the error for both
modern flux-selector spellings, SHALL remain unchanged. Fixedbasis-specific
scientific policy SHALL remain in its separate
compatibility adapter. The `resolve_rhime_config` wrapper SHALL be removed.
During 0.8, the deprecated `params_from_config` adapter inside HBMCMC SHALL
preserve its dictionary, normalization and override behavior using the modern
decoder; its normalized route SHALL translate applicable historical selections
after overrides. It SHALL NOT be re-exported by RHIME or needed by modern runners.
Its removal with the entire legacy subpackage is governed by the separate
HBMCMC isolation/retirement capability. The namespace restriction and
legacy-only routing are explicit exceptions to general adapter compatibility.

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

- **WHEN** a runner or direct factory caller supplies a supported deprecated spelling
- **THEN** `RhimeConfig.from_params` resolves the canonical value through its
  modern alias owner, emitting a deprecation warning for the changed option
- **AND** canonical inputs do not emit deprecation warnings
- **AND** HBMCMC-specific historical behavior remains in the HBMCMC translator

#### Scenario: Dictionary compatibility adapter

- **WHEN** a 0.8 caller uses HBMCMC's `params_from_config` with supported overrides and normalization controls
- **THEN** the deprecated adapter retains its established dictionary contract
- **AND** modern runners can instead read, combine and resolve without using it
