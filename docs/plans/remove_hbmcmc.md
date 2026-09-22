# Remove the direct HBMCMC implementation

Status: implementation plan

The direct `fixedbasisMCMC` / `inferpymc` implementation can now be retired.
The remaining known active user of `--legacy-fixedbasis` has agreed to move
off that route. The compatibility boundary will be old configuration syntax,
not a second inversion implementation.

## Target state

Keep only `openghg_inversions.hbmcmc.run_hbmcmc` as a transitional wrapper.
It reads an existing fixedbasis-style INI file, translates supported names,
copies the effective configuration for provenance, and calls `run_rhime`.
It does not expose `--legacy-fixedbasis`, generate new legacy configurations,
or call `fixedbasisMCMC` or `inferpymc`.

Keep the modern `output_format="legacy"` adapter, historical filename
convention, and the currently translated likelihood semantics. Removing the
private likelihood-compatibility controls from `run_rhime` would change
scientific behaviour and is a separate decision.

## Implementation

1. Simplify `run_hbmcmc.py` to its single RHIME route. Remove the direct
   legacy flag, callable-signature reflection, template generation, and old
   executor imports. Retain focused old-INI translation and validation.
2. Delete the legacy executor, sampler, preparation, plotting, setup,
   component, and output-helper modules. Fold the small config-provenance copy
   needed by the wrapper into that wrapper.
3. Remove now-orphaned fixed-basis aliases and adapters, including the
   fixed-basis preparation aliases, inferpymc-only likelihood component, and
   legacy basis wrapper. Preserve modern boundary sensitivity and retained
   `BasisFunctions` APIs.
4. Delete legacy-only tests and fixtures. Keep modern RHIME, modern legacy
   output, and wrapper translation coverage. Add a parser assertion that the
   removed flag is rejected and a wrapper-to-RHIME regression covering the
   retained compatibility contract.
5. Replace live HBMCMC API documentation with migration guidance. Document
   old-to-new parameter names, return types, output behaviour, removed APIs,
   and the release boundary. Remove packaged legacy templates and stale
   duplicate instructions. Add a Towncrier removal fragment.

## Compatibility promises

- Existing supported fixedbasis-style INI files continue through
  `run_hbmcmc.py` and execute `run_rhime`.
- `output_format="legacy"` continues to create the HBMCMC-compatible NetCDF
  product from modern `InversionOutput` data.
- Old output filenames and effective-config copies remain available through
  the wrapper.
- Exact direct `fixedbasisMCMC` / `inferpymc` execution, legacy debug return
  dictionaries, `rerun_output`, and the old plotting helpers are removed.

## Validation

- Focused wrapper, parameter, basis, inversion-input, model, and output tests.
- The remaining `rhime_contract` tests, with direct-fixedbasis contracts
  removed from the contract map.
- Ruff on changed Python files and `git diff --check`.
- Documentation build when the configured documentation environment is
  available.
- No slow BP1 inversion is required unless the registered-case planner maps
  the final changed paths to a stable regression case after local checks pass.

## Out of scope

- Removing the modern legacy NetCDF formatter.
- Changing RHIME likelihood equations or defaults.
- Removing deprecated parameter aliases directly from the public RHIME API.
- Adding a new installed compatibility command; the existing module/script is
  sufficient for the transition.
