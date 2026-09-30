# Remove the direct HBMCMC implementation

Status: complete in PR
[#714](https://github.com/openghg/openghg_inversions/pull/714), merged on
22 September 2026

The direct `fixedbasisMCMC` / `inferpymc` implementation has been retired.
The compatibility boundary is old configuration syntax, not a second
inversion or sampling implementation.

## Delivered state

Only `openghg_inversions.hbmcmc.run_hbmcmc` remains as a transitional
wrapper. It reads an existing fixedbasis-style INI file, translates supported
names, copies the effective configuration for provenance, and calls
`run_rhime`. It does not expose `--legacy-fixedbasis`, generate new legacy
configurations, or call a separate sampler.

The modern `output_format="legacy"` adapter, historical filename convention,
and currently translated likelihood semantics remain supported. These are
modern RHIME behaviours and must not be treated as remnants of a live HBMCMC
execution path.

PR #714 also removed the legacy executor, sampler, preparation, plotting,
setup, component, and output-helper modules; their private aliases and
adapters; packaged legacy templates; and legacy-only tests and fixtures.
Focused coverage remains for old-INI translation, wrapper-to-RHIME routing,
modern legacy-format output, configuration provenance, and historical wrapper
filenames.

## Remaining compatibility promises

- Existing supported fixedbasis-style INI files continue through
  `run_hbmcmc.py` and execute `run_rhime`.
- `output_format="legacy"` continues to create the HBMCMC-compatible NetCDF
  product from modern `InversionOutput` data.
- Old output filenames and effective-config copies remain available through
  the wrapper.
- Exact direct `fixedbasisMCMC` / `inferpymc` execution, legacy debug return
  dictionaries, `rerun_output`, and the old plotting helpers are not part of
  the current package.

## Consequences for later work

- New runtime, sampling, trace, serialization, and postprocessing changes
  target the RHIME implementation only. They do not need an HBMCMC execution
  adapter or dual-container path.
- `run_hbmcmc.py` should be tested as an input-translation route through
  `run_rhime`, not as a separate implementation surface.
- The modern legacy NetCDF formatter remains in scope for output-regression
  tests, including all-chain and DataTree work.
- Retirement of the wrapper itself is a separate user-migration decision. It
  must not be coupled to removal of already-deleted HBMCMC internals.

## Historical implementation record

The completed removal:

1. simplified `run_hbmcmc.py` to its single RHIME route;
2. deleted the direct executor and its private support modules;
3. removed orphaned fixed-basis aliases and adapters while preserving modern
   boundary sensitivity and `BasisFunctions` APIs;
4. replaced legacy-only coverage with wrapper and modern-output regressions;
5. replaced live HBMCMC API documentation with migration guidance and a
   Towncrier removal fragment.

The change deliberately did not remove the modern legacy NetCDF formatter,
change RHIME likelihood equations or defaults, or remove public RHIME
parameter aliases.
