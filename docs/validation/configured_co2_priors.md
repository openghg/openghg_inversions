# Configured CO2 priors: validation record

The implementation is based on reviewed shared-API commit
`02b0313fb28636dcee1ee53c7727d5088215a68c` (PR #1003). Both the remote branch
and PR head were verified before creating the isolated implementation branch.
The inherited #996 and #997 equations and existing model defaults are retained.
The parent's subsequent `6c61837f` lint/documentation-only commit was inspected
and merged before submitting broader validation; it changes no scientific target.
Python package metadata remains unchanged: supported runtimes are 3.12 and 3.13,
not the inherited request's Python 3.10.

## Scope

Explicit preparation owners parse boundary family/reference/period/physical
width choices, one biosphere calibration policy including marginal gross SD,
and the CO2-only shared-baseline mode/level width. They accept labelled
scientific inputs separately. Existing CO2/linked configuration owners select
frozen prior artifacts, and checkpoint target checks protect those inputs.
Reload checks calibration integrity using the saved baseline and retains the
saved covariance. It never calibrates against current inputs or row subsets.

Two bounded changes to scientific owners support this workflow: a level-only
period preparation avoids reshaping a zero-length Dask pattern axis; the
biosphere loader returns saved moments after its existing integrity check.
No model mathematics or production widths were added.

## Focused evidence

On the worktree's lockfile environment (Python 3.13.7), 56 focused tests passed
across `test_co2_prior_configuration.py`, `test_co2_config_owners.py` and
`test_biosphere_prior.py`, excluding the new graph comparisons. Four additional
sample/prior-predictive output-alias checks passed. Ruff and `git diff --check`
passed. The initial saved-checkout environment could not collect tests because
it contained an older incompatible OpenGHG; it was not changed. A worktree-local
environment was created from the existing lockfile.

The new tests compare configured preparation with explicit Python preparation,
all boundary families with/without periods, compatible unit conversion, every
biosphere policy in both families, all four shared outer modes, exact artifact
reopening, unchanged whole-support means on subsets, and rejection of unknown,
conflicting or inapplicable options. The graph tests check parameter identities,
absolute predictions, normalized densities and gradients at nonzero states for
ordinary/cached CO2, linked ordinary/cached builders, and shared CO2. They also
save/reopen annotated absolute predictive traces and run the documented example.

## Broader validation

Slurm job `19885280` ran 299 cases per environment through the repository tox
runner on Python 3.12 and 3.13. Each environment passed 294 and failed the same
five new tests. The failures identified an incorrect `SigmaAlignment` import in
the new shared configuration owner, an incomplete numerical preparation fixture
and an unnamed fixed contribution in the documented example. These were fixed;
the CO2 inference fixture now uses a real prepared-input object with explicitly
valid retained lognormal moments. The shared additive-sigma and example checks
pass after correction. Heading lengths were also checked in the new RST pages.

Slurm job `19885522` completed successfully at scientific implementation commit
`fa7386b77882db2a790a11dd39958f1855b0f050`. All 45 new configuration cases passed
on both Python 3.12.11 and Python 3.13.7 through `py312-openghgCur` and
`py313-openghgCur`, including the five corrected cases. Compiler/header
preflights passed. No required targeted test remains failing: the first run
covered all 254 existing regressions per interpreter, and the successful rerun
covered the complete new module. The broad suite was not repeated after changes
limited to the new configuration owner, fixture and example.

Both logs remain under `inversion_outputs/config-priors-validation/` in the
implementation worktree on BP1: `slurm-19885280.out` records the broader run,
and `slurm-19885522.out` records the successful rerun. The terminal callback
ticket is `sw-20261102T170920Z-45184f7cc32c`. Warnings concern the existing
Numba object-mode/performance and PyTensor gradient APIs; no new test failed.
Final focused Ruff, formatting and whitespace checks pass.
This is implementation parity validation; no new full scientific sampling
experiment or physical recovery claim is made.

The shared recipe uses an explicit frozen Dataset/trace workflow. Generic CO2
prepare/sample/product manifests and linked shared-baseline inference remain
outside that entry point. Existing linked #996/#997 builders retain their
supported routes. Archived experiments and shared production inputs were not
modified. A full Sphinx site build has not been run.


## PR review follow-through

The full configured `run_co2` route now resolves the writer's concrete product
and trace paths before sampling and checks them against selected boundary and
biosphere artifacts. The shared path checker rejects resolved aliases, hardlinks
and ancestor/descendant directory overlaps. The containment rule is the same
one already used by the child linked-workflow branch; this parent fix makes it
available consistently without changing scientific defaults or naming.

Fourteen new full-run regressions use real matching saved priors and require
sampling to remain uncalled and input bytes unchanged. They cover direct,
symlink and hardlink aliases for trace/basic outputs and both enclosing-Zarr
trace writes and writes inside a selected prior store. All fourteen passed.
Forty-five existing full/saved execution, output, CLI and stage-contract tests
also passed. Focused Ruff and whitespace checks pass. The curated API landing
now includes the seven public prior-configuration exports and the shared
wrapper links directly to its labelled scientific input contract.

Dual-Python CI-equivalent validation and a documentation build are submitted
from the exact fix source, with completion recorded separately when available.
