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

## Broader validation pending

The Python 3.12/3.13 tox runs, including the new graph tests and relevant
boundary, biosphere, configuration and handoff coverage, are submitted through
the durable Slurm callback. Their results will be recorded here after completion.
This is implementation parity validation; no new full scientific sampling
experiment or physical recovery claim is made.

The shared recipe uses an explicit frozen Dataset/trace workflow. Generic CO2
prepare/sample/product manifests and linked shared-baseline inference remain
outside that entry point. Existing linked #996/#997 builders retain their
supported routes. Archived experiments and shared production inputs were not
modified. A full Sphinx site build has not been run.
