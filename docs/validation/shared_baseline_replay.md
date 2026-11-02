# Shared atmospheric baseline: replay evidence

This comparison fixes the repaired WUR BASE July 2021 observations, retained
transport, complete affine baseline and aggregation covariance. It changes the
baseline correction prior; it does not regenerate the native coherent prior.
The accepted experiment is `shared-baseline-20261101/results-r3`. Earlier
attempts are excluded. Archived files and shared production inputs are read-only.

There are 5,314 site-hours across 25 sites. Frozen observation weights give each
site equal total weight and divide that weight equally over its retained rows.
For a transported correction D, centering subtracts the column vector of means
a'D. BC periods are gated **before** those means are computed. Predictions on
subsets retain the full-support means. No observation-sized projection matrix
is constructed. The inversion reference is the scalar
409.8755930639068 ppm throughout the comparison.

All arms have one atmospheric delta with Normal(0, 5 ppm) prior and the same
16 eight-day A-pattern coefficients (SD 0.8716963536860672). The observation
likelihood retains the five-hour OU timescale and independent site amplitudes
with HalfNormal(0.75 ppm) priors. Inner priors and the fixed outer ocean are
common to all arms. These widths reproduce a scientific experiment and do not
select production defaults.

| Arm | Sampled outer correction | Selected physical flux states |
| --- | --- | ---: |
| fixed_outer | none | 240 inner |
| additive_net | six net-biosphere and six fossil Gaussian unit-density corrections | 240 inner |
| multiplicative | centered departures of GPP, TER and fossil lognormal states | 240 inner + 18 outer |
| multiplicative_shrunk | same, with residual gross SD multiplied by 0.5 | 240 inner + 18 outer |

The additive arm removes the original sampled outer states. Shared GPP/TER
unit-density support makes separate additive gross corrections equivalent in
covariance to a single net correction with summed variance. It does not identify
a physical GPP/TER allocation. Bio and fossil density widths were matched to
the original multiplicative arm's median centered pointwise prior SD separately:
0.24042540090120165 and 0.012403095591407683 ppm. This matches magnitudes,
not complete covariance matrices.

The accepted fixture verifies exactly zero inner/outer arithmetic
cross-covariance. That fact is fixture-specific. Retaining or dropping states
in other inputs must preserve supported joint covariance or fail explicitly.
The original coherent operator and unresolved covariance are not recomputed.
The shrunk outer residual-gross SD changes from 11496.096988869254 to
5748.048494434627 annualized Tg CO2/year; net SD remains
11629.581348676706 in the same units. Net/protected-state covariances are
preserved. These are retained-state covariance checks, not observational
identification of gross exchange.

## Accepted prototype results

These values were read from the accepted manifests and score summaries. They
are **prototype evidence**, not results of the new production implementation.
Each fit used four chains, 1,000 tuning plus 1,000 retained draws, seed 20260924,
state target acceptance 0.99 and amplitude target acceptance 0.9.

| Arm | Maximum R-hat | Minimum bulk ESS | Baseline RMSE (ppm) | Total RMSE (ppm) | Atmospheric delta mean (ppm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_outer | 1.00805 | 3173 | 1.87006 | 3.23963 | 1.79241 |
| additive_net | 1.00541 | 3082 | 1.63179 | 3.15242 | 2.12548 |
| multiplicative | 1.00692 | 3193 | 1.58801 | 3.15199 | 2.15823 |
| multiplicative_shrunk | 1.00834 | 3418 | 1.58265 | 3.15540 | 2.16576 |

All prototype arms have zero divergences and maximum-tree-depth events for
both sampling steps. Baseline RMSE compares the posterior mean complete
baseline against the archived WUR background; total RMSE compares posterior
mean absolute concentration against observations. Neither is a flux score.
The inner net-biosphere posterior mean errors are respectively +2237.86,
+2989.76, +3024.74 and +3051.02 annualized Tg CO2/year against raw WUR BASE
July inner-domain truth. Smaller baseline concentration residuals therefore do
not imply smaller inner flux errors. These point comparisons do not establish
differences beyond Monte Carlo uncertainty; an implementation replay must
compare chain-based MCSEs before interpreting small differences.

The largest frozen weighted pattern mean is 9.640386904562204e-16 ppm.
Archived first-draw predictions reconstruct within 1.7053025658242404e-13 ppm.
These numerical checks preserve the complete prior baseline once and expose
physical outer response, its removed mean, centered outer response, BC terms
and atmospheric delta separately. Delta's physical source is unidentified.

## Input identities and replay

`examples/shared_baseline/replay.py` consumes an explicit archive path and
verifies every source hash in its preparation receipt before materializing
arrays. It never imports archived Python runners. The relevant frozen inputs
include the repaired prepared product SHA256
`f70a9971887d3f9825224372c2bb9900780041c24bb001f76ec8197827856262`,
the eight-case unit-density fields SHA256
`dd3a330413b3359d17497be2c2248956bccfc4b04ac3df2567557824e3436dea`,
and coherent functional source SHA256
`41bfdd2ddeca137dea6b21dc1d420364b76bb066636058be2e700ca84d38b992`.
The receipt also identifies each site operator, its preparation receipt, the
row table, enhancement product and frozen `designs.npz`. An altered input fails
before model construction. Use a new output directory for each arm.

Bounded graph preflight, from the repository root (configure writable Numba,
Matplotlib and PyTensor caches in the execution environment):

```bash
PYTENSOR_FLAGS=floatX=float64,base_compiledir=/tmp/shared-replay-pytensor \
python -m examples.shared_baseline.replay \
  --archive /path/to/shared-baseline-20261101/results-r3 \
  --output /path/to/new-output/fixed_outer --arm fixed_outer --check-only
```

Repeat for `additive_net`, `multiplicative` and `multiplicative_shrunk`.
Preparation reconstructs the supported BC prior using the frozen scalar
reference and eight-day period choices, verifies the resulting pattern operator
against the accepted design, calls the production shared-baseline primitive,
and checks its serialized/reopened values. It then selects at most two rows
per site and BC period **after** freezing the full-support centering, and checks
ordinary/cached normalized log densities and gradients at nonzero states.
The cached quadratic likelihood itself is checked against the ordinary
observed likelihood and its state gradients. Reopening does not recenter or
recalibrate the frozen preparation.

For the matched sampling replay, omit `--check-only`; defaults are four chains,
1,000 tuning plus 1,000 retained draws and the accepted seed/acceptance targets.
`--chains`, `--draws` and `--tune` allow a bounded implementation comparison;
record any departure from the accepted fit. The adapter writes the frozen
shared preparation into the trace, diagnoses all coefficients and amplitudes,
reopens multiple draws to verify absolute prediction closure and compares inner
physical flux means against accepted chains with MCSEs. Comparison ratios use
the conservative sum of independent run MCSEs, not posterior standard deviations.
Outputs contain the accepted posterior hash and exact implementation revision.

## Production preflight

All four arms passed a bounded production preflight on the 200 retained rows
selected by the procedure above. Full-support preparation and source hash
verification happened first; all four preparations survived save/reopen.
Ordinary and cached graph log densities and full gradients agreed exactly at
the tested nonzero point in this environment. The actual cached quadratic
likelihood was compared separately against the observed likelihood and its
state gradients; maximum absolute gradient differences were:

| Arm | Maximum cached likelihood state-gradient difference |
| --- | ---: |
| fixed_outer | 3.64597e-13 |
| additive_net | 1.33671e-13 |
| multiplicative | 2.58238e-13 |
| multiplicative_shrunk | 1.11022e-13 |

The prepared prototype BC, weights, physical BC and complete prior
concentration were compared with the frozen production preparation. The
shrunk arm used #996 with the original outer physical flux functionals and
multiplier 0.5, then checked its covariance against the accepted frozen result.
Preflight receipts are in
`inversion_outputs/shared-baseline-20261102/preflight/<arm>/report.json`.
They record base revision `fa6d3d29890b22b14d0e2a5be7e7e29d2c09d49e`;
the shared API and adapter were uncommitted worktree changes at preflight time.
These receipts establish bounded graph/serialization checks, not a sampling
result or a check of the committed final revision.

The atmospheric score helper was checked against the accepted fixed-outer
trace independently: baseline bias -1.4971882412544835 ppm and RMSE
1.8700568112489104 ppm reproduce the archived scores within floating-point
roundoff. Their first-order delta-method MCSEs are 0.00256650 and
0.00213288 ppm respectively; total RMSE is 3.239625634623932 ppm with
MCSE 0.00241530 ppm. This checks score accounting and Monte Carlo precision,
not agreement of an unsampled production posterior.

All four accepted traces also passed the complete score-accounting helper,
including physical inner flux functionals, MCSEs and absolute closure of four
reopened sample positions per arm. Those checks adapt prototype variable names
to the production diagnostic names in a new temporary artifact; they do not
resample or alter the archive. Maximum absolute reconstruction differences
were respectively 2.27374e-13, 2.84217e-13, 3.41061e-13 and 2.84217e-13 ppm.
Recomputed atmospheric and inner flux scores reproduce the archived CSVs at
floating-point precision. The helper and per-arm receipts are preserved under
`inversion_outputs/shared-baseline-20261102/accounting/`.
The repeated additive/multiplicative preflight receipts include replay-script
SHA256 `9baf3ce3e728dd29e6be1242085e648d5713b4675f8c755d840f753f90b51823`;
later adapter edits add source hashes, failure receipts and peak memory
accounting without changing model or scoring equations.

Production sampling results remain to be recorded. The prototype took approximately
94–101 minutes per arm with four cores and a reported process peak RSS of
1.7 GiB; this is historical sizing evidence, not a guarantee of resource use.
Run substantial replays through Slurm with the installed callback workflow.
No production sampling pass is claimed by this document yet.
