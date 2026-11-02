# Linked shared workflow: completed synthetic controls

The saved workflow and both supported Python environments completed successfully.
These are small synthetic development controls. Real-data campaign acceptance
and a blanket convergence pass remain outside this result.

## Experiment and truth

The source is `examples/shared_baseline/linked_workflow.py` at
`5cda695fa87ecc220c2d317ae95bd992967ee112`. The synthetic domain has one footprint
cell and one site, SYNTH, with six hourly CO2 rows and four O2 rows on
1 January 2020. The declared acquisition window is 1–2 January. Both channels
pass through maintained surface preparation and the explicit direct-template
adapter. The state coefficients scale the complete declared source templates;
there is no unresolved process in this synthetic template model, so its declared
aggregation covariance is zero.

State-space truth is the positive coefficient vector GPP=1.15, TER=0.85, FF=1.2,
CO2 ocean=1 and O2 ocean=1. Coefficients are dimensionless. GPP/TER and ocean
states are selected as outer; FF is inner. In fixed mode, the outer prior means
equal their known fixed truth. In adjustable modes, the prior means are one and
the original arithmetic covariance is diagonal with variance 0.04. The shrunk
mode applies the example's explicitly resolved GPP/TER calibration with residual
standard-deviation multiplier 0.5. Outer ocean states stay fixed at one.

Observation-space truth is the reviewed frozen shared-baseline reconstruction
at those state coefficients, with independent physical CO2/O2 atmospheric
deltas: each is a scalar correction to its frozen physical prior baseline.
The zero control uses (0, 0) ppm and the injected control uses
(+0.35, -0.45) ppm. No random observation noise is added. Native boundary
references are 400 ppm CO2 and 209000 ppm O2. The separate atmospheric delta
priors have standard deviation 1 ppm. The boundary family is level-only.

The fixed-independent likelihood uses R = (0.05 ppm)^2 I, with fixed measurement
standard deviation 0.05 ppm. Fixed OU adds species/site covariance with a 2-hour
timescale and fixed amplitude 0.1 ppm. Inferred OU keeps that timescale and uses
HalfNormal amplitude priors with scale 0.1 ppm. Each control uses two PyMC chains,
1000 tuning and 1000 retained draws per chain, with seed 246. The inferred OU
route uses the maintained matched cache and compact deterministic reconstruction.

## Results

Slurm array 19886194 completed all five tasks with exit code 0. Each task ran
both controls, saved preparation and sample checkpoints, reopened the samples,
and exported native concentration and physical-state products. All ten fits
reported zero divergences. Batch peak memory ranged from 632752 to 1121472 KiB.

The table reports posterior mean atmospheric deltas in ppm, in CO2/O2 order.
Each row combines its zero/injected pair; R-hat is the maximum and bulk effective
sample size (ESS) the minimum over both fits' sampled physical state, baseline
and applicable OU-amplitude quantities. True zero/injected deltas are (0, 0)
and (+0.35, -0.45) ppm respectively.

| Outer mode | Likelihood | Zero means CO2, O2 (ppm) | Injected means CO2, O2 (ppm) | Maximum R-hat | Minimum bulk ESS |
|---|---|---:|---:|---:|---:|
| fixed | fixed_independent | 0.010, -0.014 | 0.363, -0.468 | 1.0191 | 471 |
| multiplicative | fixed_independent | 0.014, -0.020 | 0.364, -0.468 | 1.0100 | 635 |
| multiplicative | fixed_ou | 0.026, -0.041 | 0.372, -0.486 | 1.0064 | 867 |
| multiplicative | inferred_ou | 0.019, -0.029 | 0.368, -0.478 | 1.0112 | 474 |
| multiplicative_shrunk | fixed_independent | 0.013, -0.019 | 0.364, -0.469 | 1.0037 | 612 |

Every baseline truth lies inside its reported marginal 95% interval. All stated
state truths also lie inside their marginal 95% coefficient intervals, including
the point intervals for fixed outer states. Every saved physical coefficient
draw is positive and finite. These interval checks describe this constructed
experiment; they do not establish repeated-simulation coverage or SBC.

Saved concentration products satisfy the explicit accounting identity
`modelled_concentration = physical_atmospheric_total - removed_mean_correction`
on every retained row and draw, with maximum absolute difference
1.14e-13 ppm across both tracers and all ten fits. Physical source contributions
use original sensitivities and full physical coefficients; centered atmospheric
adjustments and their removed means remain separately available.

Two fits retain a diagnostic concern: fixed-independent zero has maximum R-hat
1.01915, and inferred-OU zero has maximum R-hat 1.01124. The minimum bulk ESS
across all ten fits is 471.2. The recorded outcome supports execution, target
consistency and the reported marginal recovery intervals. Further inference
acceptance should assess the actual campaign model and its diagnostics.

## Software validation and evidence

Immutable core source `4df5eede2b82e760aaf6223272c387891fc7aa4e` passed 430 tests
on Python 3.12 and 430 on Python 3.13 in Slurm job 19886130, exit code 0,
16 minutes 24 seconds, batch MaxRSS 5167244 KiB. The later example-only target
correction passed 54 focused preparation/template tests and independent review.
The previous prerequisite run passed 390 tests on each Python version.

[Machine-readable recovery results](linked_shared_workflow_results.json) retain
full intervals, positivity, accounting differences and diagnostic values.
[Validation history](linked_shared_workflow_preflight.md) records the first
array's target-definition error and its correction; those earlier estimates
are superseded by the controls reported here. [Usage](../usage/linked_shared_workflow.rst)
explains preparation and saved-stage interfaces.

Raw artifacts remain under the worktree's
`inversion_outputs/linked-workflow-validation/source-5cda695fa87e/inversion_outputs/linked-controls/19886194/`.
CI logs are `inversion_outputs/linked-workflow-validation/slurm-19886130.out`;
control logs are `inversion_outputs/linked-controls/slurm-19886194_<index>.out`.

The implementation is ready for review and a separately scoped coherent
real-input acceptance run. Shared-baseline PARIS export remains explicitly
unsupported pending its scientific output mapping. No protected validation
inputs or production campaign outputs were changed.

## Gathered-state integration follow-up

PR #1006 review identified a post-sampling label failure for gathered physical
states. Full-state restoration now right-aligns the sampled array with the
complete prior, retaining the gathered MultiIndex. Product reconstruction
reverses every prefixed physical coordinate name, including arbitrary index
levels such as `region_in_source`.

The regression exercises real configured model construction and two prior
predictive draws. A copy of those draws supplies a synthetic posterior solely
for output validation; it is not a posterior inference result. The four cases
combine removed/all-retained outer states with NetCDF/Zarr sample checkpoints.
They check unchanged selected values, fixed outer arithmetic means, exact
physical labels, graph-free replay and native-channel accounting. Supported
Python validation is registered separately; these software checks do not extend
the scientific campaign claims above.
