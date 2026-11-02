# Linked shared atmospheric baseline: design and acceptance

This follow-up starts from PR #1005 commit
`613567501527607ab80ab97badcc86020ed4af13`, checked against the remote on
2 November 2026. It is developed separately from the ongoing PR #1003 four-arm
CO2 replay. The sections below distinguish the proposed acceptance conditions
from implementation evidence; they are not an OpenSpec specification or a
claim that the full acquisition-to-products route is already delivered.

## Scientific state and response contract

There is one retained flux vector `x`, with one joint arithmetic prior. Both
tracers consume that same draw. The supplied O2 operator already contains the
campaign's signed O2-per-CO2 responses. Those may arise from spatially varying
ratios before convolution; a universal or even per-region scalar ratio must
not be inferred from source names.

For tracer `t`, let `a_t` give equal total weight to each retained site and
equal weight to its retained observation times. Freeze these weights and the
column means on the complete declared support. For any transported correction
`D_t`, form `D_tc = D_t - 1 (a_t.T D_t)`. Gate BC coefficient periods first.
The two tracers may have different site/time support, weights and column means.
Subsets select prepared rows without recomputing them.

For multiplicative outer states, the mean is

```text
mu_t = H_inner,t x_inner
     + f_t + prescribed_BC_t + H_outer,t m_outer
     + 1 delta_t + D_BC,tc v_t
     + H_outer,tc (x_outer - m_outer).
```

The fixed affine contribution `f_t` and all fixed outer prior contributions
are included once. `delta_co2` and `delta_o2` are independent atmospheric
level corrections in physical concentration units. They are shared between
BC and outer baseline within a tracer; they are not one common numerical
coefficient across tracers. Their physical source remains unidentified.
Per-tracer BC correction priors remain independent unless a separately
supported explicit covariance is supplied; none is inferred here.

The joint flux covariance induces cross-tracer concentration covariance.
Prepared joint aggregation-error covariance, including its cross-tracer
blocks, passes through unchanged. The existing fixed-OU channel/site blocks
remain as implemented. Calibration changes retained-state moments against
the saved prior; it does not regenerate native covariance, coherent transport,
the fixed affine term or unresolved aggregation error.

Initial modes are fixed outer, multiplicative outer, and multiplicative outer
with a resolved positive #996 gross-biosphere calibration. Inner ocean states
retain their existing treatment; outer ocean remains fixed. Removing states
with covariance against sampled states must fail explicitly unless the
operation is implemented coherently. Nonzero covariance among retained inner
and adjustable outer states is preserved.

## Why collapsed additive net biosphere is initially unsupported

For one common transport support, write signed CO2 gross corrections as
`g` and `r`. CO2 sees `u (g + r)`. With signed O2-per-CO2 ratios `q_g` and
`q_r`, O2 sees `u (q_g g + q_r r)`. Changing gross allocation by `(k, -k)`
keeps the CO2 net response fixed and changes O2 by `u (q_g - q_r) k`.
The net-only collapse therefore loses a linked direction whenever the paired
gross responses differ. Spatially varying pre-convolution responses make a
single net ratio still less justified.

A future additive route needs either the necessary gross degree of freedom
with a declared joint prior, or an explicit fixed gross-allocation convention
and its resulting transported response. The existing CO2-only net correction
does not provide that convention. This follow-up rejects that unsupported
collapse rather than selecting an oxidative ratio or uncertainty from WUR fit.
The [input evidence](linked_shared_baseline_inputs.md) records actual campaign
conventions and a small linear oracle.

## Native centred arithmetic and physical units

Each tracer has one scalar concentration reference `c_t` across the inversion.
Carry native float64 anomaly `A_t`, dimensionless mass response `M_t`, and
that reference. For `g_t = M_t 1`, the likelihood-centred prescribed boundary
is `A_t 1 + c_t (g_t - 1)`. Compute this expression directly. Constructing
`A_t 1 + c_t g_t` and subtracting a roughly 200,000 ppm O2 reference loses
small corrections unnecessarily. Correction widths are never multiplied by
the reference.

Physical outputs restore the scalar once. BC raw correction, its removed
mean and centred atmospheric correction remain separate, as do physical outer
response, removed mean, centred outer correction and delta. First support
requires common concentration units for the existing linked builder and noise
interface. Acquisition/preparation owns any compatible Pint conversion of
values and covariance axes before this boundary. A unit label alone is not a
conversion. Species-specific widths and references remain explicit.

New prepared shared artifacts retain native centred operands. Legacy CO2-only
artifacts remain supported by their existing route. A linked builder must
reject an artifact without the native operands needed for O2 precision and
request rebuilding from the original BC correction; it must not silently
reconstruct a supposedly precise anomaly from absolute concentrations.

## Acquisition-to-products acceptance plan

The workflow owner can audit inputs and build fixtures before model readiness.
Model signatures and state meaning must be independently reviewed before
dependent configuration/manifests integration is released. Use the maintained
`Co2O2PreparedInputs` and linked codecs/runners as owners; no new model compiler,
acquisition registry or generic lifecycle is required.

| Stage | Required handoff and acceptance evidence |
| --- | --- |
| Acquisition | Select species-specific observations and paired flux products with explicit identities. Keep synthetic UOB truth/observations distinct from WUR. Declare either a pinned delivered BC baseline or a fresh repaired-endpoint convolution. |
| Corrected BC ingress | Verify delivery manifest and catalog-registration hashes. Preserve float64 native A/M and require explicit scalar references. Do not assume an old object-store retrieval applies the endpoint repair. |
| Timing | Pinned deliveries retain their archived midpoint-label forward-fill/max-six-hour policy. Fresh native-field convolution using the maintained timing interface is a different numerical preparation and needs its own identity and comparison. Do not claim delivery parity without a matched check. |
| Scientific preparation | Align labelled native channel rows and joint retained states. Preserve paired O2 transport, signed-ratio provenance or its explicit unavailability reason, separate ocean roles, independent errors and the full joint aggregation covariance. Never substitute an overlay hash or catalog ID for a provider UUID. |
| Shared preparation | Resolve each BC family/period/reference and physical width once; resolve one calibration policy against the saved prior. Freeze per-tracer full-support means and weights; store original/selected prior identity, selectors, calibration and native centred operands. Unsupported additive allocation fails before sampling. |
| Prepared save/reopen | Existing DataTree/NetCDF or Zarr codecs preserve both native supports and all state/covariance identities. Reopen exact content, compare means/operators and build without re-centring or current-context calibration. A subset reuses whole-support values. |
| Model build | A nonzero analytic point confirms one flux latent affects both tracers with the supplied signs/responses, separate deltas have their declared widths, and inner/outer/affine terms close absolutely once. Ordinary and refreshed cached normalized densities and gradients agree, including joint error covariance. |
| Sample checkpoint | Save the complete frozen per-channel preparations beside a restored posterior; retain exact prepared content identity, prior identity and run policy. Reuse the maintained compact linked recording path from PR #992 when integrated, rather than inventing a posterior variable filter. Observed/predictive absolute restoration must occur exactly once. |
| Product replay | Reconstruct total concentration and each raw/removed/centred contribution from saved draws and saved operators, without a live model or data retrieval. Preserve labelled physical inner/outer flux functionals and units. Unsupported generic products fail explicitly. |
| Small end-to-end control | Synthetic UOB zero-baseline and declared injected-baseline controls exercise acquire/prepare/save/build/sample/reopen/products through public interfaces. Initially use bounded draws for implementation checks; record sampling uncertainty rather than asserting recovery from a short chain. No WUR width tuning. |

Before any substantial campaign fits, require small analytic and linear oracles,
nonzero ordinary/cached cache checks, mismatched-unit failures, near-unity mass
precision tests, frozen unequal-support subsets, covariance/label guards,
CO2-only restriction parity, and saved posterior reconstruction. Scientific
widths used by these controls are experimental inputs, not production defaults.

## Implementation and validation record

The model implementation is committed as
`3e065f31c8e732c2f553827c7f3cb794f6762300`. It adds native-channel frozen
preparation, ordinary/cached linked builders, full trace annotation and pure
saved-draw reconstruction. Its eight focused linked cases and eleven existing
CO2 shared-baseline cases pass. Existing CO2 v1 artifacts retain their previous
fallback; linked artifacts explicitly require the native centred operands.

The [independent review](linked_shared_baseline_review.md), committed as
`f26a684e`, clears that scientific API for dependent integration. Twelve
independent cases compare raw-input analytic means, dense normalized Gaussian
densities and gradients with both ordinary and refreshed cached targets. They
also check near-unity mass at O2 reference 209300 ppm, separate deltas, exact
frozen subset operators, one shared Dask execution, inner-ocean preservation,
actual trace storage/reconstruction, pattern-label rejection and the CO2-only
marginal restriction. Focused Ruff and whitespace checks pass.

The [bounded saved-input check](linked_shared_baseline_inputs.md), committed
as `0e3994d5`, uses 28 original UOB rows (16 CO2 and 12 O2) and the unchanged
282 active-state marginal. Current prior validation accepts those moments
without projection or jitter. The maximum selected cross-tracer unresolved
covariance is 45.33039137343721 ppm squared. The coherent affine remainder
reaches 8.353646793292402 ppm and is retained, rather than assumed zero.
Native corrected-endpoint closure differs by at most 5.69e-14 ppm for CO2
and zero for O2. Prepared MultiIndex save/reopen, joint covariance labels and
frozen subset means pass.

The historical UOB observations are inner-only enhancements, while the repaired
BC delivery contains absolute baselines. No likelihood combines those unmatched
quantities, no observation is altered, and no scientific sampling is claimed by
that check. A coherent absolute UOB control is an acquisition/preparation task.

The original #1003 replay remains separate. All four matched CO2 arms
completed at scientific revision `6c61837f`; its report and independent
label-aligned target audit are recorded in documentation commit `bef3c486`.
All convergence gates passed, with one multiplicative TER posterior-mean
comparison explicitly flagged at 2.103 summed Monte Carlo standard errors.
Its original Python 3.12/3.13 validation passed 495 cases per interpreter.
These results do not validate this follow-up's new source. Its separate
compatibility and documentation checks are recorded below.


### Repository lint and full documentation build

Slurm job `19885984` completed with exit code zero on 2026-11-02 against
`3e8dac3da6fd01c6a3d8d0954b0dbe346648c5ad`. Both tox environments,
`lint` and `docs-full`, passed. The full Sphinx build includes the linked
usage guide and generated API pages. Its ten existing warnings comprise seven
duplicate `SiteOptions` descriptions and three ambiguous `PreparedInputs` /
`DataConfig` cross-references; none originates in the new linked guide.
The log is `inversion_outputs/linked-validation/tox-docs-19885984.out`.
The supported-runtime scientific result is recorded separately below; the
documentation build does not establish sampling or campaign acceptance.


### Supported-runtime scientific validation

Slurm job `19885982` completed with exit code zero on 2026-11-02 in 37 minutes
40 seconds against `3e8dac3da6fd01c6a3d8d0954b0dbe346648c5ad`. All **262 selected
cases passed on Python 3.12.11 and all 262 passed on Python 3.13.7**. This is
the relevant scientific compatibility matrix, not a full-repository test run.
It covers linked and CO2-only shared preparation/review, BC and biosphere
priors, existing linked baselines, ordinary/cached fixed-OU targets, additive
boundary options and CO2/linked configuration. Compiler/header preflights
passed on both interpreters. The 374 warnings per environment concern existing
Numba dispatch/performance and future PyTensor, xarray and PyMC interfaces;
none is a test failure. The log is
`inversion_outputs/linked-validation/tox-scientific-19885982.out`.
Later commits through this result only add documentation/API pages; the tested
scientific owners remain unchanged. No additional fit or production width was
introduced by this validation.

### Separate workflow and coherent acceptance handoff

Draft PR #1006 at `ac387295c208ea80daf81d221f9f646fe1be712f` integrates the
reviewed scientific dependency `fe068ad1` with configured preparation, frozen
v4 numerical save/reopen, exact manifests, maintained compact recording and
graph-free basic physical/atmospheric products. Its independent owner reports
430 cases passing on each supported Python version and ten completed
zero/injected synthetic fits. The one-cell, one-site toy deliberately has zero
aggregation covariance; it does not validate native coherent campaign inputs.
Maximum atmospheric accounting error is 1.14e-13 ppm, with no divergences;
two fits retain R-hat concerns (1.01915 and 1.01124), so a blanket convergence
or coverage claim is inappropriate. Shared-baseline PARIS export remains
explicitly unsupported pending its reviewed removed-mean mapping.

The separately authorized single Verification Games task was submitted to an
isolated BP1 worktree after the reviewed integration gate. Its remit is bounded
coherent real-input acceptance, preserving full effective transport, affine
terms and joint unresolved covariance with verified absolute UOB controls.
Software target checks do not substitute for that acceptance. Additive
net/gross versus positive gross-shrunk lognormal is a separate bounded research
prototype within that task. One physical correction state can drive distinct
tracer response operators; a collapsed CO2 net-only pattern is not the required
representation for a linked additive model. No additive production family,
guessed oxidative ratio, production calibration or publication is authorized
by the comparison. Final coherent acceptance and the retain/defer research
recommendation remain outstanding.
