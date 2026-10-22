# OPE-152 amplitude evaluator: remaining research ideas

Research record, 22 October 2026. These are proposed experiments, not supported
sampler options or implementation requirements. OPE-152 owns follow-up work.
The immediate priority is the likelihood/gradient stability defect identified
during review of [PR #928](https://github.com/openghg/openghg_inversions/pull/928#issuecomment-6339071109).
Any subsequent optimization must preserve the stable likelihood and both
gradients, including cases where a finite Cholesky factor hides cancellation.

## Target and measured starting point

Keep the existing scientific target: `aggregation_error_rank=256` with its
diagonal tail, fixed five-hour OU timescale, unchanged observations and priors,
and the 12 site amplitudes conditional on the current flux/boundary state.
Retain the HalfNormal amplitude prior with scale 0.75 ppm. The numerical
comparison uses float64 and one numerical thread per process. Sampling
comparisons retain amplitude/state target acceptance 0.9/0.95 and the existing
tree-depth limits; reducing those limits is not an efficiency result.

The [native prototype report at revision
`5ed1ff90`](https://github.com/openghg/openghg_inversions/blob/5ed1ff90d47cca4ea58ac5c21a262331b669c1b2/docs/plans/ope152_native_amplitude_performance.md)
records a 7.24% reduction in compiled conditional-density time across six fixed
posterior anchors. In the short paired sampling comparison, mean retained
density-evaluation time fell 6.99% and retained transition time fell 4.74%.
Compilation startup increased, and amplitude bulk effective samples per
retained sampling second decreased in 11 of 12 coordinates. These short runs
do not establish a posterior-efficiency gain.
[PR #939](https://github.com/openghg/openghg_inversions/pull/939) was closed
without merging on 22 October. Its branch and benchmark evidence remain
available as a reference. It is not a maintained native backend.

The remaining cost is chiefly numerical amplitude evaluation. The prototype
already calls a BLAS triangular solve; merely requesting TRSM is not a new
optimization. The proposals below target the broad triangular solve and the
work needed for the amplitude-gradient trace. Their savings are unmeasured.

## Exact numerical candidates

In the prepared OU mode basis, write the covariance as
`C = D(sigma) + F F.T`, with diagonal weights `w_i = 1 / D_ii`, rank 256 and
`K = I + F.T diag(w) F = L L.T`. The amplitude gradient requires the diagonal
of the inverse covariance, in addition to the residual solve. The current
fast route obtains the necessary factor leverage through a triangular solve
with one right-hand side per observation.

1. **Compute the small inverse triangular factor, then multiply.** Compare
   `T = inverse(L)` followed by a matrix product with the weighted factor
   against the existing broad triangular solve. Preserve the resulting
   squared column norms and the guarded fallback. A small triangular inverse
   plus matrix multiplication may use the installed BLAS more efficiently,
   but adds work and rounding error. It is not an established improvement;
   assess both accuracy and the entire density/gradient evaluation.
2. **Compute the 12 required site traces directly.** Let `f_i` be row `i` of
   `F`, and define `G_g = sum_{i in site g} w_i**2 f_i f_i.T`. The gradient's
   leverage contribution is exactly `trace(K^-1 G_g)`. Test whether these
   small contractions can replace the observation-wide solve without forming
   every individual leverage. The weights change with the amplitudes, so
   these 12 matrices cannot simply be cached unchanged. Their construction,
   inverse/solve cost and numerical errors must all be included.
3. **Use layout and blocking as controls.** Compare a small set of explicit
   factor orientations and blocked solve/norm reductions. Count copies and
   temporary memory. A faster isolated BLAS call does not establish a faster
   conditional evaluator.

Start with the first two candidates only after the stability fix is validated.
Use the same six residual/amplitude anchors and interleaved timing order,
including the ordinary guarded route and independent adversarial numerical
oracles. Check likelihood, amplitude-gradient and general residual-gradient
parity, input ownership and rejection behavior. A candidate must preserve the
joint stable fallback; a condition estimate alone is not an accuracy proof.
Only a repeatable whole-density improvement warrants another sampling
comparison.

## A distinct auxiliary-variable experiment

An alternative uses the exact representation in the same prepared mode basis:

```text
r = F z + epsilon,  z ~ N(0, I),  epsilon ~ N(0, D(sigma))
K = I + F.T D^-1 F
m = K^-1 F.T D^-1 r
z = m + L^-T eta,  L L.T = K,  eta ~ N(0, I)
```

For a fixed flux/boundary state, draw `z` once, then update the amplitudes
conditional on that `z`. Computing `r - F z` once gives a diagonal likelihood
whose amplitude value and gradient cost O(number of observations) per
evaluation. The factorization and exact Gaussian draw still cost once per
outer sweep. An invariant amplitude kernel, including its priors and transform
Jacobian, preserves the augmented conditional target; this is an exact
representation of the chosen rank-256 covariance, not a change to its tail.

The proposed order is:

1. Draw a fresh `z | flux, boundary, sigma, observations`.
2. Update `sigma | flux, boundary, z, observations`, keeping `z` fixed through
   the amplitude trajectory.
3. Discard `z`, refresh the accepted-sigma cache, and run the existing
   **marginalized** flux/boundary update.

Do not reuse `z` after that marginalized state update. The next sweep draws it
again using the new conditioning state. The ordering and removal of `z` are
part of the proposal; mixing conditional and marginalized updates casually
would not justify the target-preservation argument.

Historical work already tested an exact Gaussian `z` update. The
[2 August verification-games worklog](https://github.com/openghg/verification-games/blob/main/notebooks/worklog.d/2026-08-02-105459-collapsed-gaussian-site-sigma-calibration.md#L64)
records the order site-amplitude Slice, exact Gaussian factor draw, then state
NUTS **conditional on `z`**. Its real-data smoke runs were faster per draw but
failed the state-mixing checks, including a rank-256 site-block variant.
Those runs used different covariance representations and precision choices;
they are historical warnings, not a matched benchmark for the current target.
The source was cross-checked against local verification-games snapshot
`3409afc531b0f1e0377ceafd4e0d70ebbd5a0e55`; that revision was not available
through the GitHub API when this note was written, so the link points to the
verified public worklog on `main`.

Keeping the current marginalized flux update is the narrow distinction in
the new proposal. It has not been implemented or tested. It may avoid the
historical flux/factor conditional geometry, while amplitude/factor dependence
could still reduce amplitude mixing enough to erase the computational gain.

## Order of work and acceptance evidence

First fix and independently test the stability defect, then measure screening
overhead on the unchanged ordinary rank-256 anchors. Next investigate the
small exact numerical alternatives above. Treat auxiliary augmentation as a
separate later sampler experiment, with an independent small Gaussian oracle
and a bounded representative comparison before any production-length run.

For sampler changes, report startup, warmup and retained sampling separately;
count amplitude density calls, cache refreshes and state trajectories. Compare
divergences, tree sizes, step sizes, convergence and bulk/tail effective
samples per total sampling second for amplitudes **and** flux/boundary states.
The existing short-run evidence does not support a full metric experiment or
a universal effective-sample-rate gain. Preserve the benchmark inputs,
diagonal tail, priors and source revisions so later measurements remain
comparable with the [earlier amplitude investigation](ope152_amplitude_nuts_performance.md).
