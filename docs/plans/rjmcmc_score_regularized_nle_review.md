# Review of the score-regularized marginal NLE experiment

## Status

The BP1 N1 result on `codex/rjmcmc-score-regularized-nle` must not be treated
as a scientific architecture hard stop. The implementation and run completed,
but the tiny-domain simulator did not generate the declared independent joint
law. Existing N1 artifacts remain useful provenance for the failed
experiment, not evidence against score-regularized NLE.

Reviewed identities:

```text
reported branch head:        c4f674a17a587f9e2e89488b9c541c8f61667edd
scientific execution source: 475d5db6026a8472fd3c44eac2e0d2369686c78b
initial plan source:          d044b082c18a63e815aacaa41b2cfc139a74e2af
```

The terminal report and immutable run root should be preserved unchanged.
Any correction requires a new source identity and fresh simulator/fitted
artifacts.

## Blocking simulator defect

The declared simulator is

\[
T\sim\operatorname{Gamma}(a,b),\qquad
\xi\sim P_{\rm allocation},\qquad
\epsilon\sim\mathcal N(0,I),
\qquad T\perp\xi\perp\epsilon.
\]

`score_regularized_flow_tiny_domains.py` instead constructs allocation,
root-total, and Gaussian-noise catalogues from separately restarted scrambled
Sobol engines and combines equal row indices. Distinct seeds authenticate
different one-dimensional nets; they do not make their row-aligned point
sets a low-discrepancy sample from the product distribution.

For the frozen \(q=1\) near-Gaussian case, the empirical correlations between
the latent root-total uniform and Gaussian-noise uniform are:

```text
training domain:                  +0.750000
model-selection-validation:       -0.750000
development-reporting-test:       -0.574219
```

The lower-left quadrant frequencies are `0.5`, `0.0`, and `0.0`, rather than
the product-law value `0.25`. These discrepancies persist at the large
power-of-two catalogue sizes. The independent review found similarly large
positive or negative correlations in all three two-cell regimes.

Consequences:

1. the training, selection, and reporting domains each approximate a
   different incorrect copula;
2. marginal moment tests for \(T\), \(\xi\), and \(\epsilon\) can all pass
   while the joint simulator is wrong;
3. the simulator-score conditional-expectation identity used by the loss no
   longer has the implemented target, because the effective latent law
   changes with the row-aligned root total; and
4. the one-of-24 generalization result, non-monotone training-size behaviour,
   and large retained-mass-gradient errors cannot be interpreted as failures
   of the intended flow architecture.

The simplest corrected NLE training simulator should use genuinely
independent PCG64 streams. A joint scrambled-Sobol net containing every
allocation, mass, and noise coordinate is also valid for the tiny cases, but
does not transfer simply to the full PARIS dimension. QMC is not required for
NLE simulation.

Add tests of joint two-dimensional projections, quadrant occupancy, and
cross-covariance for every public domain. Seed inequality and marginal moment
tests are insufficient.

## Invalid RQMC uncertainty calculation

The development generalization gate estimates

\[
\operatorname{MCSE}=\operatorname{sd}(v_s)/\sqrt S
\]

from the rows of one scrambled-Sobol realization. Those rows are dependent,
so this is not an IID Monte Carlo standard error and is not a valid
between-scramble RQMC error estimate. Use several independent joint scrambles
and their between-replicate variance, or replace the MCSE-derived gate with a
separately justified deterministic tolerance.

## Boundary-heavy oracle and weighting defects

Two additional problems make the boundary-heavy scientific metrics
unreliable independently of the simulator coupling.

The inherited checkerboard mask is applied before renormalizing the prior and
exact-posterior weights used for likelihood errors. In the boundary-heavy
two-cell case it retains about 36.46% of the prior mass but only
\(2.96\times10^{-7}\) of the exact posterior mass and excludes the posterior
mode. The reported “posterior-weighted p99” is therefore conditional on a
negligible, nonrepresentative posterior subset. This explains why a
prior-weighted median error above 30 nat can coexist with a much smaller
reported posterior-weighted p99; the two summaries are not merely differently
weighted views of adequate common support.

The boundary-heavy two-cell evidence/posterior reference is also not
established as an exact oracle. Re-evaluation of its outer quadrature at
orders 32, 64, 96, 128, 192, and 256 gave log evidences of approximately:

```text
-18.382, -4.326, -1.547, -4.411, -1.562, -1.217 nat
```

Posterior means and standard deviations likewise did not converge. The
declared 64-node quadrature reference cannot support 0.05-nat evidence or
tight posterior gates in this regime. Resolve the tiny boundary oracle with
an independently converged support-aware method before using it for NLE
certification.

## What remains useful

The reusable mathematical and software work is substantial:

- the normalized hybrid likelihood and explicit Gaussian complement;
- the simulator component-score and fixed-observation chain-rule algebra;
- the forward-JVP/reverse-parameter differentiation recovery;
- literal zero-mass and Gaussian limits;
- float64 serialization and separate-process replay;
- create-only artifacts and protected-path controls; and
- focused normalization, permutation, score, optimizer, and malformed-input
  tests.

The branch implemented about 15,000 lines across 43 changed files, including
both an earlier likelihood-only flow and the score-regularized successor.
Before promotion, split reusable score-flow code from archived experiment
drivers and reports. Executable examples should not import private helpers
from other experiment scripts.

## Interpretation and protocol lessons

Even after correcting the simulator, the completed run would have tested only
one frozen configuration:

- an eight-layer rational-quadratic-spline flow;
- different one- and three-dimensional FlowJAX specializations;
- one scalar root-total condition;
- a fixed 1:1 NLL/partial-mass-score loss;
- two initializations;
- one Adam schedule and early-stopping rule; and
- no observation-score supervision.

Failure of that configuration would not exhaust conditional flows,
score-regularized NLE, multi-root factorization, or conditional-row
constructions. In particular, the fixed-\(y\) mass-gradient gate also depends
on the observation score, while training supervised only the partial
root-mass score at fixed standardized \(x\).

The experiment protocol authenticated the wrong simulator extremely well.
Hashes, exact replay, immutable matrices, and hard gates protect provenance;
they do not validate the scientific joint distribution. The current process
also applies certification-level ceremony during model development and
forbids informative ablations after the first result.

Use two levels from now on:

1. **Exploratory public-oracle development.** Keep data domains separate and
   retain every run, but allow compact iterations over score scaling,
   likelihood-only pretraining, score curricula, direct fixed-\(y\)
   supervision, observation-score supervision, optimizer diagnostics, and
   additional initializations. Report decomposed losses and histories.
2. **Frozen promotion.** Once one or two candidates are credible, apply the
   complete six-case matrix, independent seeds, exact likelihood/gradient/
   evidence/posterior thresholds, create-only certificates, and protected
   holdout.

Scientific target, candidate configuration, runtime environment, and
execution provenance should have separate identities. Compiler flags and
autodiff scheduling belong to execution identity, not the mathematical
target.

## Recommended next sequence

1. Preserve and relabel the existing N1 report as an invalid-target and
   invalid-oracle experiment.
2. Correct the simulator with IID streams and add joint-independence tests.
3. Replace the single-net MCSE rule.
4. repair and independently converge the boundary-heavy oracle and remove or
   justify the posterior-excluding metric mask;
5. run a cheap single-case/single-batch overfit and a small NLL-versus-score
   ablation before another full matrix.
6. Publish loss histories, best epochs, initialization spread, decomposed
   validation risks, and a compact machine-readable result with the report.
7. Only after the corrected tiny programme works should target-rank PARIS
   compilation or training resume.
