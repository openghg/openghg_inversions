# Bocquet Projection Validation

## Purpose

This is the operational plan for validating the exact Gaussian projection
guarantees described in `docs/reports/rhime_bocquet_reduced_gaussian.md`. It is
separate from the mathematical reference and from the longer-term plan for
posterior inference over partitions.

The primary demonstration should be a working semi-synthetic inversion using
real TAC/MHD transport sensitivities, a native-grid spatial prior, and synthetic
observation anomalies. It should establish that:

1. a directly fitted projected analysis equals the same projection of the
   native Gaussian analysis;
2. native-resolution posterior summaries can be computed in parallel spatial
   blocks and assembled into full-domain maps; and
3. adaptive Equation 45 partitions are useful posterior-compression decisions,
   while DFS and Fisher provide data-independent comparisons.

This work remains experimental. It must not change production RHIME or
`fixedbasisMCMC`.

## Implementation status (2026-07-17)

Completed on the experimental branch:

- a dense NumPy oracle for general positive-definite \(B\), non-diagonal
  \(R\), nonzero prior means, arbitrary full-row-rank restrictions, and fixed
  regional prolongations;
- exact direct-versus-projected posterior mean/covariance tests, innovation
  closure, residual independence, and positive-semidefinite unresolved
  covariance checks;
- stable weak-signal DFS, additive DFS/Fisher/Equation 45 dyadic node scores,
  and exact fixed-count dynamic programming for each additive objective;
- chunked native posterior mean and marginal-variance maps for the current
  independent-relative-error model;
- named partition-level DFS, base-error Fisher, aggregation-aware Fisher,
  Equation 45, and projected Bayesian-information diagnostics;
- a native-resolution semi-synthetic TAC/MHD report comparing root,
  land/ocean, rectangular inner/outer, and three fixed-count adaptive
  partitions at
  `docs/plans/figures/bocquet_projection_validation/`; and
- a matrix-free separable exponential covariance operator with optional
  land/ocean class blocking, ported with provenance for the first correlated
  prior experiment.

Still to do:

- run and profile the correlated-prior operator in a TAC/MHD inversion rather
  than only its exact small-grid parity tests;
- repeat the data-adaptive experiment across synthetic realizations and report
  partition stability and training-versus-holdout optimism; and
- add country/region masks when an aligned public fixture is chosen.

The mixed-resolution block reconstruction below is deliberately deferred. The
native posterior API retains chunked output so the experiment can be resumed
without changing the current interfaces.

## Statistical model

Use one native linear-Gaussian model throughout:

\[
x\sim\mathcal N(\mu,B),
\qquad
y\mid x\sim\mathcal N(Hx,R).
\]

The observation-space covariance and native posterior are

\[
S=HBH^T+R,
\]

\[
m_a=\mu+BH^TS^{-1}(y-H\mu),
\qquad
P_a=B-BH^TS^{-1}HB.
\]

For every declared full-row-rank restriction \(\Gamma\), construct

\[
B_\Gamma=\Gamma B\Gamma^T,
\qquad
\Lambda_\Gamma=B\Gamma^TB_\Gamma^{-1},
\]

\[
B_{c,\Gamma}=B-\Lambda_\Gamma B_\Gamma\Lambda_\Gamma^T,
\qquad
R_\Gamma=R+HB_{c,\Gamma}H^T.
\]

For a nonzero native prior mean, the exact reduced likelihood includes an
offset as well as the transformed design and covariance:

\[
y\mid\alpha
\sim
\mathcal N\!\left(
H\mu+H\Lambda_\Gamma(\alpha-\Gamma\mu),
R_\Gamma
\right).
\]

Equivalently, its affine intercept is
\(H(\mu-\Lambda_\Gamma\Gamma\mu)\). Omitting this term is valid only for a
centered anomaly model or a compatible prior mean, and would otherwise break
the direct-versus-projected posterior check.

The direct reduced inversion must reproduce

\[
m_{\Gamma,a}=\Gamma m_a,
\qquad
P_{\Gamma,a}=\Gamma P_a\Gamma^T.
\]

The marginal data distribution remains
\(\mathcal N(H\mu,R+HBH^T)\) for every \(\Gamma\). Representation selection is
therefore a compression or reporting decision, not ordinary model evidence.

## Data and covariance inputs

Use the complete available TAC/MHD test-data period rather than an artificially
coarsened search grid:

- real aligned footprint-times-prior-flux columns for the native EUROPE grid;
- the vendored UKMO land/ocean classes at
  `openghg_inversions/basis/algorithms/country-EUROPE-UKMO-landsea-2023.nc`;
- retained observation uncertainty inputs, with every additional model-error
  term stated explicitly;
- synthetic source anomalies and observation anomalies generated through the
  real transport design, avoiding the known corrupted stored boundary
  contribution; and
- a fixed random seed and an artifact manifest recording sites, times, units,
  filtering, covariance parameters, and land/ocean treatment.

Start with the repository's independent-relative-error covariance because it
already has an additive reference implementation. Add the distance-based
covariance as the first correlated-prior experiment.

Relevant prototype sources to collect into the experimental repository boundary
before relying on them are:

- `verification-games/src/verification_games/grid_covariance.py`:
  separable exponential covariance application and projected grid covariance;
- `verification-games/src/verification_games/rhime_calibration/analytic.py`:
  analytic Gaussian inversion with dense prior covariance; and
- `verification-games/scripts/run_controlled_aggregation_error_coarsening.py`:
  exact conditional reduction and full aggregation-covariance comparison.

The implementation must not retain a runtime dependency on the private
`verification-games` checkout. Port the smallest reusable mathematical helpers,
with provenance and focused tests, or define a stable public adapter boundary.

## Experiment 1: small exact algebra oracle

Before using the full grid:

1. Construct a small positive-definite dense \(B\), non-diagonal \(R\), and
   synthetic \(H\).
2. Enumerate several aggregation, overlapping-summary, and mixed-resolution
   restrictions.
3. Compare direct reduced posterior moments with
   \(\Gamma m_a\) and \(\Gamma P_a\Gamma^T\).
4. Verify innovation covariance invariance and positive-semidefinite
   aggregation covariance.
5. Include both literal aggregate restrictions and fixed piecewise-regional
   prolongations so their different coefficient semantics are tested.

This is the CI-level correctness oracle. It is not the scientific
demonstration.

## Experiment 2: native-grid analytic reference

Compute the native posterior through the observation-space factorization of
\(S\), without constructing a dense native \(P_a\):

1. Factor \(S\) once.
2. Compute \(S^{-1}(y-H\mu)\).
3. Apply \(BH^T\) in native-grid chunks to obtain posterior-mean blocks.
4. Compute posterior marginal variances in chunks from
   \(B-BH^TS^{-1}HB\).
5. Retain an operator or factor representation for covariance queries spanning
   multiple blocks.

Where the prior supports efficient sampling, generate coherent full-domain
posterior draws by conditional Gaussian simulation:

\[
x_a^{(s)}
=x_0^{(s)}
+BH^TS^{-1}\left[y-Hx_0^{(s)}-\epsilon_0^{(s)}\right].
\]

This native result is the oracle for all reduced analyses. The main performance
measure is wall time and peak memory for the common observation-space
factorization and for each output block.

## Experiment 3: mixed-resolution block consistency (optional)

This is currently lower priority than the projection oracle, native TAC/MHD
reference, and objective comparisons. The native posterior implementation
should retain a block/chunk interface so this experiment remains possible, but
the first proof of concept need not produce stitched block analyses.

Partition the spatial output into native-resolution blocks, initially
\(32\times32\) grid locations with smaller edge blocks.

For each block \(b\):

1. Retain every native variable inside \(b\).
2. Either marginalize everything outside \(b\), or retain declared coarse
   summaries outside it and marginalize only their unresolved complement.
3. Perform the exact reduced inversion using the induced prior and aggregation
   covariance.
4. Compare the block posterior mean and covariance with the corresponding
   marginal of the native oracle.

Independent block calculations should reproduce and can be assembled into:

- the full posterior-mean map;
- pointwise posterior-standard-deviation maps; and
- within-block posterior covariance.

They do **not** by themselves recover cross-block covariance or coherent
full-domain draws. Derived totals spanning blocks must use the retained native
covariance operator, direct projected inference for that total, or coherent
conditional simulations.

Running a complete inversion independently for every block is a validation
method, not necessarily the efficient implementation. The production-quality
analytic route should reuse the common factorization of \(S\) and parallelize
only blockwise applications and output assembly.

## Experiment 4: requested and adaptive summaries

Validate several representation classes against the same native oracle:

1. Prespecified land/ocean totals.
2. Prespecified country or regional totals where an aligned mask is available.
3. A user-specified rectangular inner region plus its outer complement.
4. Fixed-count dyadic partitions optimized using DFS.
5. Fixed-count dyadic partitions optimized using Equation 45.
6. Fisher-selected partitions under the additive Equation 36 approximation.

For every representation, compare direct reduced posterior moments with the
projected native result. For Equation 45, report both the selected
posterior-mean update retained and the full Bayesian KL retained by the same
partition; these are related but distinct objectives.

## Data-adaptive evaluation

Equation 45 is an adaptive posterior-compression rule. It is coherent as a
Bayesian decision conditional on the observed data, but the maximized training
criterion is optimistic and the selected geography is exploratory.

Use blocked site/time splits that respect relevant temporal and transport
dependence:

1. Select \(P\) on training blocks.
2. Report its training Equation 45 value.
3. Evaluate held-out retained-update, projected-KL, and compression metrics.
4. Compare with a prespecified partition and data-independent DFS/Fisher
   selections.
5. Repeat across semi-synthetic realizations to measure partition stability,
   interval coverage for prespecified functionals, and training-versus-holdout
   optimism.

With the exact aggregation covariance, the conditional predictive distribution
of held-out observations given training observations is inherited from the same
native Gaussian model and is therefore partition-invariant. Held-out predictive
log density and standardized residuals remain useful covariance-closure checks,
but they cannot rank exact representations. They become discriminating model
checks only on the separate reduced-model track where the unresolved complement
is omitted or approximated.

Cross-fitting evaluates the adaptive pipeline but does not automatically
produce one calibrated posterior because different folds can select different
representations.

## Dynamic programming and dense covariance

For diagonal or whitened \(B\), Equations 45 and 38 reduce to additive node
scores in the current canonical dyadic dictionary. Use exact fixed-count
dynamic programming as the optimization oracle and compare greedy and SLS
results with its frontier.

For dense \(B\) with geographic semantics, the reduced covariance is generally
dense and the score is globally coupled. The scalar-node DP is not exact. Start
with direct objective evaluation and profile:

- covariance construction;
- \(K\times K\) factorization;
- repeated partition evaluations; and
- memory for covariance operators and candidate summaries.

Only then investigate cached pairwise tile interactions, sparse precision,
separable structure, or incremental Cholesky updates. Do not whiten silently:
whitening changes the meaning of the selected coordinates.

## Generalized model-selection interpretation

Ordinary Bayesian model selection cannot distinguish exact Bocquet
representations because they induce the same \(p(y)\). An optional generalized
Bayesian formulation is

\[
\pi_\beta(P\mid y)
\propto
\pi(P)\exp\{\beta J_P(y)\},
\]

or, for the full lifted-posterior criterion,

\[
\pi_\beta(P\mid y)
\propto
\pi(P)
\exp\left\{-\beta
D_{\mathrm{KL}}\!\left[p(x\mid y)\,\|\,q_P(x\mid y)\right]\right\}.
\]

This is a Gibbs posterior over compression actions, not a posterior derived
from competing generative models. The temperature \(\beta\) and prior
\(\pi(P)\) encode decision preferences. It may be useful for randomized
compression or uncertainty over reports, but it is deferred until the
deterministic optimization and calibration experiments are complete.

A conventional model-selection problem requires \(P\) to define a genuinely
different prior or likelihood, for example a low-rank regional field with the
unresolved complement omitted or separately approximated. That is a different
model from the exact Bocquet projection and remains on the partition-inference
track.

## Deliverables and acceptance criteria

1. Public experimental Gaussian covariance and projection helpers with focused
   algebra tests and source provenance.
2. A reproducible TAC/MHD semi-synthetic native-grid inversion report.
3. Blockwise mean and variance maps matching the native oracle within declared
   numerical tolerances.
4. Direct-versus-projected posterior parity for every requested representation.
5. DFS, Fisher, Equation 45, and full projected-KL metrics reported separately.
6. Exact DP comparison wherever additivity assumptions hold.
7. Held-out and repeated-simulation diagnostics for data-adaptive selection.
8. Timing and peak-memory results sufficient to decide whether dense geographic
   covariance needs specialized updates.

Stop and investigate before scientific interpretation if innovation covariance
is not invariant, aggregation covariance has materially negative eigenvalues,
direct and projected moments disagree beyond tolerance, or selected results
depend strongly on undocumented filtering or covariance regularization.
