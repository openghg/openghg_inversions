# BP1 Direct Marginal NLE Flow Plan

## Decision purpose

This plan declares the first neural-likelihood experiment after the terminal
root-GMM result recorded in
`rjmcmc_conditional_residual_gmm_16_component_bp1_report.md`.  Its purpose is
to learn and simulate the marginal observation model induced by a fixed basis
map while retaining a normalized conditional likelihood.

This is a new architecture and a new run family.  It does not rescue, extend,
or reinterpret either preserved GMM run.  No GMM artifact is a training input,
and the protected residual-density catalogue remains sealed.

The implementation and G0 checks may expose programming or numerical defects.
The architecture, data domains, optimizer settings, development matrix, and
scientific thresholds below become immutable when the G0 source revision is
recorded.  G1 may start only from that committed full Git SHA.

## Terminology and truth

The **native truth model** is the source-pinned additive independent-Gamma
cell model used by the six BP1 exact development cases.  Conditional on
retained region masses, native cell fractions follow the corresponding
within-region Dirichlet laws.  Measurement error is the source-pinned
independent Gaussian observation error.

The **basis map** is the fixed linear aggregation from native cell values to
retained region masses.  It is part of the model declaration, not selected
from the observations.

The **exact marginal likelihood comparator** is the deterministic quadrature
likelihood already implemented by the C1 tiny-screen definitions.  It
integrates the native allocations conditional on the retained masses.  This
observation-space density, rather than a fitted GMM or finite-bank density, is
the truth used by the likelihood, gradient, posterior-summary, and evidence
scores.

The **learned marginal simulator** is the forward sampler belonging to the
same normalized conditional flow used for density evaluation.  It is not a
posterior sampler and is not trained against protected observations.

## Normalized residual-image construction

For fixed context, let:

- `m` be the positive retained region masses;
- `A m` be the conditional observation mean;
- `D = diag(noise_sd**2)` be the fixed measurement covariance;
- `Q` be the complete canonical aggregation-residual image basis in
  `D**(-1/2)` observation coordinates;
- `r = D**(-1/2) (y - offset - A m)`;
- `z = Q.T r`; and
- `r_perp = r - Q z`.

For region `j`, let `C_j` be the exact covariance in `Q` coordinates of its
unit-mass conditional Dirichlet residual.  These matrices are computed from
the authenticated native concentration field and design, then stored and
fingerprinted with the fitted artifact.  Define

```text
V(m) = I + sum_j m_j**2 C_j
L(m) L(m).T = V(m)
u = solve(L(m), z).
```

The flow learns the remaining standardized non-Gaussian density
`q_phi(u | c(m))`.  The conditioner contains `log(sum(m))` followed, when
there is more than one retained region, by the canonical additive-log-ratio
shares.  Every conditioner coordinate is transformed by source-recorded
affine constants.

The fitted observation density is

```text
p_phi(y | m) =
    prod_i noise_sd[i]**(-1)
    * phi(r_perp)
    * q_phi(u | c(m))
    * det(L(m))**(-1).
```

It is normalized for every admissible `m`.  Gaussian measurement noise is
included in the projected training simulations; the exact Gaussian
orthogonal complement remains outside the learner.  No Monte Carlo
convolution occurs during likelihood evaluation.

Forward simulation uses the inverse construction:

1. draw `u` from the conditional flow;
2. set `z = L(m) u`;
3. draw a standard Gaussian vector and project it into the complement of
   `Q`;
4. form `r = Q z + r_perp`; and
5. return `offset + A m + noise_sd * r`.

Thus density evaluation and simulation describe one fitted marginal model.

## Pinned software and architecture

The NLE dependency is the opt-in `nle` project feature:

- FlowJAX `17.2.1`;
- the JAX and JAXlib versions resolved by the committed Pixi lock;
- JAX 64-bit mode enabled before model construction; and
- CPU execution for the BP1 comparability run.

FlowJAX `17.2.1` is the newest selected release compatible with the
repository's Python 3.10 contract.  The experiment uses
`triangular_spline_flow` with:

- standard multivariate-normal base distribution;
- conditional dimension equal to the retained-region count;
- 8 flow layers;
- 8 rational-quadratic-spline knots;
- `tanh_max_val = 3.0`;
- `invert = True`, prioritizing likelihood evaluation; and
- the library's pinned default triangular initialization.

The BP1 matrix contains only root partitions, so residual ranks are one for
the two-cell family and three for the four-cell family, and the conditioner
contains only standardized log total mass.  The artifact and tests must
nevertheless reject incorrect mass order, dimensions, nonpositive masses,
and context identities.

## Simulator domains

Training, internal early stopping, model-selection validation, development
reporting test, confirmation, and protected holdout are distinct domains.
Protected-domain identifiers cannot be requested by any G0, G1, or G2 code
path.

For each public domain:

1. construct a fresh domain-keyed scrambled-Sobol bank of native conditional
   Dirichlet residuals;
2. construct a separately scrambled Sobol net for total mass and projected
   Gaussian measurement noise;
3. map total mass uniformly in log space over the closed exact-development
   grid range expanded by `0.5` natural-log units at each end;
4. transform Gaussian coordinates with the binary64 normal inverse after
   clipping only exact Sobol endpoints to the nearest open-unit-interval
   binary64 values; and
5. whiten the projected noisy residual with the exact `L(m)`.

All seeds, arrays, source identities, and preprocessing constants are
checksummed.  Whole draws remain in one domain; no row from validation or
test may enter optimization.

## Pinned fitting rule

Each candidate is fit by conditional maximum likelihood with:

- two deterministic initialization attempts derived from the domain seed;
- Adam learning rate `5e-4`;
- batch size `1024`;
- at most `100` epochs;
- a `10%` split taken only from the training domain for optimizer early
  stopping;
- patience `10`;
- return of the best early-stopping state; and
- no resume across attempts or training sizes.

Both attempts must finish with finite parameters and finite independent
validation NLL.  The attempt with minimum independent model-selection
validation NLL is published.  Ties use the lower initialization index.

The development training-size ladder and independent bank sizes remain:

```text
training:  4,096; 16,384; 65,536; 262,144
validation: 65,536
test:      131,072
```

## Gates

### G0: implementation and replay

G0 must pass before any development matrix task is submitted:

- experimental unit tests for covariance construction, normalized density,
  simulator shape and moments, strict serialization, artifact
  authentication, and malformed inputs;
- conditional-flow `sample_and_log_prob` agreement with a separate
  `log_prob` evaluation to absolute tolerance `1e-10` in float64;
- exact reconstruction of the density/simulator preprocessing maps to
  absolute tolerance `1e-10`;
- same-node and cross-process artifact replay;
- Ruff on changed experimental code, tests, and committed drivers;
- focused Pyright on the same Python files;
- a committed smoke run; and
- recorded Python, NumPy, SciPy, JAX, JAXlib, FlowJAX, Equinox, Optax, and
  Paramax versions.

Completion markers are written last.  A partial or unauthenticated G0 is a
hard stop.

### G1: six-case development ladder

G1 contains the complete 24-task Cartesian product of:

```text
regime: near_gaussian, skewed, boundary_heavy
family: two_cell, four_cell
tiling: root
training size: 4,096; 16,384; 65,536; 262,144
```

Each task fits only the development-selection seed and must publish both
initialization records, the selected authenticated flow, all domain
identities, density-generalization diagnostics, and the unchanged C1
scientific scores.

The simulator-test NLL gap passes when its absolute validation-versus-test
difference is at most
`max(0.02 * residual_rank, 5 * pooled_NLL_MCSE)`.

The scientific thresholds are exactly the source-pinned C1 thresholds for
conditional log likelihood, finite-difference coordinate gradient, log
evidence, and posterior summaries.  Neither approximate evidence nor its
error is an input to the basis map, flow conditioner, architecture, or
training weights.

The merger may publish a common lock only when all six cases have a common
all-larger passing suffix of at least two training sizes.  The locked size is
the smallest first size of such a suffix.  Missing, malformed, or failed
tasks are not silently excluded.

If no common lock exists, this declared NLE architecture stops.  The merger
must preserve the failures, publish no lock, and leave G2 and the protected
catalogue closed.

### G2: confirmation

Only a valid G1 common lock permits the 18 confirmation shards: six cases
times three source-pinned confirmation seeds, all at the single locked
training size.  Every shard must pass fitting, generalization, replay, and
unchanged scientific gates.

The G2 merger additionally applies the unchanged between-bank log-evidence
range threshold.  Approximate evidence differences are leakage diagnostics,
not structural information.

### G3: protected holdout

G3 remains forbidden unless G2 publishes a passing, holdout-eligible
certificate naming every authenticated selected artifact.  No retraining,
retuning, architecture change, threshold change, or artifact substitution is
allowed after a protected reveal.

## Scientific invariant

The target is one common native model marginalized after a fixed linear basis
map.  Its exact marginal likelihood is invariant to computational partition
and to any numerical catalogue or training size.  The NLE is a
fixed-partition approximation to that marginal likelihood.  Differences
caused by training seed, training size, compute layout, or flow
approximation are errors to diagnose; they must never become
observation-dependent basis weights or evidence for a partition.

## Execution and preservation

G0, the complete G1 matrix, and the merger run in that order from a fresh
detached full-SHA worktree and fresh run root.  G2 is conditional on the G1
lock.  Every artifact, stdout/stderr log, scheduler record, failure,
completion marker, merger report, source SHA, and checksum is preserved.
Nothing is written to `PARIS_inversions`.
