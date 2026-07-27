# BP1 Marginal sbi-NSF NLE Plan

## Decision purpose

This plan declares one independent neural-likelihood experiment after the
terminal root-GMM and direct FlowJAX results recorded in:

- `rjmcmc_conditional_residual_gmm_16_component_bp1_report.md`; and
- `rjmcmc_marginal_nle_flow_bp1_report.md`.

The experiment asks whether an established autoregressive neural-spline
likelihood can learn and simulate the observation marginal induced by a fixed
linear basis map with sufficiently accurate retained-mass derivatives.

This is a new architecture and run family. It does not extend the GMM
component ladder or the failed direct triangular flow. Neither preserved run
supplies training samples, weights, parameters, checkpoints, or model-selection
information. The exact development definitions and unchanged scientific gates
are reused only as comparators.

The implementation and G0 checks may expose programming or numerical defects.
The architecture, public data domains, optimizer, development matrix, and
scientific thresholds below become immutable when the G0 source revision is
recorded. G1 may start only from that committed full Git SHA.

## Terminology and truth

The **native truth model** is the source-pinned additive independent-Gamma cell
model used by the six BP1 exact development cases. Conditional on a retained
root mass, native cell fractions have the corresponding Dirichlet law.
Source-pinned independent Gaussian measurement error is then added.

The **basis map** is the fixed linear aggregation from native cell values to
the retained mass. It is part of the model declaration and is never selected
from realized observations.

The **exact marginal likelihood comparator** is the deterministic C1
quadrature likelihood. It integrates the native allocations conditional on
the retained mass. It supplies the likelihood, gradient, evidence, and
posterior-summary truth scores.

The **learned marginal simulator** is the forward sampler belonging to the
same normalized conditional neural spline used for density evaluation. It is
not a posterior estimator.

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
unit-mass conditional Dirichlet residual. Define

```text
V(m) = I + sum_j m_j**2 C_j
L(m) L(m).T = V(m)
u = solve(L(m), z).
```

The estimator learns the remaining normalized density
`q_phi(u | c(m))`. The conditioner contains `log(sum(m))` followed, for
non-root extensions, by canonical additive-log-ratio shares. Conditioner
coordinates use source-recorded affine transforms. The BP1 matrix contains
only root partitions and therefore has a one-dimensional conditioner.

The fitted observation density is

```text
p_phi(y | m) =
    prod_i noise_sd[i]**(-1)
    * phi(r_perp)
    * q_phi(u | c(m))
    * det(L(m))**(-1).
```

This is normalized for every admissible `m`. The projected Gaussian
measurement noise is part of the training simulator; the exact Gaussian
orthogonal complement remains outside the learner. No stochastic convolution
occurs during likelihood evaluation.

Forward simulation draws `u` from the authenticated NSF, applies `L(m)`,
independently draws the exact Gaussian complement, reverses the residual-image
map, and adds the conditional mean. Density evaluation and simulation
therefore describe one fitted marginal model.

## Pinned software and architecture

The opt-in project feature is extended with:

- `sbi == 0.26.1`;
- the PyTorch and `nflows` versions resolved by the committed Pixi lock;
- CPU execution;
- `torch.float64` inputs, parameters, preprocessing buffers, likelihoods, and
  gradients; and
- deterministic Torch algorithms with one intra-op and one inter-op thread.

The density estimator is built by the pinned `sbi` likelihood factory with:

- model `nsf`;
- standard-normal base density;
- autoregressive rational-quadratic-spline transforms;
- 8 transforms;
- 16 spline bins;
- 128 hidden features;
- no library z-scoring, because authenticated exact preprocessing is applied
  outside the estimator; and
- the pinned library defaults for settings not listed above.

This is deliberately distinct from the failed FlowJAX triangular spline.
There is no architecture search. No MAF, MDN, ensemble, alternate hidden
width, alternate number of transforms, or post-G1 capacity escalation is
permitted inside this run.

## Simulator domains

Training, optimizer validation, model-selection validation, development test,
confirmation, and protected holdout are disjoint whole-draw domains. Protected
identifiers cannot be requested by G0, G1, or G2 code paths.

For each public domain:

1. draw a domain-keyed scrambled-Sobol bank from the exact conditional
   Dirichlet allocation law;
2. use a separately scrambled Sobol net for total mass and projected Gaussian
   noise;
3. transform total-mass coordinates through the exact root Gamma inverse CDF,
   so `(m, y)` are ancestral draws from the declared native joint model;
4. transform Gaussian coordinates with the binary64 normal inverse after
   clipping only exact Sobol endpoints to the nearest open-unit-interval
   binary64 values;
5. apply the exact conditional moment whitening above; and
6. standardize `log(m)` with its analytic Gamma log-mean and log-SD,
   `digamma(alpha_total) - log(rate)` and
   `sqrt(polygamma(1, alpha_total))`.

The training objective is therefore prior-predictive conditional maximum
likelihood, as required by one-round NLE. All seeds, Sobol coordinates, arrays,
native parameters, and transforms are checksummed. Whole draws remain in one
domain.

## Pinned fitting and serialization rules

Each candidate is fit with:

- two deterministic initialization attempts;
- Adam learning rate `3e-4`;
- weight decay `1e-6`;
- batch size `2048`;
- at most `200` epochs;
- a separately generated 65,536-draw optimizer-validation domain;
- patience `20`;
- restoration of the lowest optimizer-validation-NLL state; and
- no resume across attempts or training sizes.

Both attempts must finish with finite parameters and finite independent
model-selection validation NLL. The lower model-selection validation NLL is
published; exact ties use the lower initialization index.

The development sizes are:

```text
training:                  4,096; 16,384; 65,536; 262,144
optimizer validation:     65,536
model-selection validation: 65,536
development test:        131,072
```

The selected Torch state dictionary is serialized without pickle. Tensor
names, shapes, dtypes, and canonical little-endian bytes are stored in sorted
name order behind canonical JSON metadata and a format magic value. Trainable
state is float64; fixed index buffers are int64. The envelope binds
the source revision, architecture, preprocessing, simulator context, domain
identities, and a SHA-256 digest over all bytes. Deserialization reconstructs
the estimator from the pinned factory and requires an exact state-key match.

## Gates

### G0: implementation, differentiation, and replay

G0 must pass before any G1 task is submitted:

- focused tests for normalized log density, sampler shapes and moments,
  analytic retained-mass gradients, malformed inputs, canonical
  serialization, and authentication;
- `sample_and_log_prob` agreement with separate `log_prob` evaluation to
  absolute tolerance `1e-6` in float64. This pre-G0 tolerance reflects the
  pinned `nflows` inverse/forward spline round-trip error observed during
  implementation (`4.31e-7` maximum on the rank-three unit fixture);
- Torch autograd mass gradients versus central finite differences to scaled
  error at most `1e-6` on a fitted smoke artifact;
- exact preprocessing-map reconstruction to absolute tolerance `1e-10`;
- same-process, separate-process, and cross-node artifact replay;
- Ruff on changed experimental code, tests, and drivers;
- focused Pyright on the same Python files;
- a committed operational smoke; and
- recorded Python, NumPy, SciPy, Torch, `sbi`, `nflows`, and associated
  serialization dependency versions.

Completion markers are written last. A partial, unauthenticated, non-normalized,
or non-differentiable G0 is a hard stop.

### G1: six-case development ladder

G1 is the complete 24-task Cartesian product:

```text
regime: near_gaussian, skewed, boundary_heavy
family: two_cell, four_cell
tiling: root
training size: 4,096; 16,384; 65,536; 262,144
```

Every task publishes both initialization records, the selected authenticated
NSF, all public-domain identities, density-generalization diagnostics, and the
unchanged C1 scientific scores.

The model-selection-validation versus development-test NLL gap passes when its
absolute difference is at most
`max(0.02 * residual_rank, 5 * pooled_NLL_MCSE)`.

Scientific thresholds are exactly the source-pinned C1 thresholds for
conditional log likelihood, finite-difference coordinate gradient, log
evidence, and posterior summaries. The repeat-bank evidence threshold remains
reserved for G2. Evidence and evidence error are diagnostics only; neither is
an input to training, the conditioner, the basis map, or any architecture
choice.

The pure merger may publish a common lock only if every case has a common
all-larger passing suffix of at least two training sizes. The locked size is
the first size of the smallest such suffix. Missing, malformed, failed, or
unauthenticated tasks cannot be omitted.

No common lock is terminal for this declared sbi-NSF architecture. The merger
must preserve all failures, publish no lock, and leave G2 and the protected
catalogue closed.

### G2: confirmation

Only a valid G1 lock permits 18 confirmation shards: six cases times three
source-pinned confirmation seeds at the single locked size. Every shard must
pass fitting, density generalization, replay, and all unchanged scientific
gates. The merger also applies the unchanged between-bank log-evidence range
threshold.

### G3: protected holdout

G3 remains forbidden unless G2 publishes a passing, holdout-eligible
certificate naming every authenticated selected artifact. No retraining,
retuning, architecture change, threshold change, or substitution is allowed
after any protected reveal.

## Scientific invariant

The target is one common native model marginalized after a fixed linear basis
map. Its exact limit is invariant to computational partition, basis
representation, training size, and compute layout. Approximate likelihood and
evidence differences are leakage diagnostics, not structural information, and
must never become data-dependent basis weights.

## Execution and preservation

G0, the complete G1 matrix, and the merger run in order from a fresh detached
full-SHA worktree and fresh run root. G2 is conditional on the G1 lock. Every
artifact, log, scheduler record, failure, completion marker, merger report,
source SHA, and checksum is preserved. Nothing is written to
`PARIS_inversions`.
