# BP1 Support-Exact Native-Quadrature Marginal Plan

## Decision purpose

This plan declares one independent normalized marginal-likelihood experiment
after the terminal results for:

- the scrambled-RQMC finite allocation bank;
- the 8- and 16-component root residual GMMs;
- the direct noisy FlowJAX likelihood; and
- the noisy sbi-NSF likelihood.

The experiment asks whether preserving the exact conditional-allocation
support and boundary law, rather than fitting a Euclidean density to its
pushforward, resolves the BP1 value and retained-mass-gradient failures.

This is not a component escalation, a larger learned flow, or a conditional
posterior estimator.  It is a deterministic support-aware transport and
Gaussian convolution.  It has a controlled quadrature limit and a forward
simulator belonging to the same finite normalized density.

No prior run supplies nodes, weights, residual factors, artifacts, model
selection, or scientific decisions.  The exact C1 case definitions and
thresholds are reused only as frozen comparators.  The new source and run
root must be fresh full-SHA paths.

## Terminology and truth

The **native truth model** is the source-pinned additive independent-Gamma
cell model.  Conditional on retained regional masses, within-region native
fractions follow the exact Dirichlet law implied by the common Gamma rate.
Independent source-pinned Gaussian measurement noise is then added.

The **basis map** is the fixed linear aggregation from native cells to
retained masses.  It is part of the representation, not a learned or
data-selected object.

The **truth likelihood** is the source-pinned high-order C1 deterministic
quadrature marginal.  It supplies likelihood, retained-mass-gradient,
evidence, and posterior-summary scores.

The **quadrature marginal simulator** samples the same finite weighted
Gaussian mixture used for density evaluation.  It is distinct from sampling
the continuous native truth, although the two agree in the quadrature limit.

Approximate evidence differences remain leakage diagnostics only.  They are
not basis weights, quadrature weights, architecture inputs, or structural
information.

## Support-exact construction

For each retained region, sort native cells by stable cell identity and build
the canonical count-balanced binary allocation tree.  Every internal node
splits its parent mass with the exact independent Beta law

```text
rho_v ~ Beta(sum alpha_left, sum alpha_right).
```

For quadrature order `O`, replace each Beta law by its normalized
`O`-point Gauss--Jacobi probability rule.  Form the tensor product over all
hidden tree nodes.  The resulting native fractions:

- are non-negative;
- sum to one within each retained region to represented precision;
- use positive normalized product weights;
- preserve the endpoint concentration of shapes below one; and
- converge weakly to the exact conditional Dirichlet allocation law.

The BP1 development matrix contains root partitions only.  The two-cell cases
therefore use `O` weighted components and the four-cell cases use `O**3`.
This experiment makes no production-scaling claim for large hidden dimension.

Let `D = diag(noise_sd**2)` and let `Q` be the complete canonical
aggregation-residual image basis in error-whitened observation coordinates.
For component `s`, let `F_s` be its projected residual per unit retained
mass and let `A` be the exact conditional-mean design.  For retained masses
`m`,

```text
r = D**(-1/2) (y - offset - A m)
z = Q.T r
mu_s(m) = F_s m.
```

The finite conditional density is

```text
p_O(y | m) =
    prod_i noise_sd[i]**(-1)
    * phi(r - Q z)
    * sum_s weight_s phi(z; mu_s(m), I).
```

It is normalized for every admissible `m`.  All component weights are
deterministic and mass-independent.  The complete residual image makes the
Gaussian orthogonal complement exact.  The evaluator uses stable
`logsumexp`, bounded chunks, and an analytic retained-mass gradient through
both the conditional mean and component means.

Forward simulation selects a component with the authenticated quadrature
weights, draws independent Gaussian image and complement noise, and reverses
the same fixed transforms.  Density and simulation therefore describe one
finite marginal approximation.

## Frozen quadrature ladder and charts

The only development ladder is

```text
O = 24, 32, 40, 48.
```

It cannot be extended after G1.  There is no learned component count,
training-sample ladder, optimizer, epoch budget, or architecture search.

For four-cell root cases, the published artifact uses the column-first
neutral-to-the-right Dirichlet chart:

```text
(cell 0 + cell 2) / total,
cell 0 / (cell 0 + cell 2),
cell 1 / (cell 1 + cell 3).
```

The frozen C1 oracle uses its source-pinned chart independently.  A
development chart audit also evaluates the row-first chart at the same
candidate order and records maximum likelihood and gradient discrepancies.
Chart choice is a computational partition only; it does not alter the exact
limit or supply scientific information.

Two-cell cases have one Beta coordinate.  Stable-cell permutation tests must
show that relabelling and consistently permuting native arrays leaves the
observation density unchanged.

## Artifact and portability contract

The artifact is canonical non-pickle data containing:

- float64 residual factors and normalized log weights;
- observation-mean design, noise scales, and residual-image basis;
- native shapes, stable cell identities, partition labels, and chart;
- quadrature order and SciPy version;
- source, driver, protocol, and context digests; and
- whole-artifact SHA-256.

Replay requires exact field, dtype, shape, context, and whole-file
authentication.  Completion markers are published last.  Cross-node replay
compares canonical artifact bytes exactly and likelihood/gradient values to
the predeclared binary64 portability tolerance.  A non-canonical residual
basis or context mismatch is a G0 hard stop.

## Gates

### G0: implementation and replay

Before any development task is submitted, G0 must pass:

- focused tests for two- and four-cell quadrature construction;
- positive normalized product weights and within-region mass conservation;
- exact Dirichlet first- and second-moment agreement;
- normalized one-dimensional observation density by independent numerical
  integration;
- analytic retained-mass gradients against central differences;
- sample shapes and simulator moments;
- complete residual-image and dense-Gaussian parity;
- stable-cell and region permutation invariance;
- row-first versus column-first convergence;
- canonical serialization, authentication, and malformed-input rejection;
- same-process, separate-process, and cross-node replay;
- experimental Ruff and focused Pyright;
- a committed bounded smoke; and
- exact Git, driver, protocol, Python, NumPy, and SciPy identities.

The smoke uses the near-Gaussian two-cell root at `O=8`.  It is operational
only and cannot publish scientific passage.

### G1: complete development matrix

G1 is the 24-task Cartesian product:

```text
regime: near_gaussian, skewed, boundary_heavy
family: two_cell, four_cell
tiling: root
quadrature order: 24, 32, 40, 48
```

Every task must authenticate its artifact and replay, verify quadrature
moments, and apply the unchanged C1 likelihood, retained-mass-gradient,
evidence, and posterior-summary gates.  The chart audit is diagnostic and
must be finite; the scientific gates already test the published artifact
against the independently constructed truth likelihood.

The pure merger may publish a common lock only if every case has a common
all-larger passing suffix of at least two orders.  The locked order is the
first order in the smallest such suffix.  Missing, malformed, failed, or
unauthenticated tasks cannot be omitted.

No common lock is terminal for this support-quadrature architecture.  The
ladder may not be extended and no learned correction may be added inside the
run.

### G2: independent simulator confirmation

Only a valid G1 lock permits 18 confirmation shards: six cases times seeds
`1877`, `4099`, and `8317`, all at the single locked order.

Each shard replays the identical deterministic density artifact and draws
131,072 complete observations from its weighted-mixture simulator.  It must:

- reproduce the artifact's analytic observation mean within five independent
  Monte Carlo standard errors per coordinate;
- reproduce its analytic covariance with maximum standardized entry error at
  most five Monte Carlo standard errors;
- produce finite likelihoods for every simulated observation;
- pass a source-pinned component-frequency chi-square aggregate with expected
  bins merged before testing so every expected count is at least 20;
- repeat every unchanged scientific gate; and
- preserve seed-independent density, evidence, and gradient values exactly.

The G2 merger also applies the source-pinned between-seed evidence-range gate
and may publish a holdout-eligible certificate only if all 18 shards pass.

### G3: protected holdout

G3 remains forbidden unless G2 publishes a passing, holdout-eligible
certificate naming every authenticated artifact.  No quadrature-order change,
chart change, threshold change, learned correction, or artifact substitution
is allowed after a protected reveal.

## Scientific invariant

This is a non-RJ marginal-likelihood approximation to one common native model.
For every fixed basis representation, the quadrature limit integrates the
same conditional native allocation and Gaussian observation kernel.  Its
exact evidence is invariant to computational chart, partition, and `K`.

Finite-order evidence differences diagnose numerical leakage.  They must
never become data-dependent basis weights, structural probabilities, or
evidence for spatial resolution.

## Stop rules

- Stop at the first hard gate.
- Preserve every partial artifact and failure.
- Never reuse an earlier run root or artifact.
- Publish completion markers last.
- Do not write to `PARIS_inversions`.
- Do not open the protected catalogue without the required passing G2
  certificate.
