# Full-tiling topology plus HMC

## Status

The core sampler, dedicated durable checkpoint schema, native real-data
driver, and HPC validation plan are implemented on
`codex/rjmcmc-compound-hmc`. Three frozen-input calibration attempts now
separate representation correctness from metric generalization:

- the first H2 attempt at `88b12a5` stopped before transition one because the
  fresh physical state was not an exact log/exp replay boundary;
- `fe9e546` fixed that defect without relaxing the audit;
- the diagonal-metric H2 passed at \(K=50\) but had no admissible candidate at
  \(K=250\);
- the bounded H2b refinement selected
  \(\epsilon=0.08409,L=5\), but a held-out \(K=250\) topology had acceptance
  0.031 and 42 divergences.

The diagonal H2b result was therefore a certified hard stop, not a prompt for
further step-size refinement. H2c then tested a static,
permutation-invariant two-eigenvalue leaf metric that separated the normalized
common log-mass direction from centered log-mass contrasts. H2c also reached a
certified hard stop at both \(K=50\) and \(K=250\): no frozen-grid candidate
served every development topology.

The next phase is specified in
[`rjmcmc_topology_conditioned_hmc_next_phase.md`](rjmcmc_topology_conditioned_hmc_next_phase.md).
It replaces full permutation invariance by permutation equivariance and builds
a topology-conditioned Euclidean metric from fixed reference curvature. The
metric changes with the tiling but remains constant within each HMC
trajectory, so this does not require Riemannian HMC.

The first H2d source candidate, `7f7b150`, failed its D0 finite-binary64
structural audit and must not be resumed. Four forward-valid proposals among
10,004 at \(K=50\) had no representable reverse physical fraction. Commit
`e6199150e680d43e6e3c1388db45773c5337802a` replaced that path with exact
involutions of the authoritative log-mass coordinates and changed the
schedule identity. Focused local checks passed, but the BP1 rerun was
interrupted before establishing a D0 result. No H2d calibration certificate
currently exists. Continue from
[`rjmcmc_bp1_handover.md`](rjmcmc_bp1_handover.md).

The failed-run evidence is retained under
`/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/88b12a5717d4b490b6ccd67986c20c2a99094fed`;
its report SHA-256 is
`27027ee2c4393ff8e1d399c5afe4d0b751ff9665956bf303aabc756086279a4d`.
The corrected diagonal run and H2b hard-stop evidence are retained under
`/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/fe9e546ab57a6b0ff852057e0e6afa13725a5419`;
the original H2 report SHA-256 is
`b149c99ed48071b2fce6873603e4eb138915bf0005d0f8189fcbeefb3779dd5e`.
The H2c evidence is retained under
`/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/7a1a1cc673a4b6a6ce0ed7b5123494ebd205b467`,
including `report/H2C_RESULTS.md` and
`calibration/H2C_HARD_STOP.json`.

The fixed-basis reference experiments established that the local continuous
kernel, rather than the Gamma root model itself, was the main fixed-topology
bottleneck:

- fixed-basis local sampling gave root bulk ESS of about 31 at both
  \(K=50\) and \(K=250\);
- diagonal NumPyro NUTS raised root bulk ESS to 1,684 and 2,071 respectively;
- dense NumPyro NUTS at \(K=50\) raised it to 6,114 with no divergences.

The next experiment therefore composes the existing posterior-invariant
fixed-\(K\) topology transition with a gradient transition over all continuous
coordinates. It is a mixing experiment, not a claim that the structural
problem has been solved.

## Decision

Use native PyMC's one-transition API:

```text
custom topology BlockedStep -> PyMC HamiltonianMC
```

inside a `pm.CompoundStep`. The topology step is always followed by the HMC
step, including after an accepted topology proposal, a valid rejection, or an
invalid structural self-transition. Conditioning whether HMC runs on the
topology outcome would create a different state-dependent schedule without an
invariance argument.

The H2c implementation still uses static HMC:

- fixed step size;
- fixed leapfrog count;
- a frozen topology-neutral total/contrast leaf block;
- an ordered diagonal block for fixed coefficients, with zero cross terms;
- no retained-sampling adaptation;
- no leaf-identity-specific covariance transfer between tilings.

This makes the kernel ordinary Markov-chain composition and keeps exact
restart state small. Step-size and metric adaptation can be performed in a
separate discarded calibration run, then frozen for production.

## Continuous chart

For canonical leaf order on topology \(\tau\), use

\[
x_i = \log m_i,\qquad y_j = \log c_j,
\]

where \(m_i\) is positive leaf mass and \(c_j\) is a positive fixed
coefficient. Define

\[
L=\operatorname{logsumexp}(x),\qquad
T=e^L,\qquad
s_i=e^{x_i-L}.
\]

The scientific full-tiling target is defined in \((T,s,c)\) coordinates.
The required transformation terms are

\[
\log |J_m| = \sum_i x_i-(K-1)L,\qquad
\log |J_c| = \sum_j y_j.
\]

Thus the computational target is

\[
\log\pi(x,y,\tau)
=
\log\pi(T,s,c,\tau)
+\sum_i x_i-(K-1)L+\sum_j y_j.
\]

This symmetric chart avoids choosing a distinguished simplex reference leaf.
Let

\[
P_1=\frac{\mathbf1\mathbf1^\mathsf T}{K},
\qquad
P_\perp=I-P_1.
\]

The H2c leaf position scale is

\[
G_{\rm leaf}
=g_{\rm contrast}P_\perp+g_{\rm total}P_1.
\]

The binary64 implementation evaluates the equivalent stable form

\[
G_{\rm leaf}
=g_{\rm contrast}I+(g_{\rm total}-g_{\rm contrast})P_1,
\]

so equal eigenscales reduce bit for bit to the legacy scalar identity at the
production \(K\) values.

It has eigenvalue \(g_{\rm total}\) along the normalized common direction
\(\mathbf1/\sqrt K\), eigenvalue \(g_{\rm contrast}\) on every zero-sum
contrast, and satisfies \(PG_{\rm leaf}P^\mathsf T=G_{\rm leaf}\) for every
leaf permutation \(P\). It is therefore dense but topology-neutral; no leaf
identity is transferred between tilings. Fixed coefficients retain distinct
diagonal entries because their identities are stable, and leaf/fixed cross
terms remain zero.

With `is_cov=True`, PyMC uses the complete matrix \(G\) as the quadratic
kinetic-energy coefficient:

\[
K(p)=\tfrac12p^\mathsf T Gp,\qquad p\sim N(0,G^{-1}).
\]

Thus the configured position scale is momentum precision, not momentum
covariance.

The PyMC model is compiled once. Topology-dependent design columns and
Dirichlet shapes have fixed array shapes at fixed \(K\) and are supplied
through mutable `pm.Data`. Accepted topology moves update those arrays before
the HMC step; rejected and invalid moves leave them unchanged.

## Structural transition

The H2d structural transition acts directly on the authoritative log-mass
coordinates used by HMC:

- an edge flip transfers the two old child coordinate bit patterns to the two
  new perpendicular children in canonical order;
- a resolution relocation transfers the old destination coordinate to the
  merged parent and the two old merge-child coordinates to the new destination
  children; and
- unchanged leaves retain their coordinate bits.

This is an exact involution with unit Jacobian and no Beta auxiliary draw. The
Metropolis-Hastings ratio uses the exact transformed-coordinate target
difference plus reverse-minus-forward discrete catalogue probability. Every
forward-valid proposal must have a materializable reverse which recovers the
topology and log-coordinate bits exactly; a reverse path may not be discarded
because an intermediate physical fraction rounded to an endpoint.

After every accepted, rejected, or invalid topology attempt:

1. retain the authoritative current log coordinates;
2. atomically install the matching design, Dirichlet arrays, topology
   precision, and leapfrog potential;
3. run exactly one non-adapting PyMC HMC transition;
4. decode the HMC endpoint; and
5. fully rebuild `FullTilingPosteriorState` as an independent cache and target
   oracle.

## Reproducibility and checkpointing

The baseline uses one authoritative NumPy PCG64 stream. Each compound sweep
uses it for the structural proposal and decision, then draws a seed used to
reset all randomness in the PyMC HMC step. Because HMC adaptation is disabled
and the metric is static, exact continuation requires:

- current scientific state and topology;
- exact PCG64 state;
- compound sweeps completed;
- fixed \(K\);
- requested HMC step size and leapfrog count; the one-ULP-adjusted effective
  value is a trace diagnostic rather than continuation state;
- complete PyMC position-scale specification;
- schedule, target, precision, and backend-version identity.

Boundary-inclusive trace arrays retain the authoritative PyMC \(x\) and \(y\)
coordinates directly; they are not reconstructed as `log(exp(x))`. Durable
resume additionally requires the selected checkpoint to belong to a complete
parent segment whose `complete.json` certifies the current hashes of its
manifest, trace, summary, and checkpoint.

A fresh initializer is a different boundary from a durable continuation. Its
scientific state is constructed in positive physical coordinates, so before
the first retained boundary it is mapped through the HMC log chart and fully
rebuilt by the scientific state oracle until

```text
exp(log_leaf_mass) == leaf_mass
exp(log_fixed_coefficient) == fixed_coefficient
```

bit for bit. This deterministic canonicalization changes only the binary64
representation of the fresh starting point; it is not an MCMC transition and
does not weaken the scientific target or audit tolerance. It is idempotent
and does not mutate the caller's state. Durable continuations bypass it
because their physical state and authoritative log coordinates are already
joint replay inputs.

Production sampling requires the actual hashed strict-JSON calibration file,
not only a caller-supplied identifier. Its v2 schema binds the frozen input,
target controls, \(K\), coordinate/metric identities, resolved static kernel,
bounded development search, role-specific topology and master-PCG64 seeds,
untouched held-out validation, source artifact hashes, and code revision.
Development candidates use common random numbers within each topology;
validation uses separate streams, and calibration topology seeds are disjoint
from retained-production starts. The fixed-basis NUTS metric-source topology
and all mobile calibration topologies are recorded by canonical hash; exact
collisions are rejected independently of seed identity. Per-sweep
post-structure/pre-HMC coordinates make the candidate score HMC-only rather
than a mixture of structural and continuous displacement. Version-1 diagonal
calibrations and checkpoints fail closed; they remain evidence for the earlier
experiment, not continuation inputs for H2c. The driver reports
transformed-target preflight compilation, production-kernel setup/compilation,
and transition execution separately so sampling throughput excludes
compilation.

The HMC object's iteration counter and internal RNG do not define the next
scientific transition because its RNG is reset from the checkpointed PCG64
stream before every HMC call. Any recorded diagnostic counter must still be
derived from the global sweep coordinate so direct and resumed diagnostics
match.

## Required local checks

- PyMC transformed target against
  `FullTilingPosteriorState.log_target + log|J_m| + log|J_c|`.
- Independent closed-form transformed density.
- Gradient finite-difference checks on small problems.
- Topology acceptance/rejection remains the existing scientific transition.
- Accepted, rejected, and invalid topology attempts each invoke HMC once.
- HMC never changes the tiling.
- Full posterior rebuild agrees with every accepted HMC endpoint.
- Frozen step size, total/contrast position-scale matrix, and leapfrog count.
- Exact total/contrast eigenvalues, block ordering, positive definiteness, and
  invariance under arbitrary leaf permutations.
- Per-sweep post-structure/pre-HMC log coordinates isolate HMC displacement
  from structural remapping.
- Exact seeded replay.
- Exact awkward-boundary sample/continue replay.
- Exact, idempotent fresh-boundary log/exp canonicalization without mutating
  the supplied initializer.
- Fail-closed problem, \(K\), precision, settings, and checkpoint checks.

## Local validation completed

On 2026-07-25:

- the integrated outer pytest invocation passed; PyMC-dependent cases marked
  as skipped in the parent are executed and required to pass inside isolated
  float64 child processes;
- Ruff check and format-check passed on all new/changed implementation and
  test files;
- Pyright reported zero errors on the core, I/O, and native driver;
- exact authoritative-coordinate replay, arbitrary-boundary continuation,
  strict no-pickle checkpoint reload, calibration mismatch rejection,
  certified-parent rejection, artifact failure injection, and trace reopen
  audits are covered;
- a regression with deliberately non-roundtripping fresh leaf and fixed
  values verifies exact draw-0 coordinates, oracle reconstruction, manifest
  lineage, and checkpoint state;
- the full repository tox matrix was intentionally not run; the agreed gate
  for this experimental track is the focused experimental suite.

## Real-data question

The first real-data screen should answer a narrow question:

> Does replacing local root/pair/fixed updates by a joint gradient transition
> remove the persistent likelihood start separation while topology remains
> mobile?

It should not initially be used to compare posterior summaries. Four
overdispersed topology starts at each of \(K=50\) and \(K=250\) are needed,
with the same frozen PARIS input and scientific target as the earlier
fixed-basis control. The initial total/contrast metric may be estimated only
from checksum-verified fixed-basis NUTS draws. Step size and path length are
then calibrated on separate discarded mobile compound runs, including an
untouched held-out topology, and frozen identically across retained mobile
chains at a given \(K\).

## Deferred work

- NUTS in the mobile compound kernel.
- Online metric adaptation.
- Topology-conditioned Euclidean HMC is now the immediate next phase rather
  than deferred work.
- Leaf-identity-specific metrics or leaf/fixed cross blocks.
- Position-dependent Riemannian HMC.
- A source-HMC/topology/destination-HMC proposal with one joint acceptance
  decision.
- Variable-\(K\) topology transitions.
- A scientifically different multi-root prior.
- Promotion out of the experimental namespace.
