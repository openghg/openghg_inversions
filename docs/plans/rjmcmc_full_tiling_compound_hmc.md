# Full-tiling topology plus HMC

## Status

The core sampler, dedicated durable checkpoint schema, native real-data
driver, and HPC validation plan are implemented on
`codex/rjmcmc-compound-hmc`. Focused local validation is complete; the next
gate is the frozen-input HPC calibration and real-data screen.

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

The first implementation uses static HMC:

- fixed step size;
- fixed leapfrog count;
- frozen topology-neutral diagonal PyMC position scale;
- no retained-sampling adaptation;
- no transfer of a dense leaf metric between tilings.

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
A scalar leaf block in the PyMC position scale is invariant to permutations
of canonical leaf positions. With `is_cov=True`, this diagonal is the
quadratic kinetic-energy coefficient and hence momentum precision; sampled
momentum has its reciprocal covariance. Fixed coefficients may use distinct
diagonal entries because their identities do not change with topology.

The PyMC model is compiled once. Topology-dependent design columns and
Dirichlet shapes have fixed array shapes at fixed \(K\) and are supplied
through mutable `pm.Data`. Accepted topology moves update those arrays before
the HMC step; rejected and invalid moves leave them unchanged.

## Structural transition

The existing edge-flip/resolution-relocation implementation remains the
authoritative topology kernel. It is evaluated and accepted in the existing
scientific mass / root-share coordinates, including its proven proposal and
Jacobian accounting. The transformed PyMC target must not be substituted into
that Metropolis-Hastings calculation because that would omit or double-count
coordinate Jacobians.

After the topology decision:

1. encode the accepted/current scientific masses as log masses;
2. atomically install the corresponding design matrix and Dirichlet shapes;
3. run exactly one PyMC HMC transition;
4. decode the HMC endpoint;
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

Production sampling requires the actual hashed strict-JSON calibration file,
not only a caller-supplied identifier. Its v1 schema binds the frozen input,
target controls, \(K\), coordinate/metric identities, resolved static kernel,
bounded pilot design, decision statistics, source artifact hashes, and code
revision. The driver reports transformed-target preflight compilation,
production-kernel setup/compilation, and transition execution separately so
sampling throughput excludes compilation.

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
- Frozen step size, position-scale diagonal, and leapfrog count.
- Exact seeded replay.
- Exact awkward-boundary sample/continue replay.
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
fixed-basis control. Static HMC controls should be calibrated only on
discarded fixed-topology runs, then frozen identically across the mobile
chains at a given \(K\).

## Deferred work

- NUTS in the mobile compound kernel.
- Online or topology-dependent metric adaptation.
- Dense leaf metrics or leaf/fixed cross blocks.
- Variable-\(K\) topology transitions.
- A scientifically different multi-root prior.
- Promotion out of the experimental namespace.
