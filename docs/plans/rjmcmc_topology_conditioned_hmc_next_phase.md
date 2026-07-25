# Topology-conditioned HMC next phase

## Status and decision

H2c is a certified hard stop for the topology-neutral total/contrast metric at
both tested fixed dimensions:

- at \(K=50\), \(\epsilon=0.05\) gave mean acceptance 0.919 on one
  development topology and 0.418 on the other;
- at \(K=250\), \(\epsilon=0.05\) gave 0.846 and 0.923, while
  \(\epsilon=0.1\) produced 173--268 divergences;
- no candidate passed the frozen development gates, so no held-out or
  production stage was permitted.

The evidence is retained under
`/group/chem/acrg/brendan_for_codex/rjmcmc_full_tiling_pymc_hmc/7a1a1cc673a4b6a6ce0ed7b5123494ebd205b467`.
The readable report is `report/H2C_RESULTS.md` and the hard-stop certificate is
`calibration/H2C_HARD_STOP.json`.

The next experiment is **not** automatically Riemannian HMC. It is a
topology-conditioned Euclidean HMC kernel whose metric:

1. is a deterministic function of the current tiling, frozen input, target
   controls, and fixed reference coordinates;
2. remains constant throughout each HMC trajectory;
3. changes only when the discrete topology changes; and
4. is permutation-equivariant, rather than identical under every leaf
   permutation.

For each topology \(\tau\), this defines an ordinary invariant HMC kernel
\(K_\tau\). Composing it with the existing invariant topology kernel therefore
preserves the joint fixed-\(K\) posterior. No momentum crosses the topology
transition and no Riemannian integrator is required.

The step size and fixed or randomized leapfrog-count rule may also be
deterministic functions of \(\tau\). They need not be identical for every
tiling. They must be defined from frozen topology/model information, remain
fixed during the corresponding HMC transition, and be covered by the
held-out-topology validation. Choosing them from the current continuous
position or recent acceptance history during retained sampling is a different
adaptive algorithm.

## What H2c established

H2c deliberately required

\[
P G_{\rm leaf}P^\mathsf T=G_{\rm leaf}
\]

for every leaf permutation \(P\). This restricts the leaf block to the form

\[
G_{\rm leaf}=aI+b\mathbf1\mathbf1^\mathsf T,
\]

so it can distinguish only the common total direction from one homogeneous
contrast scale. It cannot represent:

- different likelihood sensitivity for different spatial regions;
- correlations among particular leaf coefficients;
- correlations between leaf and fixed InTEM coefficients; or
- changes in those relationships when the tiling changes.

The H2c hard stop is therefore evidence against that restricted metric family,
not evidence that one must immediately use a metric that changes with the
continuous state.

The H2c A/B calibration roles were mobile trajectories initialized from
different random-recursive tilings. One topology transition preceded every HMC
transition. The sharp role differences therefore establish
topology/trajectory-conditioned stiffness, but they do not by themselves
separate the current tiling from the current continuous shares. That
separation is the first task below.

The replacement requirement is equivariance. If a relabeling of a topology
also permutes its design columns and reference shares, the resolved metric
must satisfy

\[
G_{P\tau}=P_*G_\tau P_*^\mathsf T,
\]

where \(P_*\) applies \(P\) to the leaf block and leaves the ordered fixed
coefficients unchanged. This prevents arbitrary label effects while allowing
region-specific scales, leaf correlations, and leaf/fixed cross terms.

## H2d metric

### Deterministic reference state

For each topology, construct a reference state from quantities fixed before
retained sampling:

- root total \(T_\star=1\), the declared Gamma-prior mean;
- leaf shares \(s_{\tau,\star}\) equal to the normalized nominal mass in each
  canonical leaf;
- fixed coefficients \(c_\star=\mathbf1\); and
- the exact topology-specific dynamic design matrix \(A_\tau\).

This reference depends on the topology and frozen model, but not on the
current continuous chain state or its history.

### Reference curvature

Let \(x=\log m\) and \(y=\log c\), and concatenate \(z=(x,y)\). At the
reference state the Gaussian-likelihood Jacobian is

\[
J_{\tau,\star}
=
\left[
A_\tau\operatorname{diag}(m_{\tau,\star}),
\quad
B\operatorname{diag}(c_\star)
\right],
\]

where \(B\) is the ordered fixed-coefficient design and
\(m_{\tau,\star}=T_\star s_{\tau,\star}\). With fixed diagonal observation
covariance \(R\), use

\[
Q_{\tau,\star}
=
Q_{\tau,\star}^{\rm prior}
+
J_{\tau,\star}^{\mathsf T}R^{-1}J_{\tau,\star}
\]

as a topology-specific reference precision.

The root/share prior has a convenient exact log-chart curvature. For
\(T\sim\operatorname{Gamma}(a,b)\) and
\(s\sim\operatorname{Dirichlet}(\alpha)\), with
\(\kappa=\sum_i\alpha_i\), the negative Hessian of the transformed log prior
at \((T,s)\) is

\[
Q_x^{\rm prior}
=
(\kappa-a+bT)
\{\operatorname{diag}(s)-ss^\mathsf T\}
+
bTss^\mathsf T.
\]

For the current \(a=b=4\), \(T_\star=1\), and \(\kappa=2K\), this reduces to

\[
Q_x^{\rm prior}
=
\kappa
\{\operatorname{diag}(s_{\tau,\star})
-s_{\tau,\star}s_{\tau,\star}^{\mathsf T}\}
+
4s_{\tau,\star}s_{\tau,\star}^{\mathsf T}.
\]

The fixed lognormal coefficients are Gaussian in \(y\), so their prior block
is the exact diagonal log-precision. The prior leaf/fixed cross block is zero;
the likelihood term supplies the scientifically relevant cross block.

After a declared, deterministic stabilization and conditioning audit, the
desired kinetic coefficient is

\[
G_\tau=Q_{\tau,\star}^{-1}.
\]

Do not form this inverse merely to configure PyMC. Prefer to pass
\(Q_{\tau,\star}\) directly with `is_cov=False`; PyMC's dense inverse
potential then evaluates
\(K(p)=\tfrac12p^\mathsf TQ_{\tau,\star}^{-1}p\) and draws
\(p\sim N(0,Q_{\tau,\star})\). This is equivalent to passing
\(G_\tau\) with `is_cov=True`, but avoids explicit inverse-symmetry and
conditioning errors.

The implementation must verify this convention directly, as in H2c. It must
not infer the convention from the ambiguous phrase “mass matrix.”

### Correctness boundary

The reference metric may use the observations, design matrices, priors, and
topology because all are fixed inputs to the transition kernel. It must not be
recomputed from the current \(x,y\) and then frozen only for the forward
trajectory. The reverse transition would generally choose another metric,
invalidating the ordinary Euclidean-HMC acceptance rule.

A metric that genuinely varies along the continuous trajectory is deferred to
RMHMC, including its position-dependent kinetic normalization, generalized
integrator, and reversibility checks.

## Implementation sequence

### Phase 0: topology/position curvature audit

Before changing the sampler, use the recorded H2c seeds and first-sweep
coordinates to perform eight deterministic evaluations:

1. reconstruct the exact post-structure topology for both H2c roles at each
   \(K\);
2. evaluate the exact Hessian and the SPD
   Gauss--Newton-plus-prior precision for both topologies at the same declared
   unit-total/unit-fixed nominal reference;
3. repeat at each role's recorded first HMC start; and
4. report generalized eigenvalue ranges under the H2c metric, especially the
   largest preconditioned curvature, condition number, any negative exact
   Hessian eigenvalues, and leaf/fixed/cross-block norms.

This requires no sampling. The H2c traces did not retain mobile rectangle
bounds in their calibration aggregate, so the states must be reproduced from
the frozen topology and master-PCG64 seeds and checked against the recorded
first-sweep coordinates and HMC seed.

If most of the scale change follows topology at the common reference, proceed
with the topology-conditioned metric below. If the within-topology
reference-to-recorded-position change is comparable or larger, retain the
static metric as a bounded baseline but bring the continuous-chart/RMHMC
decision forward rather than assuming H2d will generalize.

#### Phase 0 result

The read-only BP1 audit completed against commit `7a1a1cc` and the
checksum-verified H2c bundle. It reproduced all four first post-structure
states and HMC seeds exactly.

The result supports the topology-conditioned static baseline:

- the reference-topology effect changed the RMS preconditioned spectrum by
  factors 1.88 at \(K=50\) and 1.71 at \(K=250\);
- changing from the nominal reference to the recorded first HMC position
  changed that spectrum by at most a factor 1.07 at \(K=50\) and 1.01 at
  \(K=250\);
- the topology effect was about 9.6 and 60 times the largest corresponding
  position effect on the audit's log scale;
- the H2c-preconditioned Gauss--Newton-plus-prior condition numbers remained
  about 8,700--15,400 at \(K=50\) and 25,900--47,100 at \(K=250\), roughly
  1--5% worse than the unpreconditioned reference; and
- the exact transformed-target Hessian was indefinite in every audited state,
  with minimum eigenvalues from about \(-7.4\) to \(-20.8\).

Use the SPD Gauss--Newton-plus-exact-prior precision for H2d. Retain the exact
Hessian only as a diagnostic; do not pass it to the kinetic potential.

### Phase A: offline metric oracle

Implement the reference-curvature builder independently of PyMC first.

Required checks:

- exact prior-curvature formula against automatic or finite-difference
  Hessians on small cases;
- likelihood Gauss--Newton term against direct Jacobian products;
- symmetry, strict positive definiteness, condition number, and stable
  factorization, with a deterministic fail-closed or stabilization policy;
- algebraic permutation equivariance and a strict floating-point comparison
  under random leaf permutations;
- fixed-coefficient ordering and leaf/fixed cross-block placement;
- deterministic reconstruction and hashing;
- agreement between a full rebuild and any incremental update path; and
- memory and timing at \(K=50\) and \(K=250\) on BP1.

The first scientific oracle should be the full dense \((K+6)\)-dimensional
precision/metric. At \(K=250\), \(d=256\), so one dense float64 matrix uses
about 0.5 MiB. Its storage is small relative to the existing run memory. A
full \(J^\mathsf TJ\) rebuild over 1,382 observations costs roughly 91 million
multiply-accumulates; that is a more likely bottleneck than the
\(256\times256\) Cholesky factorization.
If repeated full construction is too costly, optimize without changing the
resolved metric:

1. update only Gram-matrix rows and columns for design columns changed by the
   local topology move;
2. update the prior block from the changed nominal shares using diagonal and
   low-rank terms; and
3. refactor the resulting dense matrix.

A diagonal or low-rank approximation is a separate metric candidate and must
receive a new identity and validation protocol.

### Phase B: PyMC dynamic-potential integration

Allow the structural step to install the deterministic \(Q_{\tau,\star}\)
before the following HMC transition. The target graph remains compiled once.
PyMC's leapfrog integrator retains a reference to its kinetic potential, so
changing only `hmc.potential` is insufficient. Use one deliberately mutable
potential shared with the integrator, or replace both the potential and
integrator atomically.

Required checks:

- the resolved metric belongs to the same topology as the installed design
  and Dirichlet arrays;
- accepted, rejected, and invalid structural outcomes still invoke exactly
  one HMC transition;
- the HMC trajectory holds topology and \(G_\tau\) fixed;
- the HMC step remains non-adapting;
- direct and awkward-boundary restart replay are exact;
- checkpoints bind the metric-builder identity, reference-state identity,
  resolved metric hash, and numerical-runtime identity; and
- corrupt or mismatched topology/metric pairs fail closed.

### Phase C: H2d bounded calibration

All H2c initial topologies, mobile trajectories, and outcomes have now been
observed and belong to the development record. H2d must use newly predeclared
held-out initial-topology and master-stream seeds.

For each \(K\):

1. use at least three development trajectories spanning the earlier
   largest-nominal and random-recursive initial geometries;
2. freeze two new held-out mobile trajectories before inspecting H2d results;
3. use one common, predeclared dimensionless step/path grid across all
   development topologies;
4. permit one deterministic topology-only step-size and/or path-length rule
   only if it is declared from metric diagnostics before calibration;
5. require finite states, zero divergences, and acceptable mean acceptance on
   every development topology;
6. lock the selected controls and their artifact hash; and
7. require both held-out topologies to pass without retuning.

The grid should be finer than H2c between 0.025 and 0.1 because the previous
grid deliberately did not resolve that interval. This is a new protocol, not
post-hoc continuation of H2c. A state-independent randomized distribution over
short path lengths may be included to reduce resonances, but its probabilities
must be frozen and replayed exactly.

H2d is a hard stop if no setting serves every held-out topology. Do not average
away a topology-specific failure.

### Phase D: durability, performance, and real-data screen

Only after H2d passes:

1. rerun target-identity and frozen-input dry-run gates;
2. prove exact arbitrary-phase restart and failure-injection behavior;
3. compare full and incremental metric construction;
4. report metric-build, factorization, HMC-gradient, topology, checkpoint, and
   diagnostics time separately; and
5. run four overdispersed mobile-topology chains at each \(K=50\) and
   \(K=250\).

The real-data question remains:

> Does a joint gradient update with topology-conditioned Euclidean
> preconditioning remove persistent likelihood start separation while the
> tiling remains mobile?

Compare against:

- the fixed-basis NUTS reference;
- the deterministic fixed-basis local-sampler control;
- the earlier mobile local sampler; and
- the H2c calibration failures.

Do not interpret spatial posterior summaries unless the predeclared
multi-chain likelihood and common-projection gates pass.

## Later structural proposal: RJ--HMC

Wyse, Friel, and Girolami's *Reversible jump Riemann Manifold Hamiltonian
Monte Carlo* is relevant, but it belongs after a robust topology-conditioned
within-tiling metric exists.

Their between-model construction applies a Hamiltonian path in the source
model, performs the reversible-jump map at an intermediate state, applies a
Hamiltonian path in the destination model, and makes one acceptance decision
for the complete path. It permits different metrics, step sizes, and path
lengths in different models. The paper also states that the construction can
use ordinary HMC with constant metrics; RMHMC is not essential to the
reversible-jump composition.

The paper does not solve tuning. Its Section 5 reports strong, unintuitive
sensitivity to step size and leapfrog count and concludes that these controls
still require tuning. It therefore does not bypass H2c/H2d metric
generalization.

For the full-tiling model, a later bounded prototype should compare:

1. the current independently accepted topology-then-HMC composition; and
2. a source-HMC/topology-map/destination-HMC path with one exact acceptance
   decision.

Although the initial full-tiling experiments keep \(K\) fixed, the state-vector
meaning changes with the tiling and the existing structural map and Jacobian
accounting remain authoritative. The joint proposal must retain:

- the exact forward and reverse path order;
- source and destination momentum densities and normalizers;
- topology proposal probabilities;
- the existing structural Jacobian;
- any randomized direction or path-length probabilities; and
- full round-trip and finite-state invariance tests.

This proposal is intended to repair poor landing points and
topology--continuous coupling. It should not be attempted merely because a
within-topology HMC metric is poorly scaled.

## Trigger for RMHMC

Consider a position-dependent metric only if H2d shows both:

1. the topology-conditioned reference metric is well conditioned and
   generalizes across topology labels; and
2. energy error or acceptance still changes sharply across continuous starts
   within the same frozen topology.

That pattern would be evidence that curvature varies materially with
\((x,y)\), rather than only with \(\tau\). At that point compare a full
position-dependent method with a better continuous chart or prior
standardization before coupling it to the topology proposal.

## Final model-level branch: learned aggregation error

If the partition is intended to be only a computational representation of one
common native flux model, exact marginalization changes the interpretation of
the structural problem. For every partition \(P\), define the pushforward
state \(A_P=T_P(X)\) from one proper native prior and the exact reduced
observation kernel

\[
L_P(y\mid a)
=
\int p(y\mid x)\,
p(dx\mid T_P(X)=a).
\]

Then

\[
\int L_P(y\mid a)\,p_P(a)\,da=p(y)
\]

is the same for every exact projection. Consequently the data cannot update
\(P\) or \(K\): their posterior probabilities equal their declared structural
prior probabilities. Common projected posterior quantities remain
partition-consistent. This is the correct limiting behavior, not a mixing
failure.

An NLE can approximate the non-Gaussian reduced kernel when the hidden
positive allocation cannot be integrated cheaply. The relevant learned object
is a **normalized conditional likelihood or aggregation-residual density**,
for example

\[
q_\phi(E_{\rm agg}\mid A_P,P,H,c),
\qquad
E_{\rm agg}=H\{X-g_P(A_P)\},
\]

trained from joint native simulations. It is not a neural posterior or an
unnormalized likelihood-to-evidence ratio. Keep independent Gaussian
measurement noise outside the learner where practical.

The current Gamma--Beta reduced likelihood does not yet have this property.
For an inactive subtree, its coarse design and rendered native field fill
descendants at their nominal proportions. The hidden Beta fractions are not
integrated out. Candidate frontiers are therefore partition-specific reduced
models, and unequal evidence is expected when footprint columns differ.

This branch has a strict interpretation gate:

- if the same root model and exact/adequate reduced likelihood are used, any
  learned evidence preference among partitions is surrogate error;
- if the scientific intention is for data to inform \(K\) or \(P\), then some
  partition-indexed prior, forward approximation, or discrepancy law must be
  declared as a genuine model difference; and
- a complexity or computational penalty can guide a basis decision, but it is
  not observational evidence among exact representations.

The bounded programme is:

1. reproduce evidence invariance in a linear-Gaussian two-cell/multiscale
   oracle with exact aggregation covariance;
2. use a positive Gamma--Beta two-cell model with endpoint-aware
   Gauss--Jacobi quadrature for the hidden share and generalized
   Gauss--Laguerre quadrature for the Gamma total, comparing exact
   marginalization, explicit hidden-share inference, Gaussian moment closure,
   and nominal filling;
3. enumerate the five frontiers of a \(2\times2\) canonical tree and require
   common prior-predictive evidence, recovery of a nonuniform structural
   prior, projective tower consistency, likelihood normalization, common-total
   posterior agreement, and a deliberate nominal-fill failure sentinel;
4. attempt a conditional flow only if the normalized mixture fails a
   predeclared density/calibration gate; and
5. assess the 1,382-observation PARIS problem only after a low-rank or
   factor-analytic residual representation has an auditable normalized joint
   density and held-out partition/operator tests.

In the exact-representation limit there is little inferential reason to spend
computation mixing over \(K\) and partitions. A greedy/SLS basis can instead
be treated as an algorithmic approximation or reporting decision, while exact
or learned aggregation error propagates the unresolved uncertainty. RJMCMC
would remain useful only if averaging over the structural prior is desired or
the partitions deliberately define different scientific models.

## BP1 coordination

Use bounded workers before submitting another large matrix:

1. a read-only artifact worker to summarize H2c acceptance, energy error, and
   divergence behavior by topology and continuous start;
2. a metric-oracle worker to implement and test Phase A on a dedicated branch;
3. an independent mathematical reviewer for the prior Hessian, PyMC metric
   convention, and invariant-kernel composition;
4. an integration worker for the dynamic PyMC potential and checkpoint
   contract; and
5. an HPC-plan reviewer before H2d submission.

The existing BP1 `hpc-ci-project-tracker` configuration exposes an
`openghg_inversions` workspace, but it belongs to a different project queue.
Do not add RJMCMC tasks to that queue implicitly. Either add a deliberately
scoped RJMCMC project/configuration or use free-form, read-only launch
artifacts with no queue mutation.

Any submitted Slurm work must use the existing BP1 monitoring workflow:
record job IDs and immutable run roots, watch scheduler state and logs, retain
hard-stop artifacts, and wait for terminal completion before interpreting
partial matrices.

## Promotion criteria

Proceed from topology-conditioned HMC to the joint RJ--HMC proposal only when:

- Phase 0 evidence is recorded and Phase A and B correctness/replay tests
  pass;
- H2d passes all newly held-out topologies at both \(K\) values;
- the real-data screen shows that within-tiling continuous mixing is no
  longer the dominant failure; and
- the remaining separation is localized to topology changes or
  topology--continuous landing.

Proceed to RMHMC only under the within-topology continuous-curvature trigger
above.

## Background references

- Wyse, Friel, and Girolami (2011), *Reversible jump Riemann Manifold
  Hamiltonian Monte Carlo*:
  <https://www.scss.tcd.ie/jason.wyse/Files/Papers/WyseFrielGirolami.pdf>.
- Curated project note:
  `~/Documents/inversions-knowledge/docs/research-questions/rjmcmc-hmc-nuts-and-transported-tuning.md`.
  In particular, see its sections on invariant compound kernels, frozen
  Euclidean metrics, transported tuning, and the separate
  source-HMC/RJ/destination-HMC construction.
- Curated learned-marginal note:
  `~/Documents/inversions-knowledge/docs/research-questions/learning-non-gaussian-marginal-models.md`.
- Curated projection/evidence note:
  `~/Documents/inversions-knowledge/docs/research-questions/posterior-projection-conundrum.md`.
