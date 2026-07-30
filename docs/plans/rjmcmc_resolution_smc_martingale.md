# Resolution-SMC for Gamma--Dirichlet aggregation error

## Status and scope

This note began as a prospective design after the calibrated G4
projected-bank failure.  The tiny implementation and first BP1 experiment now
exist, but the PARIS-scale method remains experimental.  The note does not
authorize use of approximate likelihood differences as evidence for a
partition or \(K\).

The durable mathematics is recorded separately in
`inversions-knowledge`:

- `docs/derivations/telescoping-gamma-beta-aggregation-messages.md`;
- `docs/derivations/non-gaussian-aggregation-error-by-marginalization.md`;
  and
- `docs/topics/positive-multiscale-priors-and-gamma-beta-trees.md`.

The first file appears in the rendered knowledge base under
**Browse → Derivations → Aggregation and basis geometry → Telescoping
Gamma--Beta aggregation residuals**.

The experiment proposed here is an alternative evaluator for the same
single-root conditional marginal likelihood.  It is not an RJMCMC kernel and
does not infer a tree.  The tree is a computational chart used to reveal one
fixed native Gamma--Dirichlet allocation progressively.

## Implementation status

The implementation branch is
`codex/rjmcmc-resolution-smc-experiment`.  Its latest verified remote commit
on 2026-07-30 was
`abdce3c30c65aebd88c3c4f27c588c71aaabe2c2`; it contains the tiny
resolution-SMC implementation, the wider R1 matrix and a bounded guided R2
experiment.

R0 and R1a have completed.  R1a established target, normalization,
conservation, replay and chart-equivalence correctness.  Direct IID and
no-resampling SMC were exactly pathwise identical.  Bootstrap resampling
reduced raw variance for the boundary-heavy target, but its guide and
frontier overhead removed the gain: the best
relative-variance-times-median-cost ratio was `1.66` relative to IID, where a
value below one would favour SMC.  The minimum intermediate ESS fraction was
`0.0668`.  The complete report is
`docs/plans/rjmcmc_resolution_smc_r1a_report.md` on the implementation branch.

The wider R1 Slurm array has now completed all 36 tasks with zero scheduler
failures.  All 36 task certificates and their replicate/level records are
present.  The three guided R2 tasks at 256, 1,024 and 4,096 particles also
completed and published passing task certificates, including exact
boundary-restart replay.  No related Slurm job remains active.

This authenticates completed computation, not the scientific performance
claim.  No merged R1/R2 analysis or final cost/variance report had been
published at the last inspection.  Do not infer success from per-task
certificates: they certify execution and local invariants, whereas promotion
depends on the aggregate oracle agreement and variance-times-cost comparison.
PARIS-scale R4 is not yet justified.

The implementation branch was created before the later literature-map update
on this planning branch.  Reconcile that documentation before merging the
experiment branch; no numerical result depends on the missing prose.

## Current design model

For one retained physical mass \(T_0\), let native cell masses have the common
Gamma--Dirichlet representation

\[
X=T_0W,\qquad
W\sim\operatorname{Dirichlet}(\alpha).
\]

The observation model is

\[
y=b+HX+\epsilon,\qquad
\epsilon\sim\mathcal N(0,D).
\]

Choose a fixed binary tree whose leaves are the native cells and whose Beta
parameters are descendant sums of the same \(\alpha_i\).  The tree is then
only a coordinate factorization of the native Dirichlet law.

Let \(\mathcal F_\ell\) contain the parent masses revealed through resolution
level \(\ell\), and put

\[
\bar X_\ell=\mathbb E[X\mid\mathcal F_\ell].
\]

The unresolved observation residual has exact conditional covariance

\[
C_{\ell}
=
\operatorname{Cov}\{H(X-\bar X_\ell)\mid\mathcal F_\ell\},
\]

obtainable from the unresolved Gamma--Beta martingale increments.  The exact
conditional likelihood message is

\[
L_\ell
=
\mathbb E[
\mathcal N(y;b+HX,D)
\mid\mathcal F_\ell
].
\]

\((L_\ell,\mathcal F_\ell)\) is a positive martingale, but \(L_\ell\) is
generally unavailable.  A moment-closure guide is

\[
\widetilde L_\ell
=
\mathcal N(y;b+H\bar X_\ell,D+C_\ell).
\]

Only \(L_\ell\) is exact.  \(\widetilde L_\ell\) matches conditional first and
second moments but is generally not a martingale and can miss
boundary-heavy/non-Gaussian allocation shape.

## Facts versus experimental inferences

### Established facts

- Nested conditional means telescope exactly.
- Parent-first Gamma--Beta increments are conditionally centred and mutually
  uncorrelated.
- Their conditional covariance contributions add exactly.
- Exact conditional likelihood messages satisfy
  \(L_\ell=\mathbb E[L_{\ell+1}\mid\mathcal F_\ell]\).
- Every compatible binary tree gives the same continuous Dirichlet target.
- The final native likelihood is normalized and directly evaluable once a
  complete allocation has been drawn.
- Standard SMC can use any positive sequence of intermediate targets if its
  proposal and incremental-weight ratios are accounted correctly.

### Working inferences to test

- Gaussian tail closure may be accurate enough to guide particles even where
  it is not an adequate final likelihood approximation.
- Revealing allocation coordinates gradually may reduce the rare-component
  failure seen in the equal-weight G4 bank.
- Data-informed resampling may deliver a lower-variance non-negative
  likelihood estimator per unit work than IID prior simulation.
- Observation-sensitive split ordering may outperform purely breadth-first
  spatial ordering.
- A locally guided one-dimensional Beta proposal may be needed if prior
  proposals collapse.

### Not implied

- Martingale covariance additivity does not prove that the unresolved tail is
  Gaussian.
- Logarithmic tree depth does not imply low integration dimension.
- A stable SMC estimate at one tree ordering does not prove chart invariance.
- A finite-particle likelihood difference is not structural evidence for
  \(P\) or \(K\).
- An unbiased likelihood estimate does not automatically make HMC practical.

## Proposed resolution-SMC sequence

Let \(\rho_{1:\ell}\) denote all Beta coordinates revealed through level
\(\ell\).  Define the prospective intermediate unnormalized target

\[
\gamma_\ell(\rho_{1:\ell})
=
p(\rho_{1:\ell}\mid T_0)\,
\widetilde L_\ell(\rho_{1:\ell}).
\]

At the native terminal level \(L\),

\[
C_L=0,\qquad
\widetilde L_L
=
\mathcal N(y;b+HX,D),
\]

so \(\gamma_L\) is the exact allocation posterior numerator.  The
intermediate closures affect variance, not the terminal target, provided the
SMC weights are correct.

### Bootstrap proposal

First propose each newly revealed Beta split from its prior.  When the prefix
prior cancels, the incremental log weight is

\[
\log w_\ell
=
\log\widetilde L_\ell
-
\log\widetilde L_{\ell-1}.
\]

Use log-domain normalization and resample only when the current ESS falls
below a recorded fraction of the particle count.

### Locally guided proposal

If bootstrap weights collapse, the next bounded variant may approximate

\[
q(\rho_v\mid\mathcal F_\ell,y)
\propto
\operatorname{Beta}_v(\rho_v)
\widetilde L_{\ell+1}(\rho_v).
\]

This is one dimensional for one eligible node.  Gauss--Jacobi evaluations can
help construct and normalize a continuous proposal, but a discrete
quadrature catalogue must not silently replace the continuous terminal
target.  The actual proposal density and its normalizer must be evaluable for
the SMC correction.

### Normalizing constant

For standard resampling conventions, the SMC normalizing-constant estimate
has the form

\[
\widehat Z_L
=
\widetilde L_0
\prod_{\ell=1}^{L}
\left\{
\frac{1}{N}
\sum_{n=1}^{N}
w_\ell^{(n)}
\right\}.
\]

Unbiasedness is assessed on the linear likelihood scale.  The logarithm is
biased and is retained only as a stable reporting coordinate.

## Relationship to other estimators

### Learned conditional likelihoods

The corrected frozen score-regularized NLE candidate failed its public
two-/four-cell promotion screen, especially for boundary-heavy likelihood
shape and retained-mass gradients.  This is a candidate-specific result, not
a reason to remove learned comparators.  It does mean that the current frozen
score flow should appear only as a documented negative control.

Before PARIS-scale SMC, use real-operator geometry in two observation-blind
bridges:

1. 4--16-cell PARIS-derived synthetic subproblems with exact or independently
   converged likelihood oracles; and
2. full-PARIS-operator model-generated observations assessed by
   simulation-based calibration and posterior/predictive coverage.

The first should compare direct IID, Gaussian closure, resolution-SMC,
NLL-only flow and, if useful, a small conditional diffusion experiment.  The
second should include only methods that survive the first.  Detailed NLE
failure evidence and experiment definitions are in
[`rjmcmc_score_regularized_nle_review.md`](rjmcmc_score_regularized_nle_review.md).

Score-based diffusion remains scientifically possible, but it is less direct
as a normalized likelihood evaluator: denoising score matching gives a
data-space score, while likelihood values require probability-flow
integration and retained-mass gradients require an additional reliable
derivative.  It is more attractive if the downstream method can consume
samples, posterior estimates or likelihood ratios rather than an explicit
PyMC/HMC likelihood.

### IID and RQMC source banks

IID and RQMC draw complete prior allocations before seeing their likelihood.
Resolution-SMC instead allows intermediate likelihood guides to select which
coarse prefixes receive further computation.  Exact terminal particles still
require all native splits, so the gain would be variance reduction per
complete path rather than elimination of \(O(N_{\rm native})\) work.

The calibrated G4 result does not establish that SMC will work.  It shows that
the posterior-relevant allocation set can be extremely sparse under direct
prior simulation.  A poor coarse guide can reproduce the same failure by
discarding useful prefixes early.

### Multilevel Monte Carlo diagnostic

Along one coupled complete allocation path,

\[
\widetilde L_L
=
\widetilde L_0
+
\sum_{\ell=1}^{L}
\{\widetilde L_\ell-\widetilde L_{\ell-1}\}.
\]

This gives a useful multilevel diagnostic.  Record

\[
V_\ell
=
\operatorname{Var}
\{\widetilde L_\ell-\widetilde L_{\ell-1}\}
\]

and the level cost.  Rapidly decreasing \(V_\ell\) supports coarse-to-fine
allocation of work.  An additive multilevel estimator is not the primary
runtime proposal because a finite estimate can be negative.

### Deterministic recursive messages

An unreduced deterministic message calculation multiplies quadrature
components and ultimately reconstructs the tensor rule.  Capping or merging
messages makes it approximate and potentially tree-dependent.  Resolution-SMC
can be viewed as a randomized bounded-population alternative in which
resampling replaces deterministic component retention.

## Split batches and ordering

The first implementation should support deterministic parent-first batches.
Do not start with particle-specific adaptive order.

Required orderings for small cases are:

1. balanced breadth-first levels;
2. a fixed parent-first priority ordering using observation energy; and
3. one deliberately unfavourable valid ordering.

For node \(v\), a first observation-space priority score is

\[
e_v
=
\mathbb E[T_v^2\mid\mathcal F_0]\,
\operatorname{Var}(\rho_v)\,
\left\|
D^{-1/2}H(u_{vL}-u_{vR})
\right\|^2.
\]

Only eligible nodes whose parents have already been revealed enter the
priority queue.  This score is a covariance diagnostic, not an error bound or
an observed-data tuning rule.

At PARIS scale, one-node stages would create 23,423 SMC steps.  The first
large experiment should therefore reveal complete breadth-first depths or
fixed batches of comparable cumulative \(e_v\).  Batch definitions must be
fixed from the prior and operator, not from realized `mf`.

## Connection with generic chaining

The shared idea is a telescoping representation across nested
approximations.  Generic chaining chooses approximating nets in the canonical
metric to bound expected suprema of stochastic processes.  This experiment
instead estimates one conditional density.  Talagrand's \(\gamma_2\)
functional is therefore not part of the first implementation.

The useful design lesson is narrower: refinement order should reflect the
operator/noise metric seen by the observations rather than geographic size
alone.  Generic-chaining or martingale maximal inequalities would become more
relevant only if a later goal required a uniform bound over observation
indices, retained masses, or partitions.

## Literature retained for later curation

The references below motivated parts of this design discussion but have not
yet been curated into a durable `inversions-knowledge` literature synthesis.
They are recorded here so that a later results write-up can distinguish
standard methodology from project-specific constructions.

### Direct foundations for the proposed experiment

- Del Moral, Doucet, and Jasra (2006),
  [*Sequential Monte Carlo samplers*](https://doi.org/10.1111/j.1467-9868.2006.00553.x),
  gives the general SMC-sampler construction for a sequence of targets known
  up to normalizing constants.  It supports the use of incremental target
  ratios and normalizing-constant estimation.  It does not analyze the
  Gamma--Beta resolution sequence proposed here.
- Stuhlmüller, Hawkins, Siddharth, and Goodman (2015),
  [*Coarse-to-Fine Sequential Monte Carlo for Probabilistic
  Programs*](https://arxiv.org/abs/1509.02962), is the closest conceptual
  precedent found for using model structure to generate information
  progressively while preserving the fine model's marginal distribution.
  Its program transformation and applications are different from the
  conditional Gamma--Dirichlet construction.
- Chopin and Papaspiliopoulos (2020),
  [*An Introduction to Sequential Monte
  Carlo*](https://doi.org/10.1007/978-3-030-47845-2), is a useful general
  reference for Feynman--Kac targets, importance resampling, normalizing
  constants, SMC samplers, particle MCMC, and sequential QMC.  It is a
  background reference rather than evidence that this particular estimator
  will be efficient.

### Guided and future-likelihood particle methods

- Heng, Bishop, Deligiannidis, and Doucet (2020),
  [*Controlled Sequential Monte
  Carlo*](https://doi.org/10.1214/19-AOS1914), formalizes the use of
  approximate optimal-control or future-likelihood information to improve
  SMC proposals.  It motivates treating the Gaussian unresolved-tail message
  as a guide rather than as the terminal scientific likelihood.
- Whiteley and Lee (2014),
  [*Twisted particle filters*](https://doi.org/10.1214/13-AOS1167), studies
  likelihood-guided changes of particle sampling law and their effect on
  marginal-likelihood variance.  Its state-space and asymptotic-in-time
  results do not transfer directly to a finite spatial refinement tree.
- Beskos, Jasra, Law, Marzouk, and Zhou (2018),
  [*Multilevel Sequential Monte Carlo with Dimension-Independent
  Likelihood-Informed Proposals*](https://doi.org/10.1137/17M1120993), is a
  related inverse-problem example of combining resolution levels with
  likelihood-informed proposals.  It assumes a different, principally
  Gaussian/PDE setting and is a follow-on lead rather than a justification
  for the present Beta proposal.

### Multilevel diagnostics

- Giles (2008),
  [*Multilevel Monte Carlo Path
  Simulation*](https://doi.org/10.1287/opre.1070.0496), supplies the standard
  cost-versus-variance logic for a telescoping Monte Carlo estimator.  The
  current plan borrows its correction-variance diagnostic; it does not yet
  claim the regularity or cost rates required by Giles's analysis.
- Beskos, Jasra, Law, Tempone, and Zhou (2017),
  [*Multilevel Sequential Monte Carlo
  Samplers*](https://doi.org/10.1016/j.spa.2016.08.004), combines multilevel
  telescoping with SMC for discretized Bayesian inverse problems.  This is
  relevant if the resolution corrections decay, but its discretization
  hierarchy is not the same as revealing latent Dirichlet coordinates.

### Downstream stochastic-likelihood inference

- Andrieu and Roberts (2009),
  [*The pseudo-marginal approach for efficient Monte Carlo
  computations*](https://doi.org/10.1214/07-AOS574), establishes why a
  suitable non-negative unbiased likelihood estimator can be used in an
  exact marginal Metropolis--Hastings construction.  This reference is
  already catalogued in `inversions-knowledge` as
  `Andrieu2009PseudoMarginal`.
- Andrieu, Doucet, and Holenstein (2010),
  [*Particle Markov chain Monte Carlo
  methods*](https://doi.org/10.1111/j.1467-9868.2009.00736.x), is the broader
  particle-MCMC continuation if the SMC estimator eventually becomes usable
  inside inference.  Neither paper makes a high-variance likelihood estimator
  practical, and neither justifies ordinary HMC with stochastic likelihoods.

### RQMC and generic-chaining boundaries

- Gerber and Chopin (2015),
  [*Sequential quasi Monte
  Carlo*](https://doi.org/10.1111/rssb.12104), provides a genuine SQMC
  construction.  It is the appropriate literature starting point if RQMC is
  later combined with resampling.  It does not justify calling ordinary SMC
  with separately scrambled Sobol propagation an RQMC method.
- Talagrand (2005),
  [*The Generic Chaining: Upper and Lower Bounds of Stochastic
  Processes*](https://doi.org/10.1007/3-540-27499-5), is the source of the
  multiscale analogy raised in discussion.  The relevant transferable idea is
  the use of nested approximations and a problem-specific metric.  Its main
  \(\gamma_2\) results concern expected suprema of stochastic processes, not
  estimation of one conditional likelihood.

These papers establish surrounding methods, not the correctness of the
project-specific target sequence.  Correctness here still depends on the
explicit Gamma--Dirichlet disintegration, exact terminal target, proposal
density, weight, and resampling calculations.  Performance remains an
experimental question.  Once results exist, the references most directly
used should be added to `inversions-knowledge/docs/literature/references.md`
and cited from a reviewed research or derivation page; the more distant leads
should remain labelled as follow-on reading.

## Existing implementation that should be reused

The first prototype should be additive rather than a new framework:

- `aggregation_error_low_rank.AdditiveDirichletAggregation` already computes
  conditional means and projected covariance factors.
- `aggregation_error_low_rank.low_rank_gaussian_log_likelihood` already
  evaluates normalized diagonal-plus-low-rank Gaussian guides.
- `aggregation_error_exact_mixture.RootResidualSpectrum` supplies the frozen
  observation-blind PARIS summary basis.
- `aggregation_error.py` and the conditional-allocation tiny screens contain
  two- and four-cell Gauss--Jacobi oracles.
- `aggregation_error_conditional_mixture` contains the authenticated
  Gamma--Beta coordinate ordering and allocation simulation.
- The G4 catalogue and report provide an observation-blind PARIS comparison
  domain.

A prospective implementation can live in
`openghg_inversions.experimental.rjmcmc.aggregation_error_resolution_smc`.
It should not alter or export existing experimental types.

## Decisions already implied

- Start with one retained root and fixed \(T_0\).
- Keep the common native \(\alpha_i\) and calibrated
  \(\eta=528.618161317525\) for the PARIS screen.
- Use float64, PCG64, canonical stable cell IDs, and deterministic tree
  construction.
- Test bootstrap SMC before a guided proposal.
- Compare IID, blocked scrambled Sobol, and SMC at matched simulator work.
- Use exact tiny oracles before PARIS.
- Treat only target/replay/numerical-integrity failures as immediate hard
  stops during exploration.
- Preserve every attempted configuration, but permit documented iteration
  through new commits and run roots.
- Do not access the protected catalogue or write `PARIS_inversions`.

## Open questions

- Does Gaussian closure predict which coarse prefixes contain
  posterior-relevant fine allocations?
- Should resolution stages reveal full depths or equal-energy node batches?
- How many fixed summary directions are needed for a useful guide?
- Does local Beta guidance offset its extra normalization cost?
- How many terminal particles are needed before
  \(\operatorname{Var}(\log\widehat Z)\) is usable?
- Can exact terminal \(HX\) updates be batched efficiently enough at
  23,424 cells?
- Does resampling destroy useful allocation diversity before boundary-heavy
  descendants are revealed?
- How sensitive are results to equivalent Gamma--Beta tree charts?
- If the estimator is useful at fixed \(T_0\), is repeated evaluation across
  \(T_0\) better handled by correlated pseudo-marginal inference, interpolation,
  or an offline conditional surrogate?

## Recommended implementation sequence

1. Implement immutable tree/frontier metadata and exact mean/covariance
   updates for two-, four-, and sixteen-cell cases.
2. Implement bootstrap resolution-SMC with no rejuvenation.
3. Establish exact tiny-oracle and replay tests.
4. Add ESS, ancestry, incremental-weight, and multilevel-correction
   diagnostics.
5. Compare valid tree orderings.
6. Add a locally guided proposal only if bootstrap degeneration is observed.
7. Run medium synthetic scaling.
8. Run an observation-blind PARIS fixed-\(T_0\) screen against matched IID
   and RQMC baselines.
9. Decide separately whether to pursue an approximate early-stopped closure
   or a fully resolved non-negative likelihood estimator.

The operational matrix is specified in
[`rjmcmc_resolution_smc_hpc_test_plan.md`](rjmcmc_resolution_smc_hpc_test_plan.md).
