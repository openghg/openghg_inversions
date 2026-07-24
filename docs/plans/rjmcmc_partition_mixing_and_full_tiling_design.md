# RJMCMC partition mixing and full-tiling design

## Purpose and status

This document records:

- the current diagnosis of slow spatial-dimension mixing in the experimental
  moving-Voronoi RJMCMC;
- the diagnostic and proposal changes that should be tried before attributing
  the problem to one mechanism;
- the distinction between a final leaf tiling and the split tree or sequential
  history used to construct it; and
- a proposed active-only RJMCMC model over full adaptive dyadic tilings.

It complements the implementation log in
[tdmcmc_numpy_numba_rewrite.md](tdmcmc_numpy_numba_rewrite.md). It is a design
reference, not a claim that the full-tiling model has been implemented or
validated. The executable validation sequence for the first diagnostic stage
is in
[rjmcmc_mixing_diagnostics_hpc_test_plan.md](rjmcmc_mixing_diagnostics_hpc_test_plan.md).

The following labels distinguish the strength of statements:

- **Code fact:** checked in the current branch.
- **Run evidence:** reported by current HPC experiments, whose run drivers and
  large outputs are not committed here.
- **Knowledge synthesis:** curated in the sibling `inversions-knowledge`
  repository.
- **Literature fact:** stated or proved in the cited primary source.
- **Inference:** a plausible explanation that still needs an experiment.
- **Decision:** the adopted direction for subsequent work.
- **Open question:** a model or implementation choice that is not settled.

## Executive conclusions

1. **The current poor mixing in `k` is not contradicted by 34--36% accepted
   up/down structural proposals.** Accepted up and down counts must nearly
   balance in any long bounded run because their difference is only the net
   change in `k`. Local accepted moves can therefore churn without traversing
   the posterior.
2. **The production run may have attempted too few genuinely independent
   changes in `k`.** Even an optimistic nearest-neighbour random-walk
   calculation gives only a few possible traversals over the observed
   separation between chains. Likelihood barriers, weakly observed cells, and
   slow coefficient/geometry relaxation can make this worse.
3. **Prior-only and powered-likelihood runs should precede parallel
   tempering.** They separate intrinsic proposal diffusion from
   likelihood-induced barriers. Tempering cannot repair a proposal that is
   already slow under the prior.
4. **A compound transition is useful but must be defined correctly.** A valid
   structural RJ kernel may be followed by ordinary coefficient-refresh
   kernels whether the structural candidate was accepted or rejected.
   Refreshing only after accepted structural moves is not generally invariant.
5. **The alternative scientific state should be the final leaf tiling, not its
   construction history.** Some tilings admit several recursive
   decompositions and many chronological split orders. Treating those
   encodings as equally weighted states silently favours tilings with more
   encodings.
6. **A direct full-tiling sampler should mix dimension-changing and
   dimension-preserving moves.** Local split/merge alone is likely to reproduce
   the current diffusive bottleneck. Candidate fixed-`K` moves include edge
   flips, paired split/merge changes, multiscale rotations, and block
   retilings.
7. **The positive coefficient prior should separate coherent total uncertainty
   from allocation contrasts.** An additive Gamma/Dirichlet construction is
   independent of the binary construction order. It also makes the projection
   analogy precise: coherent uncertainty survives aggregation, while
   fine-scale contrasts average down only when the covariance model supplies
   that averaging.
8. **The existing fixed-tree product-space work remains valuable as a small
   exact-target cross-check, not the preferred production representation.**
   Enumeration and analytic dynamic programming provide exact oracles;
   active-only RJMCMC avoids carrying all inactive coordinates. NUTS may be a
   valid within-model component, but it does not perform the dimension change.

## Terminology

For the moving-Voronoi sampler this document uses **up/down nucleus move** or
**insertion/deletion**. The traditional terms “birth/death” are retained only
when referring to existing code, historical literature, or test names. A
Voronoi insertion is not a literal binary split: the new nucleus may take cells
from several existing regions.

For a dyadic tiling, **split/merge** is literal:

- split: bisect one leaf into two admissible child rectangles;
- merge: replace two sibling rectangles by their admissible parent.

`K` denotes the number of active regions or leaves. Existing code normally
uses lower-case `k`; equations use whichever is natural in context.

## Current moving-Voronoi sampler

### Established implementation facts

**Code facts:**

- The sampler has NumPy reference and Numba accelerated paths under
  `openghg_inversions.experimental.rjmcmc`.
- A state contains a canonical sorted set of active nuclei, one dynamic
  coefficient per nucleus, a native-cell assignment, aggregated design
  columns, prediction, residual, and target caches.
- The target contains an explicit normalized prior on `k` and an explicit
  uniform prior on the canonical unordered nucleus set conditional on `k`.
- Each structural opportunity chooses insertion or deletion with probability
  one half. A direction unavailable at `k_min` or `k_max` remains an explicit
  self-transition.
- An insertion chooses an unoccupied native cell uniformly and proposes its
  coefficient from a Gaussian centred on the nearest active nucleus
  coefficient.
- A deletion chooses an active nucleus uniformly and includes the reverse
  insertion density.
- The Lunt opportunity-matched fixed-block schedule has 14 atomic transitions:
  two structural opportunities, one nucleus-location move, six always-active
  coefficient moves, and five randomly selected dynamic-coefficient moves.
- The branch has finite-state stationarity checks, continuous reciprocal-flux
  checks, NumPy/Numba parity checks, and exact restart/checkpoint tests.
- Structural state construction is incremental when possible, but complete
  rebuilding remains the validation oracle and fallback.
- Parallel tempering is not implemented in this experimental package.

These checks provide substantial evidence that the corrected structural kernel
targets the declared distribution. They do not establish adequate mixing.

### Production evidence

**Run evidence:** the completed PARIS May 2014 comparisons use 1,382
observations, 23,424 candidate inner cells, six always-active outer
coefficients, and four chains initialized alternately at `k=50` and `k=250`.
Each chain made 1,680,000 atomic proposals and retained 5,000 states. The
arithmetic-mean-one, arithmetic-SD-one run gave `k` `R-hat=1.599` and bulk
ESS 6.9, with per-chain mean `k` values 162.0, 202.8, 115.7, and 211.6. The
otherwise matched arithmetic-SD-eight run gave `R-hat=1.646` and bulk ESS 6.7.
Thus narrowing the coefficient prior did not resolve the slow
trans-dimensional mode. Other scientific summaries and outer coefficients
mixed materially better, although some country totals were also not converged.
Reported dimension-transition acceptance remained near 0.35.
This is evidence of non-convergence for `k`, not yet evidence for one
particular cause.

### Why balanced up/down acceptance is weak evidence

Let \(U_T\) and \(D_T\) be the accepted upward and downward changes by time
\(T\). When these are the only dimension changes,

\[
U_T-D_T=k_T-k_0.
\]

In a bounded run the right-hand side is small relative to the number of
transitions, so the accepted direction counts must be nearly equal even if the
chain never makes a round trip. Direction-specific acceptance remains useful
for finding an obvious implementation asymmetry, but balance is largely a
bookkeeping identity and is not a convergence diagnostic.

### An optimistic diffusion calculation

Two structural opportunities per 14-transition cycle and conditional
acceptance near 0.35 produce approximately

\[
2(0.35)=0.7
\]

accepted unit changes in `k` per cycle, or 0.05 per atomic transition. An
idealized unbiased nearest-neighbour walk needs order

\[
\frac{d^2}{0.7}
\]

cycles to diffuse across a separation \(d\). For \(d=200\), this is about
57,000 cycles. A rough slow-mode calculation for a homogeneous reflecting walk
across 496 possible values gives a relaxation scale around 71,000 cycles and
an integrated-autocorrelation scale of the same order or larger.

**Inference:** 120,000 cycles can therefore amount to only a few optimistic
traversals, even before accounting for posterior barriers and relaxation of
the rest of the state. These numbers are scale calculations, not a theorem
about the actual chain.

### Mechanisms to test

The following are compatible with the current evidence:

1. **Intrinsic nearest-neighbour diffusion.** Every accepted structural move
   changes `k` by only one.
2. **Immediate reversals.** The two structural slots are consecutive, so a
   newly inserted region can be deleted before a continuous update makes the
   new state useful.
3. **Differentiated-region ratchet.** A new coefficient begins close to its
   nearest neighbour. If it later differentiates, deleting it while leaving
   all surviving coefficients fixed can have a much larger likelihood cost.
4. **Weakly informed churn.** Uniform cell and region selection can accept many
   changes in cells with little observation sensitivity. Those changes count
   toward the acceptance rate without moving informed prediction modes.
5. **Slow refresh at large `k`.** Five random dynamic-coefficient proposals per
   cycle give a particular region approximately \(5/k\) update opportunities
   per cycle. The single location slot similarly updates one of `k` nuclei.
6. **Nonlocal ownership change from a local coefficient proposal.** A new
   Voronoi nucleus may take cells from multiple old regions even though its
   coefficient proposal is centred on only one nearest nucleus.
7. **Prior/likelihood interaction.** The induced native-grid prior changes with
   `k` when region coefficients are independent. The coefficient prior,
   geometry, model mismatch, and count prior may jointly create broad or
   separated modes.
8. **Weak scientific identification of raw `k`.** Predictions, country totals,
   and a sensitivity-weighted spatial field may be well determined even when
   many nearly equivalent weak-sensitivity partitions have different `k`.

The existing correctness tests make a new proposal-ratio defect a less likely
first explanation, but not something that should be assumed impossible.

## Diagnostic programme

### Prior and powered-likelihood ladder

Define

\[
\pi_\beta(s)
\propto
p(s)L(y\mid s)^\beta,
\qquad
\beta\in\{0,0.25,0.5,1\}.
\]

Use the same structural support, starts, schedules, and opportunity counts at
each power.

- If mixing is already slow at \(\beta=0\), the primary problem is the proposal
  graph, structural prior, or coupling within the state.
- If mobility deteriorates sharply as \(\beta\) rises, a likelihood barrier is
  implicated and tempering becomes better motivated.
- If raw `k` remains slow while flux, totals, and predictions mix, scientific
  weak identification rather than a purely computational failure is plausible.

A prior-only `k` histogram need not look uniform in a short correlated run.
Compare mobility with an explicit lazy nearest-neighbour reference chain having
the same support and approximate accepted-move rate.

### Diagnostics to retain

At minimum record:

- mean-squared displacement of `k` against lag;
- first-passage and round-trip times between declared low/high bands;
- accepted edge flows \(N(k,k+1)\) and \(N(k,k-1)\);
- self-transition and immediate-reversal counts;
- proposed and accepted changes in log likelihood, log prior, and log target;
- changed-cell count and affected-design fraction;
- whitened prediction change from each structural proposal;
- sensitivity of inserted/deleted regions and whether the event is in an
  informed or weak-sensitivity stratum;
- age of a deleted region and its coefficient contrast with neighbours;
- age since the last coefficient and geometry update for each active region;
- `R-hat` and effective sample size for native-grid flux, country totals,
  predicted observations, model mismatch, and `k`;
- accepted effective raster or prediction change per second; and
- wall time and CPU time per effective draw.

The diagnostics should distinguish an accepted change of representation from
an accepted change in a scientifically relevant prediction direction.

### Implemented diagnostic boundary

**Code fact:** point 1 is implemented as an opt-in structural-event stream.
`SamplerConfig.collect_structural_diagnostics` and the corresponding
`continue_sample` keyword add one row for each insertion, deletion, or
nucleus-location attempt. The result is separate from the retained posterior
trace and checkpointed transition kernel. Collection requires explicit
`StructuralDiagnosticsProvenance`: a stable chain ID plus the same durable
problem SHA-256 used by checkpoint I/O. This prevents unrelated segments from
being combined merely because their transition bounds and endpoint nuclei
happen to match.

The implementation records:

- global atomic-transition coordinate, validity, invalidity reason, acceptance,
  and source/candidate/result `k`;
- every cached target-component delta and every Metropolis--Hastings proposal
  term;
- removed and added nucleus identities, owner-changed cell count/fraction, and
  affected candidate design-column count;
- candidate prediction, observation-error-standardized prediction, and event
  design-column norms; and
- the signed log coefficient ratio at the edited region.

Segment endpoint nucleus sets make residence intervals reconstructable across
checkpoint boundaries, including initially active left-censored regions and
final right-censored regions. Immediate reversal requires consecutive global
atomic-transition coordinates, rather than merely consecutive structural rows.
Region lineages receive monotonic diagnostic IDs and separately retain their
origin nucleus, so deleting a region and later reusing its cell cannot collide
with the earlier lineage.
The ordinary retained output remains the source for posterior scientific
summaries such as predictions, native-grid flux, outer coefficients, and
reporting totals.

`structural_diagnostics_to_dataset` supplies an independent
`structural_transition` xarray dimension for checkpoint-segment persistence.
It labels standardization by `observation_sd` explicitly: this is complete
whitening for the fixed diagonal error model, but not for the inferred-OU
covariance.

### Parallel tempering

Parallel tempering should be a later experiment, not the first diagnosis.
Measure both swap acceptance and replica round trips through temperature,
together with `k` and partition traversal at each temperature.

The archived Brazil configuration is a warning: it enabled a four-temperature
ladder but recorded zero accepted swaps in 275,000 attempts. It is evidence
that merely enabling temperatures does not establish a functioning tempering
kernel.

Tempering can help a likelihood energy barrier. It cannot make a local
prior-only proposal traverse its geometry quickly, and it does not repair an
incorrect target or proposal ratio.

## Improving the current structural transition

### Ordinary compound kernels

Let \(K_{\mathrm{RJ}}\) be a valid structural transition and
\(K_{\mathrm{coef}}\) a valid within-model coefficient transition. Either
composition

\[
K_{\mathrm{RJ}}K_{\mathrm{coef}}
\quad\text{or}\quad
K_{\mathrm{coef}}K_{\mathrm{RJ}}
\]

preserves the target. The composition need not itself be reversible.

This supports:

- inserting ordinary coefficient rejuvenation between the two structural
  slots;
- refreshing coefficients after every structural opportunity, whether the
  structural proposal was accepted or rejected; and
- using several fixed, target-invariant continuous updates for each structural
  opportunity.

It does **not** justify running a coefficient transition only when a structural
proposal was accepted. That creates state-dependent scheduling and is not
generally invariant. If continuous refinement is intended to improve the
acceptance decision for one structural jump, it must instead be a joint or
path-augmented proposal with complete forward/reverse accounting.

### Relevance-conditioned selection

A replaceable selection rule can mix a relevance score with uniform support:

\[
q(r\mid s)
=
\epsilon\frac{1}{|\mathcal E(s)|}
+
(1-\epsilon)
\frac{w_r(s)}
{\sum_{j\in\mathcal E(s)}w_j(s)},
\qquad \epsilon>0.
\]

Here \(\mathcal E(s)\) is the eligible set and \(w_r(s)\) is a non-negative
score. The uniform component retains global support. The Metropolis--Hastings
ratio must use the selection probability in the source state and recompute the
reverse probability in the candidate state.

Start with fixed scores derived from prior flux and whitened design sensitivity
because they are cheap and auditable. Residual-dependent scores are valid in
principle but require candidate-state reverse normalizers. A global
`H.T @ residual` calculation at every attempt would likely erase much of the
incremental-geometry speed gain; cached region-level design columns are the
more plausible first implementation.

## Tiling state versus construction state

### Three different objects

Let:

- \(\mathcal P\) be the set of canonical leaf tilings;
- \(\mathcal D\) be the set of recursive decomposition-tree encodings; and
- \(\mathcal H\) be the still larger set of chronological split histories.

A canonical set of non-overlapping leaves that exactly covers the domain is
one-to-one with the tiling of interest. The maps

\[
\mathcal H\longrightarrow\mathcal D\longrightarrow\mathcal P
\]

are generally many-to-one.

For example, the four-square tiling of a square can be constructed by a
vertical root split followed by horizontal splits of both halves, or by a
horizontal root split followed by vertical splits of both halves. Independent
subtree splits can also be interleaved in several chronological orders.

Thus \(\mathcal P\) is a quotient of the construction encodings, not another
name for them.

### Multiplicity changes the induced prior

Choose one encoding space
\(\mathcal E\in\{\mathcal D,\mathcal H\}\). Let \(\phi(E)\) return the leaves
of encoding \(E\), and let

\[
m_{\mathcal E}(P)
=
|\{E\in\mathcal E:\phi(E)=P\}|.
\]

If an intended tiling target \(\pi^\star_{\mathcal P}\) is copied unchanged to
each construction encoding,

\[
\pi_{\mathcal E}(E)
\propto
\pi^\star_{\mathcal P}(\phi(E)),
\]

then its induced tiling law is

\[
\pi_{\mathcal P}(P)
=
\sum_{E\in\mathcal E:\phi(E)=P}\pi_{\mathcal E}(E)
\propto
m_{\mathcal E}(P)\pi^\star_{\mathcal P}(P).
\]

Tilings with more decompositions or histories are silently overweighted.
The two multiplicities are different: chronological histories additionally
count alternative interleavings of otherwise independent subtree splits.

A correct history augmentation has the form

\[
\pi_{\mathcal E}(E,x)
=
\pi^\star_{\mathcal P}(\phi(E),x)
r_{\mathcal E}(E\mid\phi(E)),
\qquad
\sum_{E\in\mathcal E:\phi(E)=P}r_{\mathcal E}(E\mid P)=1.
\]

Uniform weighting over the selected encoding space requires
\(r_{\mathcal E}(E\mid P)=1/m_{\mathcal E}(P)\), which may be hard to compute.
The simpler **decision** is to retain the canonical leaf tiling as the
scientific state and use construction information only as a temporary proposal
auxiliary.

If genealogy or refinement history is scientifically meaningful, a prior on
histories may be intentional, but it is then a different model and its induced
leaf law should be reported.

### This is not the scheduler defect

These two issues must remain separate:

- A deterministic or random transition schedule is valid when its complete
  component kernels preserve the declared target.
- Construction multiplicity concerns which probability distribution was
  declared on scientific tilings before sampling begins.

The old deterministic one-way structural schedule failed the first condition.
A tree-history model can have perfectly correct Markov transitions and still
target an unintended multiplicity-weighted leaf distribution.

### Multiple proposal paths

Even when histories are absent from the retained state, several random
constructions may lead from \(P\) to the same endpoint \(P'\). An
endpoint-density proposal must aggregate them:

\[
q(P'\mid P)
=
\sum_u q(u\mid P)
\mathbf 1\{f(P,u)=P'\}.
\]

Alternatively, retain enough temporary auxiliary information to define a
one-to-one forward/reverse augmented move and include its auxiliary densities
and Jacobian. Proposal paths may be hidden from the saved state, but not from
probability accounting.

For the first fixed-\(K\) prototype, keep the sampled path

\[
u=(\text{merge pair},\text{destination split},\text{new fraction})
\]

until the Metropolis--Hastings decision. The edge flip merges two midpoint
friends and splits their parent in the perpendicular orientation. Resolution
relocation merges a midpoint-friend pair in one place and splits a different
leaf. In both cases the reverse path is constructed explicitly, giving an
involution

\[
(P,u)\longleftrightarrow(P',u').
\]

Detailed balance can then be checked for each paired path. Different
auxiliaries may lead to the same endpoint without requiring an endpoint-level
sum, provided that every retained auxiliary has one well-defined reverse and
its exact source and reverse probabilities appear in the acceptance ratio.
The path is proposal state, not scientific construction history, and is
discarded after acceptance or rejection.

Expressing an edge flip or relocation as a merge followed by a split simplifies
this proposal accounting, but does **not** simplify an arbitrary endpoint
structural prior. The intermediate \(K-1\) partition is only an auxiliary and
is never accepted as a separate Markov state. Only the structural-prior ratio
between the two fixed-\(K\) endpoints enters. Treating merge and split as two
separately accepted moves would define a different chain that visits \(K-1\)
and requires a target there.

## Partition and count priors

The structural prior is distinct from the prior on leaf masses or
coefficients.

Let \(N_K\) be the number of admissible canonical tilings with \(K\) leaves. If
the intended count prior is \(p_K(K)\) and tilings are conditionally uniform,

\[
p(P)
=
\frac{p_K(K(P))}{N_{K(P)}}.
\]

Assigning the same unnormalized weight to every tiling instead induces

\[
p(K)\propto N_K,
\]

before any other count penalty. More generally, a common per-tiling weight
\(w(K)\) induces \(p(K)\propto N_Kw(K)\).

For a masked, grouped, adaptive dyadic family, computing \(N_K\) may be
difficult. Conditional uniformity is not required. A directly defined geometry
energy \(\widetilde p(P)\), whose ratios are computable, is also valid, but its
induced count prior must be measured and documented rather than described as
uniform.

For exact posterior sampling, the structural-prior **ratio used by every
accepted/rejected proposal must be exact for the declared target**. It need not
be normalized when the normalizing constant is common to the compared states:
an exactly evaluated local energy
\(\widetilde p(P)\propto\exp[-Z(P)]\) needs only
\(-Z(P')+Z(P)\). An approximate ratio instead samples an approximate,
generally unknown target unless it is followed by an exact delayed-acceptance
correction or arises from a valid unbiased pseudo-marginal estimator.

At fixed \(K\), a target uniform over the unique canonical tilings has

\[
\log p(P'\mid K)-\log p(P\mid K)=0.
\]

The count \(N_K\) therefore cancels from every fixed-\(K\) move. This does not
remove the need for exact forward/reverse eligible-choice probabilities. It
also does not solve variable-\(K\) inference: a prescribed marginal \(p_K(K)\)
with conditional uniformity requires the cross-\(K\) factor
\(p_K(K)/N_K\), or a different explicitly declared and evaluable tiling prior.

**Open question:** define the admissible production tiling family and its
structural prior before implementing its sampler. Counting construction trees
or prunings of one fixed frontier is not a substitute for counting distinct
full tilings.

## Positive prior on a full tiling

### Order-independent total and allocation

Let \(P=\{R_1,\ldots,R_K\}\) be the canonical leaves and let
\(\alpha\) be a finite additive base measure:

\[
\alpha(A\cup B)=\alpha(A)+\alpha(B)
\]

for disjoint admissible regions. Define

\[
T\sim p_T,
\qquad
W_P\sim
\operatorname{Dirichlet}
\left(\alpha(R_1),\ldots,\alpha(R_K)\right),
\qquad
X_{R_i}=T W_{P,i}.
\]

The distribution of the leaf masses depends only on the final regions. Any
binary representation of the same allocation uses

\[
\rho
\sim
\operatorname{Beta}
\left(\alpha(R_L),\alpha(R_R)\right).
\]

Beta shapes add when regions are merged, so the result is independent of which
valid binary decomposition was used.

Two useful specializations are:

1. A compatible Gamma total plus Dirichlet shares gives independent
   common-rate Gamma masses.
2. A separately calibrated Gamma or other positive root total plus Dirichlet
   shares decouples coherent total uncertainty from allocation uncertainty,
   while retaining order-independent aggregation.

By contrast, arbitrary node-specific
\(\operatorname{Beta}(\kappa_vp_v,\kappa_v(1-p_v))\) factors define a general
Dirichlet-tree prior. That prior is projectively consistent under pruning of
the selected canonical tree, but is not generally invariant to an alternative
split orientation or decomposition. It becomes order-independent only when
the node shapes come from one globally additive base measure.

### Relation to projected fine-grid uncertainty

Aggregation is a projection, not new information. For an aggregate
\(a=w^\mathsf T x\) with native covariance \(S\),

\[
\operatorname{Var}(a)=w^\mathsf T S w.
\]

For an equally weighted mean of \(n\) cells with marginal variance
\(\sigma^2\):

- independent deviations give variance \(\sigma^2/n\);
- a perfectly coherent common mode gives variance \(\sigma^2\); and
- exchangeable correlation \(\rho\) gives
  \(\sigma^2[\rho+(1-\rho)/n]\).

This is the precise version of the basis-projection analogy. Grouping cells
should not manufacture precision. Fine-scale contrast modes average down only
to the extent that the prior says they cancel; a coherent country, sector, or
root mode survives aggregation.

The positive total/allocation construction mirrors that separation:

- the root total carries coherent uncertainty that is unchanged when children
  are regrouped; and
- the shares carry mass-preserving contrasts that cancel when children are
  merged.

Under an additive-mass Gamma/Dirichlet model, a small-mass leaf can have wider
**relative scaling** uncertainty while retaining modest **absolute flux**
uncertainty. This is a consequence of that chosen base-measure model, not a
universal consistency requirement. A shared root or group factor prevents a
large aggregate from becoming spuriously precise merely because it contains
many leaves.

This interpretation is consistent with the covariance projection and
correlation-adjusted effective-cell-count material in `inversions-knowledge`.

### When can the data learn a tiling?

The Gamma/Dirichlet construction specifies \(p(x\mid P)\), not \(p(P)\).
Moreover, if every tiling is only an exact marginal view of one fixed native
model and all hidden within-leaf contrasts are integrated exactly, then

\[
p(y\mid P)=p(y).
\]

The data cannot select among exact representations of the same model. A
nontrivial posterior over tilings requires a genuine reduced-model decision,
for example:

- within-leaf allocation fixed at its nominal value;
- unresolved contrasts removed;
- unresolved contrasts strongly shrunk in a partition-dependent way; or
- an explicitly labelled generalized-Bayes compression utility.

The first full-tiling prototype should state this choice directly. A good
proposal score is not automatically a partition posterior.

## Full-tiling structural moves

### Scope of the dyadic-tiling literature

**Literature fact:** Cannon, Levin, and Stauffer (2017) study all tilings of the
unit square by \(n=2^j\) equal-area dyadic rectangles, with fixed \(n\) and a
uniform target. Their lazy edge-flip chain replaces a valid shared bisecting
edge by its perpendicular bisector. They prove relaxation time
\(O(n^{4.09})\) and mixing time \(O(n^{5.09})\), with a superlinear lower
bound. Their discussion also notes all-scale rotations and shows that changing
to an edge-length-weighted target can change the mixing regime qualitatively.

This supports a fixed-`K` geometry kernel and a small exact benchmark. It does
not establish practical mixing for \(K\approx500\), unequal-area adaptive
leaves, masks, a changing dimension, or an atmospheric posterior.

**Literature fact:** Angel et al. (2012), *The Phase Transition for Dyadic
Tilings*, is a probability/combinatorics paper, not an MCMC-move paper. Its
children, admissible “friends”, chains, recursive decompositions, and
binary-coordinate symmetries can inspire proposals, but those proposals and
their correctness are new project work.

### Candidate kernel mixture

A direct active-only sampler should eventually mix:

\[
K_{\mathrm{dimension\ RJ}}
K_{\mathrm{edge\ flip}}
K_{\mathrm{multiscale}}
K_{\mathrm{block}}
K_{\mathrm{continuous}}.
\]

Candidate components are:

1. **Split/merge RJ:** bisect one admissible leaf or merge two sibling leaves.
2. **Local edge flip:** replace one shared bisector with the perpendicular
   bisector while preserving `K`.
3. **Paired merge/split:** merge in one location and split elsewhere, preserving
   `K` while moving resolution.
4. **Multiscale rotation or coordinated flip:** change several compatible
   edges within an admissible dyadic block.
5. **Block retile:** select a block currently covered by \(m\) leaves and
   propose another valid \(m\)-leaf tiling of that block.
6. **Coordinate symmetry:** apply a valid dyadic involution within a permitted
   block. Whole-domain symmetries are most plausible in prior-only tests and
   may accept poorly under heterogeneous NAME sensitivity.
7. **Continuous rejuvenation:** update active masses/scalings after each
   structural opportunity.

For a physical-mass split,

\[
(M,\rho)\longmapsto(M\rho,M(1-\rho))
\]

has absolute Jacobian \(M\). A fixed-`K` retile also changes coefficient
meaning; equal dimension does not eliminate the need for a destination density
or invertible transport.

Each kernel must account for:

- kernel and direction selection;
- eligible leaf, edge, block, scale, and orientation counts;
- multiple constructions of the same endpoint;
- the structural-prior ratio;
- continuous auxiliary densities;
- reverse candidate counts in the candidate state;
- the Jacobian where coordinates are transformed; and
- invalid options as self-transitions unless their removed probability mass is
  explicitly renormalized.

## Product space, active RJMCMC, and NUTS

The sibling knowledge base distinguishes:

- **fixed-capacity storage**, which by itself has no probabilistic meaning for
  inactive slots;
- **persistent product space**, in which every inactive coordinate has a
  proper normalized pseudo-prior; and
- **active-only RJMCMC**, in which inactive coordinates are absent and
  dimension-changing accounting is explicit.

The existing Gamma--Beta branch uses a fixed maximal tree and an
ancestry-consistent active mask while all potential continuous coordinates
remain in a product space. It is useful for tiny exact targets, dynamic
programming, and product-space/RJ cross-checks. It is also restricted to
prunings of one selected hierarchy and does not represent every full tiling.

**Decision:** use enumeration and analytic dynamic programming as exact
correctness oracles, and retain product-space MCMC as an exact-target
cross-check. Prefer active-only NumPy/Numba RJMCMC for a production full-tiling
prototype.

A compound RJ/HMC or RJ/NUTS sampler is mathematically possible: a valid
within-model HMC/NUTS kernel can be composed with a valid RJ kernel. NUTS does
not itself perform the dimension change, and model-conditioned adaptation and
the cost of inactive coordinates remain substantial. It is therefore a later
experiment, not the first implementation path.

**Run evidence:** the completed SD-one production chains peaked near 0.76 GB
RSS, far below their allocation. Memory can therefore be traded for speed in
this problem. Precomputed or lazily cached design columns for a bounded
supertree are plausible; a full adaptive-tiling rectangle catalogue has a
different size and must be benchmarked rather than assumed cheap.

## Benchmark programme

| Scale | Target | Geometry | Kernel comparison | Primary evidence |
| --- | --- | --- | --- | --- |
| Tiny enumeration | Prior only | Voronoi and tiling | Each kernel and mixtures | Exact normalization, detailed balance, stationarity |
| Small equal-area dyadic | Uniform fixed `K` | Full tiling | Edge flip, block, mixture | Frequencies and mixing against full catalogue |
| Small adaptive dyadic | Declared `p(K)` and `p(P | K)` | Variable `K` | Split/merge plus fixed-`K` moves | Exact `P` and `K` frequencies |
| Synthetic checkerboard | \(\beta=0,0.25,0.5,1\) | Voronoi and tiling | Baseline, compound, relevance-conditioned | Barrier diagnosis, prediction and field recovery |
| NAME/EDGAR example | Full likelihood | Fixed, greedy, Voronoi, tiling | Matched opportunities and wall time | Prediction, totals, flux and structural mixing |
| PARIS production | Full declared model | Dispersed and greedy starts | Best validated kernels | Round trips, scientific ESS, ESS/time |

Include these controls:

- planted fixed basis where a planted basis exists;
- deliberately over- and under-resolved fixed bases;
- fixed `K` with movable geometry;
- a greedy/SLS basis held fixed;
- variable-`K` chains initialized low, high, and from the greedy basis.

Greedy splitting and stochastic local search are initializers, proposal guides,
or fixed-basis comparators. They are not posterior sampling.

Voronoi and tiling models should not be described as targeting the same
posterior unless their induced native-field and structural priors are actually
matched. Prediction and reporting-total comparisons remain meaningful even
when the structural models differ.

## Reusable work and evidence boundary

The sibling `inversions-knowledge` repository was inspected at revision
`e77d20cffe7ee0298d9106065c962d24198dabdc` (“Document RJMCMC with HMC and
NUTS”) on 2026-07-23. It supplies curated background for:

- active-only RJMCMC, temporary augmented move spaces, and persistent product
  spaces;
- move reachability versus the declared scientific model;
- testing the actual schedule, boundary self-mass, proposal normalization,
  finite transition matrices, and continuous auxiliary maps;
- fixed-canonical-tree Gamma--Beta/Dirichlet-tree algebra;
- projective consistency under pruning one selected tree;
- exact-representation evidence invariance;
- compound RJ plus within-model HMC/NUTS kernels; and
- covariance-weighted projection and coherent versus independent averaging.

Most relevant repository-relative pages are:

- `docs/topics/trans-dimensional-mcmc-representations.md`;
- `docs/literature/trans-dimensional-mcmc-foundations.md`;
- `docs/workflows/validating-trans-dimensional-mcmc-kernels.md`;
- `docs/derivations/rjmcmc-dimension-matching-and-augmented-spaces.md`;
- `docs/topics/positive-multiscale-priors-and-gamma-beta-trees.md`;
- `docs/derivations/gamma-beta-tree-covariance.md`;
- `docs/derivations/covariance-weighted-prolongation.md`;
- `docs/topics/prior-design-and-sensitivity.md`;
- `docs/research-trails/dyadic-search-to-gamma-beta-product-space.md`; and
- `docs/research-questions/rjmcmc-hmc-nuts-and-transported-tuning.md`.

The following are new project synthesis or decisions and should not be
attributed to that knowledge base:

- the present production `k`-mixing diagnosis;
- the powered-likelihood diagnostic ladder;
- direct full-tiling state rather than one fixed tree;
- the tiling/decomposition/history multiplicity calculation;
- the proposed full-tiling structural kernel mixture; and
- application of one order-independent additive allocation prior across
  alternative tiling decompositions.

Existing OpenGHG Inversions draft work can be reused as follows:

- the current moving-Voronoi package supplies target/proposal interfaces,
  NumPy/Numba parity patterns, finite and continuous oracles, incremental design
  aggregation, schedules, checkpoints, and output conventions;
- the Gamma--Beta branch at immutable commit
  `d12b5fd84fc3dbd20a5ad15383894bafa01b076b` supplies positive
  total/allocation calculations, partition-mask logic, count-prior dynamic
  programming on one fixed tree, and product-space exact-target cross-checks;
- the projected-prior and product-space draft PRs supply exact small Gaussian
  comparisons and inactive-coordinate semantics; and
- the greedy splitting/SLS work supplies preconstructed initial bases and
  proposal scores, not an MCMC target.

## Decisions

1. The current corrected moving-Voronoi implementation remains the baseline
   sampler. Slow `k` mixing is a performance/identifiability question unless a
   new correctness counterexample is found.
2. Diagnose the baseline with prior-only and powered-likelihood runs before
   adding parallel tempering.
3. Add proposal-flow and scientific-state diagnostics before evaluating new
   kernels.
4. Try ordinary continuous rejuvenation and a replaceable
   relevance-conditioned structural selector on the Voronoi model.
5. Treat a canonical adaptive leaf tiling as the scientific state of the
   alternative model. Do not retain construction history as an unnormalized
   model dimension.
6. Define `p(P)` and its induced or explicit `p(K)` before implementing the
   alternative sampler.
7. Use an order-independent additive total/allocation prior for comparisons
   across alternative tiling decompositions. Retain arbitrary
   node-concentration Gamma--Beta models as fixed-tree comparators.
8. Combine dimension-changing and dimension-preserving geometry kernels in the
   full-tiling prototype.
9. Use active-only NumPy/Numba RJMCMC for the first full-tiling implementation.
   Retain enumeration and analytic dynamic programming as exact correctness
   oracles, and fixed-tree product-space MCMC as an exact-target cross-check.
10. Keep the NAME/EDGAR checkerboard as a replayable example and use tiny
    exact state spaces for regression tests.

## Open questions

- What exactly is the production tiling family: unequal-depth adaptive dyadic
  rectangles on each hard domain group, or a narrower family?
- How are land/ocean, InTEM outer regions, country boundaries, and disconnected
  masks represented in the admissibility rules?
- Can the number \(N_K\) of distinct admissible tilings be computed by dynamic
  programming for the chosen family, or should a different direct geometry
  prior be used?
- What reduced-model assumption makes the tiling genuinely learnable rather
  than an exact re-expression of one native prior?
- Should the active continuous state use physical leaf masses, leaf scaling
  factors, root-plus-simplex shares, or mean-preserving contrasts?
- How should the root total and allocation concentration be calibrated to
  country, sector, and native-cell uncertainty targets?
- Should different flux components share root factors, allocation factors, or
  neither?
- Which fixed-`K` move mixture is connected under the final masks and
  admissibility rules?
- Can a paired merge/split or block-retile proposal avoid the
  differentiated-region deletion ratchet?
- Which fixed sensitivity or information score gives useful proposal guidance
  without requiring a full native-grid residual transpose on every attempt?
- Does the current posterior contain scientifically meaningful uncertainty in
  `k`, or mostly multiplicity among weakly informed partitions?

## Recommended implementation sequence

1. **Implemented:** add opt-in diagnostics for `k` flow, reversals, region
   residence, changed ownership/sensitivity, target accounting, and
   prediction-space change to the current sampler.
2. Run the paired instrumentation benchmark and the ordinary
   full-likelihood Voronoi profile, then compare prior-only
   \(\beta=0\) and posterior \(\beta=1\) mobility with production support,
   starts, schedules, and opportunity counts. Add an intermediate power only
   if the endpoints differ materially.
3. **Implemented as a NumPy reference baseline:** add active-only Gamma--Beta
   RJ on one canonical fixed-direction tree. The state, normalized priors,
   local split/merge accounting, exact tiny-tree enumeration, seeded
   structural sampler, and in-memory continuation are specified in
   [rjmcmc_gamma_beta_baseline.md](rjmcmc_gamma_beta_baseline.md). The preserved
   fixed-tree product-space cross-check remains a validation follow-up before
   alternative orientations or full tilings.
4. Interleave ordinary coefficient rejuvenation and benchmark a fixed-score
   relevance-conditioned selector as replaceable kernels.
5. Specify the full-tiling domain, canonical leaf representation, masks,
   structural prior, and reduced-model interpretation.
6. Enumerate all states of a tiny tiling problem and calculate exact target
   probabilities.
7. Implement the order-independent total/allocation prior and validate its
   aggregation identities and moments by enumeration and simulation.
8. Implement fixed-`K` edge-flip and resolution-relocation kernels under the
   tiny uniform target. Defer block retile until the local pathwise kernels
   have an exact finite-state benchmark.
9. Implement transparent active-only NumPy split/merge RJ and validate it
   against enumeration and the fixed-tree product-space exact-target
   cross-check.
10. Add active continuous rejuvenation and a simple observation likelihood.
11. Add Numba only after exact NumPy parity.
12. Add relevance-conditioned and likelihood-informed kernels behind swappable
   interfaces.
13. Add parallel tempering only if the powered-likelihood ladder identifies a
   likelihood barrier and the swap kernel passes balance and round-trip tests.

### Compact restart point

- **Next executable task:** run the paired diagnostic-overhead check and
  ordinary full-likelihood Voronoi profile described in the HPC test plan.
- **Alternative baseline underway:** the local split/merge RJ implementation
  on one fixed-direction Gamma--Beta tree now passes exact stationarity,
  empirical mobility, continuation, and RHIME design-unit equivalence checks.
  Cross-check its target against the archived product-space oracle, then add
  the separate root/fraction refresh kernels to a compound schedule before
  treating it as posterior inference or profiling larger fixed trees.
- **Alternative-model prerequisite:** settle the full-tiling structural target
  `p(P)`/`p(K)` and the reduced-model meaning before writing structural code.
- **Correctness oracle:** tiny canonical tiling enumeration and analytic
  dynamic programming, with a fixed-tree product-space cross-check.
- **First full-tiling kernel:** fixed-`K` edge flip on an equal-area tiny state,
  mixed with resolution relocation; represent both as one merge/split
  auxiliary involution. Block retile is deliberately deferred. Follow with
  adaptive split/merge only after the variable-\(K\) state/prior contract is
  explicit.
- **Data dependency:** none through the synthetic and NAME/EDGAR stages.
- **Performance rule:** add Numba and cached aggregation after, not during, the
  target/proposal validation.

## Primary references

- Green, P. J. (1995), “Reversible jump Markov chain Monte Carlo computation
  and Bayesian model determination,” *Biometrika* 82(4), 711--732.
- Cannon, S., Levin, D. A., and Stauffer, A. (2017), [“Polynomial Mixing of the
  Edge-Flip Markov Chain for Unbiased Dyadic
  Tilings”](https://doi.org/10.4230/LIPIcs.APPROX-RANDOM.2017.34),
  *LIPIcs APPROX/RANDOM 2017*, Article 34.
- Angel, O., Holroyd, A. E., Kozma, G., Wästlund, J., and Winkler, P. (2012),
  [“The Phase Transition for Dyadic
  Tilings”](https://arxiv.org/abs/1107.2636), arXiv:1107.2636v3.
- Lunt, M. F. et al. (2016), “Estimation of regional methane fluxes in Europe
  using a Bayesian, multiscale approach,” *Atmospheric Chemistry and Physics*
  16, 3213--3225.
