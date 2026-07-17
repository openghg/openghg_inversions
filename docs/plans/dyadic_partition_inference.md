# Dyadic Basis Optimization and Partition Inference

## Status and purpose

This is a design and background note, not an implemented sampler contract. It
collects the basis-partition work that is currently spread across repository
branches, planning notes, IPython histories, and a private discussion transcript.
Its first implementation milestone is to move the useful prototype code into
this repository with synthetic tests. Subsequent work should refer to that public
reference implementation rather than to private histories.

The short-term execution checklist, local TAC/MHD data contract, demo artifacts,
and hackathon stop conditions are maintained separately in
`docs/plans/dyadic_sls_hackathon.md`.

As of 2026-07-17, the exact Gaussian projection oracle, independent-prior
native posterior summaries, DFS/Fisher/Equation 45 objectives, exact additive
dynamic programs, land/ocean and rectangular comparisons, and a
native-resolution semi-synthetic TAC/MHD report are implemented on the
experimental branch. The Bocquet projection line is paused. The next partition
inference work is OGI-062 through OGI-065: this reference update, a tiny exact
fixed-contrast product-space kernel, a native PyMC adapter, and separately
reviewed transported split/merge proposals.

The next native-resolution Gaussian validation is specified separately in
`docs/plans/bocquet_projection_validation.md`.
Its blockwise projection-consistency experiment is optional; the immediate
priority is the exact projection oracle, TAC/MHD reference analysis, and
comparison of requested and adaptive representations.

The motivating problem is an atmospheric inverse model

\[
y = H_P x_P + \epsilon,
\]

where the basis partition \(P\) determines an aggregated emissions block of the
sensitivity matrix, \(x_P\) contains the corresponding coefficients, and other
blocks of the inverse problem can remain fixed. The immediate goal is a
reproducible simulated-annealing (SA) demonstration over dyadic partitions. The
longer-term goal is posterior inference over both the partition and continuous
parameters, including non-Gaussian positive priors such as lognormal priors.

This note distinguishes three statuses:

- **Established**: a mathematical result, existing repository behavior, or
  behavior observed directly in the prototype.
- **Proposed**: the current implementation recommendation.
- **Open**: a choice that needs a focused experiment or design decision.

## Executive decisions

1. **Proposed:** collect and test the dyadic data structures, multiscale
   sensitivity construction, split/merge moves, Python and Numba bisection, and
   SA runner before implementing joint MCMC. This is Phase 0 below.
2. **Proposed:** retain SA as a useful optimizer and initializer. Do not describe
   its DoFS-like energy as a non-Gaussian posterior target.
3. **Proposed:** make the first statistically exact Gaussian demonstration an
   exact Bocquet-consistent posterior-compression experiment. Compare projected
   analyses with restrictions of the native analysis and optimize fixed-count
   representations with DFS, Fisher, and the data-dependent Equation 45
   criterion. Do not describe this as posterior inference over \(P\): the exact
   scale-consistent Gaussian likelihood is partition-invariant.
4. **Proposed:** treat a collapsed Gaussian sampler over partition probabilities
   as a separate reduced-model experiment. It requires a model in which \(P\)
   genuinely changes the prior or likelihood; it is not obtained by merely
   reparameterizing one native Gaussian model exactly.
5. **Proposed:** treat non-Gaussian fixed-count continuous sampling as a later
   experiment, not as a prerequisite for the partition proof of concept.
6. **Proposed:** investigate a tree-contrast product-space sampler next. It is a
   genuine fixed-dimensional alternative to RJMCMC when all potential node
   parameters and proper pseudo-priors are included in the augmented state.
7. **Established:** storing a fixed indicator over all possible multiscale tiles
   does not by itself avoid RJMCMC. If the mathematical continuous state is a
   packed vector whose dimension changes with the number of active tiles, the
   method is trans-dimensional regardless of how the indicator is stored.
8. **Proposed:** use exact likelihood and prior terms for final partition
   acceptance. DoFS, split-contrast, projected-flux, or Laplace scores may inform
   proposals, initialize chains, or provide a delayed-acceptance first stage.
9. **Proposed:** represent land/ocean, country, inner/outer, and other hard
   partitions as constraints and groups around the dyadic optimizer. Filtering
   of observations must occur before any data-dependent basis weights are built.
10. **Proposed:** use Equation 45 as the primary data-dependent compression
    experiment, with DFS and Fisher retained as comparators. Predeclare the
    dictionary and loss, label the result as adaptive posterior compression, and
    use held-out sites or times for predictive assessment.
11. **Proposed:** generate observations from the declared product-space model
    for the first joint-inference demonstration. Use fixed InTEM outer regions
    to restrict the dynamic tree to an inner domain, and give always-active
    outer coefficients a separate prior group.
12. **Proposed:** use a fixed root-plus-contrast state as the correctness
    baseline. Add RJMCMC-like coefficient transport only as a later proposal
    improvement, without changing the augmented target.

## Terminology and notation

| Symbol | Meaning |
| --- | --- |
| \(y\) | retained observations after filtering |
| \(G\) | fine-grid contribution/sensitivity matrix for the block being aggregated |
| \(H_P\) | columns of \(G\) aggregated according to partition \(P\) |
| \(x_P\) | active region coefficients for \(P\) |
| \(P\) | a valid partition of the relevant grid cells |
| \(A(P)\) | active leaf tiles under partition \(P\) |
| \(R\) | observation-error covariance, or the appropriate effective covariance |
| \(B_P\) | prior covariance in a linear-Gaussian approximation |
| \(z\) | unconstrained continuous coordinates, often with \(x=\exp(z)\) |
| \(q_P\) | proper pseudo-prior for parameters inactive under \(P\) |
| \(K(P)\) | number of active regions or leaves |
| \(K_{\mathrm{tree}}\) | number of active tree leaves, including any unsupported leaves not yet pruned |
| \(K_{\mathrm{eff}}\) | number of supported active coefficient directions; not DFS |

"Multiscale sensitivity" means pre-aggregated columns for candidate tiles. A
"multiscale weight" is a score derived from those columns. These are not the
same object and should not share an ambiguous array name.

When hard classes are present, this note uses **superforest** for the collection
of one predetermined dyadic tree per class or connected component. A valid
global partition is a pruning of every tree in that superforest.

## Current repository foundations

The repository already contains most of the non-dynamic basis infrastructure:

- `openghg_inversions/basis/algorithms/_constrained.py` defines class-constrained
  partitioning, a `SplitStrategy`, a `PartitionStep`, axis-parallel splitting,
  inertial splitting, physical grid geometry, and split-stopping policies.
- `openghg_inversions/basis/algorithms/_contrast.py` defines observation-space
  split-contrast scores and acceptance policies. This is directly related to the
  tree-contrast formulation below.
- `openghg_inversions/basis/layout.py` defines `BasisPartition`, `BasisLayout`,
  and state metadata for grouped fixed layouts.
- `openghg_inversions/basis/operators.py` and
  `openghg_inversions/basis/basis_functions.py` construct and serialize fixed
  basis operators.
- `docs/plans/mask_constrained_basis.md` records land/sea, country, rectangle,
  fixed-outer-region, layering, geometry, and partition-strategy decisions.
- `docs/plans/state_vector_grouping.md` and
  `docs/plans/fixed_outer_regions_grouping.md` describe grouped state-vector
  metadata and inner/outer semantics.
- `docs/plans/ogi_048_basis_algorithm_options.md` records real-footprint
  comparisons and a held-out projected-flux compression score.

These components produce and describe fixed layouts. They do not yet represent
a partition as dynamic sampler state, retain all fine-grid contributions needed
to rebuild \(H_P\), or sample over partitions.

The existing RHIME model builders also assume a fixed state dimension. Dynamic
partition inference should initially be a separate experimental subsystem and
should not be inserted into the established basis wrappers or
`fixedbasisMCMC` path.

## Prototype source inventory

The following paths are historical provenance, not intended dependencies. Phase
0 must preserve the useful behavior in repository-owned code and tests.

### Clean reference branch

Branch `codex/basis-prototype-examples`, commit `b6ce565`, contains:

- `openghg_inversions/basis/algorithms/_experimental_dyadic.py`;
- `tests/test_experimental_dyadic_basis.py`.

It provides three executable examples:

1. a precomputed dyadic weight array plus threshold bisection;
2. a small split/merge annealing scaffold;
3. a Numba implementation of bisection.

The tests pass and make this the best structural starting point. It should not
be merged wholesale. Its energy is a toy objective, its fixed-temperature loop
is a local stochastic search rather than a full annealing schedule, its merge
enumeration is quadratic, and it does not account for asymmetric proposal
probabilities. Most importantly, it pre-sums an already formed two-dimensional
weight field, whereas the scientific prototype aggregates observation
sensitivity before constructing its score.

### Raw multiscale and SA history

`~/Documents/basis_functions/basis_fn_ipython_hist_14aug.py` contains:

- `make_multi`, which repeatedly sums adjacent values to build dyadic spatial
  levels;
- `make_weights`, which pads the spatial grid, constructs multiscale observation
  sensitivities, and derives candidate-tile weights;
- tile lookup, validity, split, merge, DoFS-change, and random-move helpers;
- the dense active-tile indicator used by the original local search.

The important scientific operation is

\[
h_v = \sum_{i \in v} G_{:,i},
\]

followed by a score based on \(h_v\), \(R\), and tile scale. It is generally
incorrect to first construct fine-cell scalar weights \(w_i\) and then use
\(\sum_{i\in v} w_i\), because

\[
\left(\sum_{i\in v} G_{:,i}\right)^2
\ne
\sum_{i\in v} G_{:,i}^2.
\]

The recorded call path appears to pass a transformed inverse-error array to
`make_weights`; that convention must be resolved against the intended formula
rather than copied mechanically.

### Raw bisection, Numba, and optimization history

`~/Documents/basis_functions/basis_fn_ipython_hist_17aug.py` contains repeated
versions of:

- `bucket_basis`, a non-recursive threshold/bisection partitioner;
- `bucket_basis_opt`, a threshold search;
- `simulated_annealing`, with later variants superseding earlier cells;
- a Numba bisection kernel near the final useful block.

The final Numba kernel has previously shown output parity with the Python
version and a large warm-call speedup on a favorable synthetic full-split case.
That is evidence that the kernel is worth preserving, not a general benchmark.
The masked optimizer cells after it contain overwritten definitions and known
errors and should not be copied as authoritative code.

### Alternative partition steps

`~/Documents/inversions/src/inversions/basis_algorithms.py` contains the cleaned
prototype lineage for:

- a small `NodeListPriorityQueue`;
- greedy partitioning;
- axis-parallel split steps;
- inertial split steps.

The repository version in `_constrained.py` is now the preferred implementation
of those ideas. The raw file remains useful for provenance but does not need to
be duplicated in the dyadic package.

### Research notes and transcript

- `~/Documents/basis_functions/basis_fns.org` contains contemporary notes about
  the experiments.
- `~/Dropbox/acrg_library/Bocquet_Bayesian design of control space for
  optimal.pdf` is the local paper copy that inspired the prototype's DoFS and
  multiscale-representation calculations. It is Bocquet, Wu, and Chevallier
  (2011), Part I, cited publicly by DOI below. The most relevant locations are
  Section 2.2.2 for the multiscale Jacobian, Section 4.1.2 and Equations (38)-(39)
  for DFS by representation and tile, and Section 6 for tiling versus qtree
  storage and optimization.
- `~/Downloads/ChatGPT-Bayesian_model_selection.md` records the discussion that
  motivated fixed indicators, product-space inference, collapsed alternatives,
  and tree contrasts. This document supersedes that transcript as the project
  design reference; claims from the transcript are not treated as references.

## Phase 0: collect a public, tested reference implementation

This phase is required before either the SA demo or MCMC work. Its purpose is to
stop depending on private, stateful histories while retaining a traceable route
back to their scientific intent.

### Proposed location

Use a provisional repository package such as
`openghg_inversions/basis/experimental/dyadic/`, without re-exporting it from
the public `openghg_inversions.basis` namespace. Add a runnable synthetic demo
under `examples/basis/dyadic_sa_demo.py`. A package is preferable to putting all
logic in an example because the reference implementation needs focused tests
and will later supply the partition-inference subsystem.

The package name and APIs are provisional. Each module should state the source
prototype and the semantic differences from it.

### Required code

1. **Tree and tile representation**
   - immutable tile bounds and level/depth;
   - parent and child relations;
   - canonical tile identifiers independent of dense-array offsets;
   - valid active-leaf partitions and ancestry checks;
   - conversion between a partition, a dense label map, and active tile IDs.
2. **Multiscale sensitivity builder**
   - under the RHIME multiplicative-coefficient convention, construct candidate
     tile columns by summing fine-grid observation contributions first;
   - support only the block selected for multiscale aggregation;
   - preserve a clear observation, tile, and optional source layout;
   - document memory cost and padding behavior;
   - expose weight/score construction separately from column aggregation.
3. **Reference partition algorithms**
   - cleaned Python threshold bisection;
   - parity-tested Numba bisection as an optional acceleration;
   - adapters to current `PartitionStep` implementations where their partition
     representation is compatible.
4. **Local moves**
   - enumerate valid split and merge candidates;
   - apply and reverse a move without mutating the input state;
   - return forward and reverse log proposal probabilities;
   - support a paired split-and-merge fixed-count proposal;
   - use the standard-library `heapq` pattern through a small priority-queue
     wrapper where priority-based greedy initialization is needed.
5. **SA runner**
   - explicit initial state, objective, temperature schedule, and random seed;
   - correct handling of asymmetric proposals when configured as a sampler;
   - optimizer mode that clearly reports that it is not posterior inference;
   - diagnostics for objective, accepted move, tile count, and best state;
   - no unbounded history retention by default.
6. **Synthetic tests and fixtures**
   - no private data loading;
   - tile-sum identities against direct fine-grid sums;
   - valid-partition invariants after every move;
   - split/merge round trips;
   - Python/Numba parity;
   - deterministic SA smoke test;
   - exact forward-model equality between gathered multiscale columns and a
     direct aggregation of the fine-grid matrix.

### Provisional migration map

The target names are deliberately descriptive rather than API commitments.

| Historical source | Behavior to preserve | Proposed repository destination | Treatment |
| --- | --- | --- | --- |
| `hist_14aug.make_multi` and `np_sum_adj` | dyadic adjacent aggregation | `experimental/dyadic/multiscale.py` | re-express with explicit axes, shapes, and padding |
| `hist_14aug.make_weights` | sum observation contributions, then score candidate tiles | `experimental/dyadic/multiscale.py` and `scores.py` | split aggregation from scoring; resolve the inverse-error convention |
| `hist_14aug.make_dyadic_graph`, `get_shape`, `get_ell`, `get_slice` | node relationships and grid slices | `experimental/dyadic/tree.py` | replace implicit array offsets with stable node IDs |
| `hist_14aug.get_tiles`, `get_basis_array`, `basis_valid` | active-tile decoding, labels, and validity | `experimental/dyadic/state.py` | make state immutable and validation explicit |
| `hist_14aug.get_split`, `get_merge`, `get_move`, `apply_move` | local partition transitions | `experimental/dyadic/proposals.py` | return new state plus forward/reverse log proposal probabilities |
| `hist_14aug.dof_change` | cheap local score delta | `experimental/dyadic/scores.py` | test against full recomputation and current `_contrast.py` |
| `hist_17aug.bucket_basis` | non-recursive threshold bisection | `experimental/dyadic/bisection.py` | retain one cleaned reference definition |
| final useful `hist_17aug` Numba kernel | accelerated bisection | `experimental/dyadic/_numba.py` | add only after Python semantics and parity tests are fixed |
| `hist_17aug.simulated_annealing` and `ApplyStepCondition` | local stochastic partition search | `experimental/dyadic/annealing.py` | rewrite around explicit objective, schedule, RNG, and diagnostics |
| branch `b6ce565` | clean data structures and smoke tests | all modules above | use as scaffold, not as the scientific score definition |
| merged `_constrained.py` | greedy queue, mask constraints, axis and inertial steps | existing module | reuse or adapt; do not copy prototype implementations |
| merged `_contrast.py` | mass-preserving observation contrast and design scores | existing module | reuse as the canonical split-contrast algebra |

The first Phase 0 PR should include this table in a package-level provenance
file with exact source line/commit references updated at extraction time. Once
the extracted tests cover the intended behavior, later design documents and
issues should link to repository symbols rather than the historical paths.

### Explicit exclusions

Do not initially copy:

- notebook data-loading code or absolute paths;
- earlier overwritten SA definitions;
- the broken masked optimizer experiments;
- the brute-force SciPy threshold wrapper unless a benchmark demonstrates a
  need for it;
- duplicate inertial or constrained split implementations;
- an API coupled to `fp_all`;
- a PyMC step method before the standalone transition kernel is validated.

### Phase 0 deliverables

- one importable experimental package in this repository;
- one synthetic SA example with plots or tabular diagnostics;
- focused tests runnable without atmospheric data;
- a provenance table mapping each retained function to its prototype source;
- a short benchmark separating compile time, warm Numba time, Python time,
  memory use, and problem shape.

## Dyadic representation and validity

### Established prototype behavior

The original state is a dense indicator over all candidate dyadic rectangles.
An active indicator denotes a current basis tile. A valid partition covers each
included grid cell exactly once and contains no active ancestor/descendant pair.
Splits replace one active tile with children. Merges replace a complete sibling
set with their parent.

The indicator is useful for cheap lookup and serialization, but it should not be
the sole source of validity. An immutable partition state with an active-leaf set
makes invariants explicit, while a dense indicator can be derived for vectorized
calculations or trace output.

### Split orientation in two dimensions

There is an important complication hidden by the word "tree". If every
rectangle may split along either the x or y axis, there is not one unique binary
tree unless the orientation is part of the node state. Different split orders
can also create the same final rectangles, which can unintentionally multiply
prior mass.

The first demo should choose one of:

- a canonical orientation rule, such as alternating axes or using a fixed tree;
- an explicit node decision in `{stop, split_x, split_y}` with a canonical
  representation of equivalent partitions;
- a quadtree in which a split has a fixed set of children.

**Proposed:** begin with a canonical binary tree for exact inference. Keep the
more flexible x/y choice in optimizer experiments until duplicate
representations and prior probabilities are defined.

## Objectives and what they mean

### Degrees of freedom for signal

For a suitable linear-Gaussian model, the degrees of freedom for signal (DFS)
can be written as the trace of the averaging-kernel operator. One equivalent
form is

\[
\operatorname{DFS}(P)
= \operatorname{tr}\left[
B_P H_P^T (H_P B_P H_P^T + R)^{-1} H_P
\right].
\]

This is a principled information criterion only when the likelihood, prior,
covariance definitions, and aggregation-error treatment match its derivation.
The prototype calculations were specifically inspired by Bocquet, Wu, and
Chevallier's multiscale control-space design, rather than only by this generic
fixed-representation formula.

In their notation, Equation (38) rewrites the DFS score for an admissible
representation \(\omega\) as

\[
J_\omega
= \operatorname{tr}\left[
\Pi_\omega B H^T(R+HBH^T)^{-1}H
\right],
\]

where \(\Pi_\omega\) is the representation projector. The denominator uses the
fine-grid innovation covariance because their scale-covariant aggregation-error
construction makes \(R_\omega+H_\omega B_\omega H_\omega^T=R+HBH^T\). Equation
(39) then associates a normalized DFS contribution with each candidate tile.
This gives the precomputed tile scores used by the prototype a clear source.

That formulation is not automatically identical to recomputing the generic DFS
formula independently for every gathered \(H_P\). Equivalence depends on using
the same prolongation, prior covariance, normalization, and aggregation-error
model as the paper. Phase 0 must write down which of these assumptions the
prototype retained before its incremental `dof_change` is treated as an exact
score.

The exact projected posterior should also be distinguished from its native-space
lift. Equations (29)--(31) show that the posterior in any representation is the
exact restriction of the native posterior. If the projected posterior is lifted
with the **prior** conditional distribution for unresolved contrasts, the
resulting native-space distribution is an approximation. By the KL chain rule,
maximizing projected Bayesian information gain minimizes the forward KL
divergence from the full posterior to this lifted approximation. The
data-dependent Equation (45) is the corresponding mean-only objective: it
maximizes the squared \(B^{-1}\)-norm of the captured posterior-mean update.
The full derivation and equation map are recorded in
`docs/reports/rhime_bocquet_reduced_gaussian.md`.

Under Equations (50)--(51), the data-dependent objective has additive tile
scores and an exact fixed-count dynamic program over a recursive tree. The
rank-one form appears in Equation (64). This additivity holds for diagonal or
appropriately whitened prior covariance. Dense correlated \(B\) with retained
geographic region semantics is exact but globally coupled: the reduced
covariance is dense and a local split can alter all regional scores. That case
requires full-objective stochastic search or more specialized matrix updates,
not the scalar-node DP.

This DP statement does not apply to every criterion in Section 4. Equation 37's
aggregation-aware Fisher criterion is nonlinear through \(R_\omega^{-1}\), and
the full Bayesian KL in Equations (40) and (42) contains a log determinant.
Neither is a scalar-node additive objective. The current DP is exact only for a
declared additive trace objective within its one canonical binary-tree
dictionary.

The paper describes coarse-graining its multiscale Jacobian by averaging. For a
RHIME multiplicative region coefficient, the corresponding observation column
is normally the sum of fine-cell footprint-times-flux contributions in that
region. These are compatible only after the coefficient and prolongation
normalizations are specified. The public reference implementation should test
the RHIME forward-model identity directly instead of copying either "sum" or
"average" without that convention.

The same paper is also the source for the multiscale memory tradeoff: for a
dyadic 2D-plus-time tiling dictionary it reports up to eight times the finest-grid
Jacobian storage, while its more restrictive quaternary-tree spatial dictionary
requires at most eight-thirds. Those factors apply to that hierarchy and should
not be presented as universal costs for every dyadic implementation.

For lognormal or other non-Gaussian priors, a Gaussian DFS is not the true
partition posterior and should not be presented as one. It remains useful as:

- an SA objective for generating efficient initial partitions;
- an informed split/merge proposal score;
- a pseudo-prior calibration statistic;
- a first-stage delayed-acceptance screen;
- a diagnostic for comparison with earlier experiments.

The original SA used a DoFS-like change plus a one-sided penalty when the region
count exceeded a target. It generally operated as fixed-temperature local
search unless the caller changed temperature. It did not include a Hastings
ratio, so it was not an exact fixed-temperature MCMC kernel. Those properties
are acceptable for an optimizer if they are explicit.

### Split-contrast score

The repository's `_contrast.py` expresses a candidate split as one new
observation-space contrast. If parent \(G\) has children \(A\) and \(B\), with
prior masses \(\mu_A\), \(\mu_B\), and \(\mu_G=\mu_A+\mu_B\), define

\[
f_\delta
= \frac{\mu_B}{\mu_G}h_A
- \frac{\mu_A}{\mu_G}h_B.
\]

For contrast prior scale \(\tau\) and design covariance \(S\),

\[
\lambda = \tau^2 f_\delta^T S^{-1} f_\delta,
\qquad
\operatorname{DFS}_{\mathrm{contrast}} = \frac{\lambda}{1+\lambda},
\qquad
\operatorname{EIG}_{\mathrm{contrast}} = \frac{1}{2}\log(1+\lambda).
\]

These formulas explain why a precomputed multiscale sensitivity array makes
local split scoring cheap: the new direction can be constructed from a few
child-column lookups. They describe the conditional scalar information in the
new contrast. They equal the global change between the complete parent and
split models only when \(S\) is the appropriate baseline marginal or conditional
covariance and the added contrast is independent under the stated Gaussian
prior. Calibration still requires a scientifically meaningful contrast prior
and covariance. Without those conditions, the repository's `delta_dfs` and
`delta_eig` fields are ranking proxies rather than global partition deltas.

Under those Gaussian conditions, with an independent zero-mean contrast prior
and residual \(r\) centered under the baseline model, the corresponding local
log Bayes factor also contains a residual-dependent term:

\[
\log BF_{1,0}
= -\frac{1}{2}\log(1+\lambda)
+ \frac{1}{2}
  \frac{\tau^2(f_\delta^T S^{-1}r)^2}{1+\lambda}.
\]

The first term penalizes the additional contrast and the second rewards fit to
the realized observations. This makes the conceptual difference precise:
\(\lambda\), DFS, and expected information describe what the observing system
could resolve under the specified Gaussian design, while a marginal likelihood
or Bayes factor also responds to the observed data.

### Predictive and compression scores

The OGI-048 experiments compare held-out full-grid modeled observations with
modeled observations obtained after replacing the prior flux by region means.
This is a useful compression score. It does not use observed mole fractions and
is not posterior predictive accuracy.

Directly comparing prior modeled observations with all multiplicative basis
coefficients set to one is uninformative: any complete basis preserves the sum
exactly, so the error is zero. A region-mean projected-flux comparison avoids
that identity and measures information lost by compression.

Additional useful scores are:

- temporal or site holdout projected-flux NRMSE for basis construction;
- observed-\(y\) posterior predictive RMSE or log predictive density after an
  inversion;
- an outer holdout for comparing tuning rules or complexity priors;
- positive-versus-negative flux preservation where signed fluxes matter.

### When holdout data are required

Bayesian inference over \(P\) and \(x_P\) may use all observations once through
the joint posterior. This is not double use of the data, and holdout is not
required for the posterior to be mathematically valid.

Holdout or cross-fitting becomes important when:

- observations are used to select a single partition \(P^*\), after which an
  ordinary fixed-partition posterior is reported without accounting for
  selection uncertainty;
- a complexity penalty, proposal heuristic, or approximation is tuned and then
  assessed on the same observations;
- predictive performance is the evaluation target.

If holdout data tune the partition and are also used to report performance, an
outer holdout or nested scheme is needed. Conversely, DoFS and projected-prior
scores can depend on the observation operator and error model without using the
realized observed values \(y\); this differs from fitting the partition directly
to residuals.

Equation (45) occupies an intermediate decision-theoretic case. If a
predeclared rule chooses \(P(y)\) as a posterior-compression action, the exact
posterior of the selected linear summary remains coherent conditional on the
observed data. However, the selected geography is adaptive, its training score
is optimistic as an evaluation statistic, and ordinary intervals do not become
selection-adjusted for confirmatory claims. Bocquet et al. describe the risk of
using the same observations to construct and invert on an adaptive grid as a
possible "inversion crime." For the POC, report the training Equation 45 score,
evaluate prediction or compression on held-out sites or times, and compare with
the data-independent DFS and Fisher selections. Use cross-fitting or an outer
holdout if tuning rules are themselves compared.

For a fixed candidate, the expected Equation 45 score is one-half its DFS, but
after selection

\[
\mathbb E[J_{\widehat P}(y)]
=\mathbb E[\max_P J_P(y)]
\geq\max_P\mathbb E[J_P(y)].
\]

The selected training value is therefore optimistic. Cross-fitting is useful
for evaluating the adaptive pipeline and its stability, but fold-specific
partitions do not automatically combine into one calibrated posterior.

## Joint Bayesian target

The desired non-Gaussian target is

\[
p(P,x_P,\theta\mid y)
\propto
p(y\mid x_P,P,\theta)
p(x_P\mid P,\theta)
p(P\mid\theta)
p(\theta),
\]

where \(\theta\) includes observation-error, prior, boundary-condition, and
other model parameters. A prior on \(P\) replaces the SA region-count penalty.
Possibilities include a leaf-count prior or a depth-dependent split prior.

The inference method must preserve this target. Scores used only for proposal
selection do not need to equal the target, provided their forward and reverse
proposal probabilities appear in the Metropolis-Hastings ratio. Approximate
targets require a correction step if exact posterior inference is claimed.

### Inference choices are composable

"Collapsed," "fixed-count," "product-space," and "RJMCMC" answer different
questions. Collapsing describes whether continuous parameters are integrated
out. Fixed or variable count describes the partition support. Product-space and
RJMCMC describe how parameters across models are represented and moved.

| Choice | Strength | Main limitation | Recommended use |
| --- | --- | --- | --- |
| SA optimization | cheap use of local scores; good initialization | not posterior inference | first scientific demo and chain initialization |
| fixed-\(K\) sampling | fixed continuous dimension and controlled comparison | does not infer uncertainty in \(K\) | first exact joint sampler for a declared partition-dependent model |
| Gaussian collapsing | exact fast structure scores when conjugate | not directly available for lognormal coefficients | tiny oracle, Gaussian benchmark, or conditional block |
| product space | globally fixed dimension and ordinary fixed-shape traces | memory, pseudo-priors, and inactive-variable geometry | preferred variable-\(K\) prototype after fixed-\(K\) |
| RJMCMC | native packed variable-dimensional state | mappings, Jacobians, proposals, and ragged output | defer until product-space limitations are measured |

For example, one can run collapsed fixed-\(K\) partition sampling, collapsed
variable-\(K\) product-space sampling, or non-collapsed RJMCMC. The terms are
not competing names for one decision.

## Collapsed partition inference

### Partition-dependent linear-Gaussian case

With a Gaussian likelihood and conditionally Gaussian prior, coefficients can
be integrated out:

\[
p(P\mid y) \propto p(y\mid P)p(P).
\]

This posterior is nontrivial only when \(P\) genuinely changes the statistical
model. In the exact Bocquet construction, the projected prior and aggregation
residual are both induced from one native Gaussian model and

\[
p(y\mid P)=\mathcal N(y;H\mu,R+HBH^T)
\]

is partition-invariant. The formulas below therefore describe a separate
reduced-model or non-Gaussian model-selection track, not posterior inference over
exact Bocquet representations.

This is not "basis selection only" in the sense of discarding coefficient
uncertainty. The joint posterior factorizes as

\[
p(P,x_P\mid y)=p(P\mid y)p(x_P\mid P,y).
\]

One may sample \(P\) from its marginal posterior and then draw \(x_P\) from its
conditional posterior. Model-averaged quantities follow from both draws.

Using the same \(y\) in these two conditional operations is not double counting;
it is ordinary probability factorization. The problematic workflow is selecting
one \(P^*\) from \(y\), treating it as fixed, and reporting a conditional
posterior as though no selection occurred.

### Non-Gaussian case

For lognormal \(x_P\), the coefficient integral is generally unavailable.
Laplace integration can provide an approximate partition score, initializer, or
proposal. It is not exact merely because the subsequent coefficient update uses
NUTS. Exact alternatives include:

- delayed acceptance with a final exact likelihood/prior correction;
- an unbiased pseudo-marginal likelihood estimator;
- direct joint sampling of partition and continuous state.

The first two need separate numerical validation. In particular, plugging a
biased approximate marginal likelihood directly into MH generally changes the
stationary distribution.

## Fixed-count joint inference

Keep \(K(P)=K\) constant by proposing a split in one part of the partition and a
merge elsewhere. There are two materially different fixed-count experiments.

### Collapsed Gaussian fixed-count sampler

**Proposed first exact partition sampler for a declared partition-dependent
benchmark.** Suppose

\[
x_P\mid P \sim N(m_P,B_P),
\qquad
y\mid x_P,P \sim N(b+H_Px_P,R).
\]

For this benchmark, \(b\), \(m_P\), \(B_P\), and \(R\) are fixed model inputs.
Estimating partition-specific nuisance parameters and then substituting their
point estimates into the expression below would be an empirical-Bayes
approximation, not the same collapsed posterior. Gaussian nuisance terms can be
included by enlarging the Gaussian state and collapsing them jointly; unknown
scales, truncations, and other non-Gaussian terms require separate treatment.

Then

\[
y\mid P \sim N(b+H_Pm_P,S_P),
\qquad
S_P=R+H_PB_PH_P^T,
\]

with residual \(r_P=y-b-H_Pm_P\). The exact log marginal likelihood is

\[
\log p(y\mid P)
=-\frac{1}{2}\left[
n\log(2\pi)+\log|S_P|+r_P^TS_P^{-1}r_P
\right].
\]

It should be evaluated by a Cholesky factor \(S_P=L_PL_P^T\), using
\(2\sum_i\log(L_P)_{ii}\) for the log determinant and a triangular solve for
the quadratic form. For the benchmark, require symmetric positive-definite
\(R\) and proper \(B_P\), and report a diagnostic failure rather than silently
adding arbitrary jitter.

The fixed-count target is

\[
\pi_K(P)
\propto
p(y\mid P)p(P\mid K),
\qquad P\in\mathcal P_K.
\]

This sampler has no continuous state during the partition update. It therefore
needs neither NUTS nor a split/merge coefficient transform. Conditional Gaussian
coefficient draws can be generated afterward when model-averaged coefficient or
field summaries are required.

The exact conditional moments are

\[
m_{P\mid y}
=m_P+B_PH_P^TS_P^{-1}r_P,
\qquad
B_{P\mid y}
=B_P-B_PH_P^TS_P^{-1}H_PB_P.
\]

This is the smallest statistically defensible route from the stochastic local
search to posterior partition inference. It also supplies an exact oracle for
testing later non-Gaussian methods.

### Non-Gaussian fixed-count sampler

For the intended lognormal model, retain a continuous vector of fixed length
\(K\), gather the active multiscale columns, and update the continuous parameters
conditional on the current partition.

This is ordinary Metropolis-Hastings-within-Gibbs, not RJMCMC, because the
continuous state remains in \(\mathbb{R}^K\). It still needs:

- a reversible pairing between old and proposed coefficients, or a proposal for
  the proposed partition's coefficients;
- forward and reverse probabilities for selecting split and merge candidates;
- exact likelihood and prior evaluation;
- tests that the gathered \(H_P\) is in the same order as \(x_P\).

This experiment answers whether local partition moves and conditional NUTS mix
adequately before variable dimension and pseudo-priors are introduced.

### NUTS feasibility and deferred work

Fixed dimension is sufficient for a valid conditional NUTS trajectory, but it
does not give packed coefficient positions stable spatial meaning. PyMC's native
compound sampler learns one step size and one metric per NUTS instance and
chain. During warmup, all visited partitions feed those same adapters without a
partition label. The resulting metric is either dominated by the partitions
visited during warmup or is a generic compromise across their different
conditional geometries.

Once adaptation is frozen, a positive-definite but poorly matched metric affects
efficiency rather than the invariant distribution. The practical risks are
divergences, excessive tree depth, very small global step size, and failure to
move after reaching a partition not represented during warmup.

Any later conditional-NUTS experiment should therefore use:

- canonical node IDs as region identity, with packed indices derived only for
  computation;
- exact agreement between coefficient packing and \(H_P\) column order;
- prior-whitened log-coefficients;
- identity or diagonal geometry before dense adaptation is considered;
- structure moves active throughout discarded warmup;
- several distinct valid initial partitions;
- adaptation frozen for all retained draws;
- diagnostics stratified by structural summaries rather than only by packed
  coefficient index.

PyMC has no built-in structure-indexed metric or step-size cache. A joint
partition move that also transports affected coefficients requires a custom
step or external orchestrator; the default compound assignment will not design
that transport. External NUTS backends are not available when a separate
discrete step is present, so this route would initially use native PyMC NUTS.

This work is intentionally deferred. Elliptical slice sampling, pCN, or a simple
fixed diagonal HMC kernel in prior-whitened coordinates are useful exact
comparators for the lognormal model because they do not require one learned
posterior metric to remain meaningful across partitions.

## Hackathon proof of concept

### Objective

Keep two claims separate:

1. Under the exact Bocquet model, demonstrate adaptive posterior compression by
   optimizing Equation 45 and comparing it with DFS and Fisher selections.
2. Under a transparent but genuinely partition-dependent linear-Gaussian toy
   model, demonstrate that the existing DoFS-guided stochastic local search
   machinery can become an exact posterior partition sampler.

For the second claim, DoFS guides computation but does not replace the
likelihood, coefficient prior, or partition prior. It must not be presented as
sampling \(P\) under the exact Bocquet projection model.

The minimum successful result is fixed \(K\). Variable \(K\) is a stretch goal
that is unusually accessible in the collapsed Gaussian case because there is no
variable-dimensional coefficient vector left in the sampled state.

### Metropolized local search

Let \(\mathcal N_K(P)\) be the unique fixed-count neighboring partitions
obtained by one compatible split and one compatible merge. Enumerate neighbors
as canonical partitions, not as move histories, because more than one history
may otherwise produce the same proposed state.

If the implementation first enumerates move paths rather than unique states,
the proposal probability for \(P'\) is the sum over every path that produces
\(P'\). Deduplicating states without summing their path probabilities would
change the proposal law.

For the linear-Gaussian model, define the full DoFS diagnostic

\[
D(P)=\operatorname{tr}\left[
B_PH_P^TS_P^{-1}H_P
\right].
\]

The neighbor score \(d(P,P')\) can be \(D(P')\), or equivalently
\(D(P')-D(P)\) within a fixed current-state normalization. A prototype local
split-contrast calculation may replace this only after it is shown to rank or
reproduce the direct full-matrix change under the benchmark assumptions. Define
the proposal with a uniform floor:

\[
q_\beta(P'\mid P)
= \frac{\varepsilon}{|\mathcal N_K(P)|}
+ (1-\varepsilon)
  \frac{\exp\{\beta d(P,P')\}}
       {\sum_{Q\in\mathcal N_K(P)}\exp\{\beta d(P,Q)\}}.
\]

Use stable log-sum-exp evaluation. The floor protects irreducibility and avoids
turning a heuristic zero into an impossible posterior transition. The exact MH
acceptance probability is

\[
\alpha(P,P')
= \min\left\{
1,
\frac{p(y\mid P')p(P'\mid K)q_\beta(P\mid P')}
     {p(y\mid P )p(P \mid K)q_\beta(P'\mid P)}
\right\}.
\]

This is a one-step Metropolized version of the local search. The DoFS score and
proposal temperature \(\beta\) affect efficiency but not the target. Choose or
tune \(\beta\) during pilot runs, then freeze it before retaining production
draws. The reverse normalizer must be recomputed on \(\mathcal N_K(P')\).

Include an explicit lazy/self-loop probability so aperiodicity does not depend
on a rejection happening somewhere. For a fixed finite binary tree, valid
remove-maximal-split/add-feasible-split exchanges should connect the fixed-count
frontiers; verify this on every enumerated toy support because mixed split
arities, hard masks, per-group count constraints, or score thresholds can break
that argument.

Do not use a long stochastic-search trajectory as one proposal in the first
demo: its total forward and reverse path probabilities are harder to compute.
The existing search remains valuable as an initializer. Exact delayed
acceptance with a DoFS first stage is another future option, but it is not needed
for the minimum demonstration.

### Partition prior

For the toy fixed-count target, begin with either:

- a uniform prior over unique valid canonical \(K\)-leaf partitions; or
- a stated depth-dependent tree prior conditioned on \(K\).

A prior depending only on \(K\) is constant in the fixed-count experiment. A
uniform distribution over split histories is not generally uniform over
geometric partitions. The implementation must test that the fixed-count neighbor
graph is connected for every demonstrated \(K\).

The coefficient prior across partitions is a separate and consequential model
choice. Using \(B_P=\tau_x^2I_K\) and a fixed prior mean is acceptable for the
toy, but independent equal-variance region multipliers at every resolution do
not generally induce the same fine-grid prior. Marginal likelihood, especially
when \(K\) varies, can be dominated by this normalization. The demo must label
the convention as a benchmark prior and compare its induced fine-grid mean and
covariance across several partitions. A coherent later alternative is a
tree-contrast prior or an explicitly induced fine-grid prior with documented
mass/depth scaling.

This benchmark also differs from the scale-consistent construction in Bocquet,
Wu, and Chevallier. In that construction the covariance belongs first to the
fine representation and transforms with the representation projection (using
\(P\) for that projection by an abuse of notation):

\[
B_P=PBP^T.
\]

Using the same numerical \(B_P\) for every partition breaks that transformation
rule. Phase 0 should therefore accept covariance construction as an explicit
partition-dependent operation even if the first proof of concept supplies
\(\tau_x^2I_K\). A Bocquet-faithful projected covariance can then replace the
benchmark without changing partition state or search control flow.

For the variable-count stretch goal, add an explicit \(p(K)\) or an unconditioned
proper tree prior and use separate split and merge proposals with their complete
candidate-count and direction probabilities. Because \(x_P\) remains collapsed,
this is a discrete-space MH extension rather than RJMCMC.

If fixed-count and variable-count kernels are mixed, use fixed mixture weights
or include any state-dependent mixture normalization in the transition
probability. State-dependent selection among individually invariant kernels is
not automatically invariant.

### Synthetic experiment

Use a small canonical binary tree and completely synthetic arrays first:

1. Construct smooth footprint-like rows over a small grid and a base flux field.
2. Select a hidden valid partition and Gaussian region coefficients.
3. Form \(H_P\) by summing fine-cell footprint-times-flux contributions within
   each active region.
4. Simulate \(y=H_Px_P+\epsilon\) with known \(R\).
5. Use the same declared \(B_P\), \(R\), and aggregation convention in the DoFS
   score and collapsed target.

For any later OGI example, apply observation filtering before constructing the
fine-grid or multiscale \(H\) and before computing DoFS proposal scores. A basis
proposal influenced by observations that are later excluded is a different
data-dependent procedure and can leak filtered information into the design.

For a very small tree, enumerate all valid fixed-count partitions and normalize
their exact posterior probabilities. This is the test oracle. The MCMC
frequencies must agree with it within Monte Carlo error before scaling up.

Construct the oracle independently of the sampler transition helpers. Build the
full transition matrix and verify row normalization, pairwise detailed balance,
\(\pi^TT=\pi^T\), graph connectivity, and agreement between its stationary
distribution and the independently normalized target. Include unequal neighbor
counts, duplicate move paths, extreme DoFS scores, nonzero prior means, and leaf
ordering permutations in focused tests.

Useful demo outputs are:

- initial greedy partition, best local-search partition, posterior MAP
  partition, and posterior cell-level split probabilities;
- DoFS and collapsed log-posterior traces shown separately;
- empirical versus exactly enumerated partition probabilities;
- uniform-neighbor versus DoFS-informed proposal acceptance and partition ESS;
- optional posterior \(K\) for the variable-count stretch goal.

An OGI footprint/flux case is a follow-up only after the synthetic oracle works.
The real-data route adds loading, filtering, covariance, and scientific-prior
questions without strengthening the first correctness demonstration.

### Initial states and poor local mixing

The empirical observation that bucket, quadtree, or greedy construction gives a
much better start than growth from one root is expected. Local split/merge moves
must traverse a long, narrow graph from the root and can become trapped in a
single high-score basin. An annealing schedule changes acceptance but not that
graph.

Use a portfolio of initial partitions drawn from the exact sampler's support:

- one strong greedy or local-search result;
- randomized greedy constructions;
- random valid growth to \(K\);
- compatible variants of existing basis algorithms.

Starting from an optimized, even data-informed, state does not change the exact
stationary distribution after convergence. Starting every chain from the same
state does weaken multimodality diagnostics.

If paired moves fail, add proposal kernels rather than tuning NUTS:

- global split/merge pairing;
- fixed-leaf-count subtree prune-and-regrow;
- boundary relocation or tree rotations;
- multiple-try selection with exact correction;
- frozen independence proposals generated by randomized local searches;
- tempering or SMC if separated modes remain.

Moves do not need to be physically interpretable to be valid. Physical
preferences belong in \(p(P\mid K)\); proposal moves belong in \(q\). A bit-level
move is acceptable as a secondary kernel if it has a clear action on canonical
partitions, computable reverse probability, and does not destroy connectivity.

### Work that transfers to later methods

| Work item | Product space | RJMCMC | Long-term value |
| --- | --- | --- | --- |
| Canonical tree, node IDs, masks, and exact-cover validation | direct reuse | direct reuse | high |
| Multiscale \(H\), tile lookup, and label conversion | direct reuse | direct reuse | high |
| Partition prior and canonical probability | direct reuse | direct reuse | high |
| Neighbor enumeration and exact proposal probabilities | reuse for indicator moves | reuse for split/merge moves | high |
| DoFS-informed proposal with uniform floor | reuse for structure proposals | reuse for structure proposals | high |
| Greedy/local-search initialization portfolio | reuse for chain initialization | reuse for chain initialization | high |
| Tiny exhaustive enumerator and detailed-balance tests | oracle for product-space marginals | oracle for RJ frequencies | high |
| Collapsed Gaussian target | benchmark and possible conditional block | benchmark and proposal calibration | medium-high |
| Fixed-count paired move | reuse directly at fixed count | split and merge pieces remain useful | medium-high |
| Synthetic generator and posterior diagnostics | regression and calibration fixtures | regression and calibration fixtures | high |
| Packed-coordinate NUTS adaptation experiments | little direct reuse | little direct reuse | low; defer |
| Non-Gaussian coefficient transport | useful only for active-only product variants | direct RJ relevance | medium; defer until needed |

This scope deliberately spends almost no time on the least transferable fixed-K
work. A failed local partition kernel is still informative: the exact oracle can
show whether failure is caused by poor traversal rather than an incorrect target,
and the same state, prior, and proposal tests remain prerequisites for more
elaborate methods.

### One-day scope and stop conditions

The minimum coherent implementation has four work packets:

1. Extract only canonical tree/state and multiscale-column code needed by the
   toy; defer Numba, masks, real data, and general public APIs.
2. Implement the collapsed Gaussian target and exhaustive tiny-tree oracle.
3. Implement uniform and DoFS-informed fixed-count MH over unique neighbors.
4. Produce deterministic tests and the comparison plots listed above.

On a canonical 4-by-4 toy tree, this is plausibly one focused implementation
day: roughly one work block for state/enumeration, two for the Cholesky target
and DoFS score, one for proposals/MH, two for transition-matrix tests, and one
for the runnable comparison. This forecast assumes direct synthetic arrays and
the minimal extraction of existing dyadic helpers. Real footprints, Numba,
arbitrary split geometry, or PyMC integration would move it beyond the intended
hackathon scope.

The first three packets constitute the scientific result. Plot polish and
variable \(K\) are secondary. If time remains, variable count requires a tree
prior and split/merge proposal accounting, but not NUTS or RJ coefficient maps.

Stop or narrow the demo if:

- incremental and direct DoFS calculations disagree;
- MCMC frequencies fail the exact tiny-tree oracle;
- the fixed-count proposal graph is disconnected;
- neighbor enumeration is already too expensive at the toy scale;
- all chains remain in their initializer's basin under both uniform and informed
  proposals.

Even the last result is useful if exact enumeration confirms substantial mass in
multiple basins: it gives direct evidence that local-move geometry, rather than
NUTS tuning, is the next problem to solve.

## Product-space inference with pseudo-priors

### Construction

Let \(z\) contain coordinates for every potential tree node or split contrast.
For partition \(P\), let \(z_A\) denote active coordinates and \(z_I\) inactive
coordinates. A product-space target has the form

\[
\widetilde\pi(P,z_A,z_I,\theta\mid y)
\propto
p(P)
p(y\mid P,z_A,\theta)
p(z_A\mid P,\theta)
q_P(z_I\mid z_A,\theta)
p(\theta),
\]

where \(q_P\) is a proper conditional pseudo-prior satisfying

\[
\int q_P(z_I\mid z_A,\theta)\,dz_I=1
\]

for every admissible \(P,z_A,\theta\). Integrating out the inactive coordinates
therefore gives the desired posterior over \((P,z_A,\theta)\). Proper
normalization, including constants that depend on \(P\), is essential: an
unknown partition-dependent constant in \(q_P\) would change posterior
partition probabilities.

The full state dimension is fixed. Inactive variables are not multiplied by
zero and then omitted from the probability model; their pseudo-prior densities
are part of the augmented target and strongly affect switching efficiency. The
pseudo-priors do not change the exact marginal posterior when normalized, but
poor choices can make transitions between partitions practically impossible.

This is the Carlin-Chib product-space construction. Dellaportas, Forster, and
Ntzoufras (2002), especially their discussion of Carlin-Chib and Metropolised
Carlin-Chib methods, is an appropriate comparison reference. The foundational
pseudo-prior reference is Carlin and Chib (1995).

### Finite-model Carlin-Chib form

For models \(m=1,\ldots,M\), let \(\beta_k\) be the parameter block associated
with model \(k\), and let \(\psi_k\) be the common pseudo-prior used whenever
that block is inactive. The augmented target is

\[
\widetilde\pi(m,\beta_{1:M}\mid y)
\propto
p(m)
p(y\mid m,\beta_m)
p(\beta_m\mid m)
\prod_{k\ne m}\psi_k(\beta_k).
\]

Ordinary Carlin-Chib refreshes inactive blocks and Gibbs-samples \(m\) from all
\(M\) conditional model weights. This becomes expensive when the model space is
large, as it is for tree partitions.

Dellaportas, Forster, and Ntzoufras instead propose one model
\(m'\sim j(m,m')\). Their hybrid algorithm updates the current block from its
within-model posterior, draws only the proposed block
\(\beta_{m'}\sim\psi_{m'}\), and accepts the model switch with their equation
(6):

\[
\alpha(m,m')
=
1\wedge
\frac{
p(y\mid m',\beta_{m'})
p(\beta_{m'}\mid m')
p(m')
\psi_m(\beta_m)
j(m',m)
}{
p(y\mid m,\beta_m)
p(\beta_m\mid m)
p(m)
\psi_{m'}(\beta_{m'})
j(m,m')
}.
\]

The important computational statement is that only the proposed block is
**sampled** from a pseudo-prior and only the current/proposed model pair is
considered. Both the proposed pseudo-prior density
\(\psi_{m'}(\beta_{m'})\) and the reverse density
\(\psi_m(\beta_m)\) still appear in the acceptance ratio, although the latter
can be retained with the current state. All unrelated pseudo-prior factors
cancel. Dellaportas et al. describe this update as an independence sampler and
contrast it with current-dependent reversible-jump transformations.

The direct tree analogue requires a common, factorized pseudo-prior, initially

\[
q_P(z_I)=\prod_{v\in I(P)}\psi_v(z_v),
\]

where each normalized \(\psi_v\) is independent of the partition in which
coordinate \(v\) is inactive. Unchanged factors then cancel in a local
split/merge ratio. If pseudo-priors depend jointly on the complete inactive set
or on \(P\) in some other way, the implementation must evaluate the full
\(q_{P'}/q_P\) ratio. Moreover, the original finite-model construction stores a
complete block for each model, whereas this design shares one global contrast
vector across partitions. Candidate-only local refreshes should therefore be
called **Dellaportas-inspired** unless they implement the complete-block update
under the stated common pseudo-prior assumptions.

For a fixed global coordinate vector and a partition-only proposal
\(P\rightarrow P'\), the corresponding acceptance probability is

\[
\alpha(P,P')
=
1\wedge
\frac{
G_{P'}(z_{A(P')},\theta)
q_{P'}(z_{I(P')}\mid z_{A(P')},\theta)
j(P',P)
}{
G_P(z_{A(P)},\theta)
q_P(z_{I(P)}\mid z_{A(P)},\theta)
j(P,P')
},
\]

where

\[
G_P(z_A,\theta)
=p(P)p(\theta)p(y\mid P,z_A,\theta)p(z_A\mid P,\theta).
\]

There is no proposal Jacobian when \(z\) is unchanged because this is ordinary
MH on one fixed-dimensional state space.

### Fixed contrast state versus transported coefficient state

These are two different ways to use the same parent/child split equations. The
difference is the permanent MCMC state, not the visual basis partition.

| Property | Fixed contrast state | Transported coefficient state |
| --- | --- | --- |
| Permanent continuous state | one root coordinate plus every potential tree contrast | coefficients attached to the currently active regions, plus explicit reverse auxiliaries or inactive storage |
| Dimension | fixed for every partition | packed active dimension changes, or a fixed container is transformed during a move |
| Split operation | changes which existing contrasts the decoder uses | maps parent coefficient and auxiliary variables to child coefficients |
| Merge operation | deactivates a contrast; its value remains in the state under its pseudo-prior | maps child coefficients back to a parent and recoverable reverse auxiliaries |
| Proposal Jacobian | none if stored coordinates do not change | required for a non-volume-preserving transformation |
| Primary advantage | simplest exact product-space target and fixed-shape trace | current-dependent proposals can place new active coefficients near plausible values |
| Primary cost | NUTS or another continuous kernel sees all potential contrasts | reversible bookkeeping, proposal densities, and Jacobians are harder to verify |

In the **fixed contrast state**, a state might be written

\[
s=(P,z_0,\delta_1,\ldots,\delta_J,\theta),
\]

where every potential split contrast exists at every iteration. The partition
selects an ancestry-consistent subset and a deterministic decoder constructs
the active region coefficients. A split activates an existing \(\delta_v\); a
merge makes it inactive again. The parent/child equations are a change of
coordinates used during decoding, not a proposal that changes the stored
state. No proposal Jacobian is added merely because the decoder contains a
nonlinear transformation. A prior specified on decoded leaf coefficients may,
however, require its own ordinary change-of-variables term; that is a property
of the target density and must not be confused with a proposal Jacobian.

In the **transported coefficient state**, the proposal itself changes stored
continuous values. A split draws or consumes an auxiliary \(u\) and applies an
invertible map

\[
(z_A',z_I',u')=T_{P\rightarrow P'}(z_A,z_I,u).
\]

The acceptance probability is

\[
\alpha_T
=
1\wedge
\frac{
\widetilde\pi(P',z_A',z_I',\theta\mid y)
j(P',P)
g_{P'\rightarrow P}(u'\mid z_A',z_I')
}{
\widetilde\pi(P,z_A,z_I,\theta\mid y)
j(P,P')
g_{P\rightarrow P'}(u\mid z_A,z_I)
}
\left|
\det\frac{\partial(z_A',z_I',u')}{\partial(z_A,z_I,u)}
\right|.
\]

This is a transformed product-space MH move. If inactive coordinates are
collapsed out and only the packed active vector is retained, the same
construction is ordinary RJMCMC. If a proposal overwrites inactive coordinates,
their previous values must be recoverable as reverse auxiliaries or accounted
for in explicit forward and reverse proposal densities. Discarding them is not
reversible.

### Transformations and pseudo-priors

An invertible transformation can construct a useful pseudo-prior from a simple
base draw, but the transformation is not itself a probability density. If

\[
u\sim g(u),\qquad z_I=T(z_A,u),
\]

then the induced conditional pseudo-prior is the push-forward density

\[
q_P(z_I\mid z_A)
=
g\!\left(T^{-1}(z_A,z_I)\right)
\left|
\det\frac{\partial T^{-1}(z_A,z_I)}{\partial z_I}
\right|.
\]

The first prototype should normally store the base contrast \(u\) directly and
give it a simple normalized pseudo-prior. This makes the decoder explicit and
usually avoids an additional Jacobian in the partition proposal. A later
transported proposal can reuse the same split geometry to improve switching,
while retaining the fixed-state implementation as a correctness baseline.

### Is it different from RJMCMC?

Yes, if the complete parameter vector is fixed and proper pseudo-priors define
the inactive coordinates. There is then no map between Euclidean spaces of
different dimensions and no reversible-jump Jacobian.

No, if the implementation keeps only a packed active vector whose length changes
with \(K(P)\). A fixed-length indicator or multiscale matrix does not alter that
mathematical dimension change. A sampler that constructs and destroys active
coefficients must perform the equivalent of RJ dimension matching even if the
bookkeeping is described differently.

### Are there three Gibbs steps?

There are naturally three update blocks, but they are not all necessarily Gibbs
updates:

1. **Inactive pseudo-prior refresh.** Draw selected inactive coordinates from
   \(q_P\). This can be an exact Gibbs/direct draw when \(q_P\) is simple.
2. **Partition update.** Propose a valid split/merge or tree decision and accept
   it using the augmented target. This is normally MH. Original finite-model
   Carlin-Chib can Gibbs-sample a model indicator when all model probabilities
   can be enumerated; that is unrealistic for a large partition space.
3. **Active continuous update.** Update active coefficients and shared
   hyperparameters conditional on \(P\), potentially using NUTS.

A practical order is inactive refresh, partition move, active update, so newly
activated coordinates have plausible values. It is not necessary to refresh
every inactive coordinate on every sweep. A proposal can draw only the
coordinates required by a proposed split. In a Dellaportas-style candidate
refresh, the refreshed inactive coordinate remains part of the augmented state
even if the subsequent partition switch is rejected. Alternatively, one joint
MH proposal can include the draw through explicit forward and reverse proposal
densities.

### PyMC feasibility

PyMC supports compound step methods for fixed model graphs and can combine
Metropolis-family updates with NUTS. That does not automatically provide
dynamic active-only NUTS:

- NUTS normally has a fixed compiled variable list and fixed point shapes;
- applying NUTS to every potential tile may make the dimension and inactive
  geometry unnecessarily large;
- changing the set of NUTS variables with \(P\) likely requires custom sampler
  orchestration, cached conditional kernels, or a fixed-count packed state;
- changes in discrete structure can alter continuous geometry and make a single
  globally adapted mass matrix inefficient. Once tuning is frozen, a mismatched
  metric affects performance rather than invalidating a correctly
  Metropolis-corrected NUTS kernel;
- PyMC's external NUTS backends use an exclusive full-model NUTS path and require
  a fully continuous model. They cannot be composed with this mixed Python step
  sequence. An initial mixed discrete/continuous product-space prototype would
  therefore need native `nuts_sampler="pymc"` or framework-independent outer
  orchestration rather than the NumPyro, BlackJAX, or nutpie routes.

The concrete PyMC design is:

1. represent \(P\), the complete fixed-shape contrast vector \(z\), and
   continuous hyperparameters in one static PyTensor graph;
2. implement the partition transition as an explicitly instantiated blocked
   step method that receives and returns a complete PyMC point;
3. explicitly instantiate native `pm.NUTS` for \(z\) and continuous
   hyperparameters; and
4. run the partition step followed by NUTS in a `CompoundStep`.

For a structure-only move, the custom step owns only \(P\) and NUTS owns \(z\),
which follows the documented compound-step boundary. A transported move changes
both \(P\) and selected entries of \(z\). PyMC 5.25 and 5.26 do not currently
reject explicitly overlapping step assignments, and a local 5.25.1 smoke test
successfully ran a custom joint \((P,z)\) step followed by NUTS on \(z\).
Automatic assignment will not create this overlap; both steps must be
instantiated explicitly. This is implementation-level, version-pinned behavior,
not a documented stability guarantee. The experimental adapter must therefore
pin supported PyMC versions and have focused smoke, serialization,
tuning-shutdown, and step-order tests.

A single NUTS mass matrix is adapted across the partitions visited during
warmup. Prior-whitened root/contrast coordinates and simple inactive
pseudo-priors should make this geometry less partition-specific. Once warmup is
over and adaptation is frozen, a poorly matched metric affects efficiency, not
the invariance of correctly implemented component kernels. The first
framework-independent product-space kernel should still precede the PyMC
adapter so exact enumeration can distinguish target errors from NUTS tuning
problems.

## Tree of split contrasts

### Motivation

A coefficient for every possible tile is redundant because parent and child
coefficients describe overlapping common modes. A tree of contrasts instead
stores:

- one root/common coefficient;
- one potential contrast \(\delta_v\) for each potential split;
- partition decisions that activate an ancestry-closed subset of contrasts.

This gives each split one explicit new degree of freedom and aligns directly
with the cheap split-contrast calculation in `_contrast.py`.

### Additive mass-preserving transform

For parent \(G=A\cup B\), let \(\alpha_G\) be the parent coefficient and
\(\delta=\alpha_A-\alpha_B\). Define

\[
\alpha_A
= \alpha_G + \frac{\mu_B}{\mu_G}\delta,
\qquad
\alpha_B
= \alpha_G - \frac{\mu_A}{\mu_G}\delta.
\]

Then

\[
\mu_A\alpha_A + \mu_B\alpha_B = \mu_G\alpha_G.
\]

The absolute Jacobian determinant of
\((\alpha_G,\delta)\mapsto(\alpha_A,\alpha_B)\) is one. Swapping the child order
changes the signs of \(\delta\) and \(f_\delta\), but does not change the scalar
information scores.

Activating \(\delta\) preserves the parent common mode while adding exactly the
child contrast direction \(f_\delta\) defined earlier.

### Positive/lognormal transform

For positive multiplicative coefficients, write \(x=\exp(z)\) and let
\(\delta=z_A-z_B\). An exact positive mass-preserving transform is

\[
x_B
= \frac{\mu_G x_G}{\mu_A\exp(\delta)+\mu_B},
\qquad
x_A = \exp(\delta)x_B.
\]

This preserves \(\mu_Ax_A+\mu_Bx_B=\mu_Gx_G\) and gives a useful positive split
coordinate. It does not preserve independent lognormal priors on the child
coefficients: Gaussian root and contrast coordinates generally induce
correlated, non-lognormal child marginals because of the normalizing
denominator. The prior must be defined and evaluated in one parameterization
rather than inferred from positivity alone.

In a product-space parameterization all potential \(\delta_v\) already exist,
so changing activation changes only the deterministic mapping to leaf
coefficients. There is no dimension-changing Jacobian. In a packed
trans-dimensional implementation, the same transform can serve as an RJMCMC
dimension-matching map. For the map
\((x_G,\delta)\mapsto(x_A,x_B)\), its absolute Jacobian
\(x_Ax_B/x_G\) must be included.

### Cheap DoFS and local Gaussian proposals

Under a local Gaussian approximation, the new contrast has scalar information
\(\lambda\) as defined above. Given residual \(r\), a Gaussian prior variance
\(\tau^2\), and observation covariance \(R\), a local conditional approximation
is

\[
V_\delta
= \left(\tau^{-2}+f_\delta^T R^{-1}f_\delta\right)^{-1},
\qquad
m_\delta
= V_\delta f_\delta^T R^{-1}r.
\]

This can initialize or propose a newly active contrast and can guide its
pseudo-prior. For a non-Gaussian positive model it remains a local surrogate;
the true non-Gaussian likelihood and prior must determine final acceptance.

### Open contrast questions

- What mass \(\mu\) should be preserved: area, prior flux, absolute flux, or a
  group-specific quantity?
- Should the prior be specified directly on tree contrasts or induced from a
  desired prior on leaf coefficients?
- How should contrast scales vary by depth, tile area, land/ocean class, or
  inner/outer group?
- Does the positive transform give adequate NUTS geometry near very small
  coefficients?
- How should x/y split orientation be encoded without duplicate partition
  representations?

## Masks, layers, and grouped state vectors

Land/ocean, country, positive/negative flux, user rectangles, and inner/outer
regions are not merely plotting metadata. They can define hard boundaries,
different weight fields, different priors, and different posterior summaries.

The current fixed-layout design should remain the external representation:

- `BasisLayout` identifies partitions and groups;
- xarray coordinates identify group, partition, and local region in retained
  artifacts;
- fixed InTEM outer regions remain a familiar compatibility option;
- inner and outer generated regions can use different weights.

Dynamic inference needs an additional immutable `PartitionState` with active
tile IDs, decisions, and group-local topology. It should not mutate
`BasisLayout` on every MCMC transition. A sampled or selected partition can be
converted to `BasisLayout` for output, plotting, or a subsequent fixed-basis
inversion.

**Proposed initial scope:** run one independent dyadic tree per hard class or
group. Allocate or infer complexity within each group and combine their active
columns in a stable group order. This avoids tiles crossing land/ocean or
inner/outer boundaries. Layer intersections and disconnected components should
be resolved before tree construction and flagged when they produce tiny or
pathological parts.

## Synthetic inner/outer proof of concept

### Why use synthetic observations

The first product-space experiment should generate observations from the same
declared forward model used for inference. It then needs no boundary-condition
product and avoids interpreting a corrupted or incomplete stored observation
baseline. The synthetic response is an anomaly relative to zero by definition:

\[
y
=
H_{\mathrm{inner},P_{\mathrm{true}}}x_{\mathrm{inner,true}}
+H_{\mathrm{outer}}x_{\mathrm{outer,true}}
+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,R).
\]

This does not assert that real atmospheric observations have zero boundary
contribution. A later real-observation experiment must supply a valid baseline
or boundary-condition block. The synthetic experiment isolates partition
inference from that unrelated data dependency.

The first exact experiment is an explicitly regional Gaussian model, not the
exact Bocquet coarsening of one native Gaussian prior. Its partition changes the
regional prior/likelihood model and can therefore have a nontrivial
\(p(P\mid y)\). This distinction must remain explicit when results are reported.

### InTEM geometry

Use an InTEM outer-region definition to separate the domain:

- the maximum InTEM label is the dynamic inner region, following the existing
  compatibility convention;
- every other InTEM label contributes one fixed outer basis column;
- only the inner region receives a dyadic tree and variable partition \(P\);
- the first prototype may keep land and ocean mixed within each fixed outer
  label to minimize the always-active state dimension;
- a later comparison can intersect inner and outer labels with land/ocean
  classes.

The packaged EUROPE map has labels `0` through `6`, with `6` as its original
inner region. The local `verification-games` checkout at commit `c0c9bcc` also
contains a reproducible CTE-HR/PARIS adaptation that keeps outer labels `0`
through `6` and assigns `7` to a smaller inner rectangle. The adapted geometry
is an external experimental precedent and is preferred for the realistic proof
of concept because it more directly limits the dynamic domain. Its construction
must be ported or vendored with provenance rather than imported from a private
checkout at runtime. The tiny exact oracle uses a generated mask and depends on
neither file.

This gives a small dynamic domain without treating the omitted domain as known
zero flux. It also matches the grouped interpretation in
`docs/plans/fixed_outer_regions_grouping.md`: fixed outer labels are explicit
state-vector components, not a mask applied after inner optimization.

### Separate inner and outer priors

Write the complete design as

\[
H_P=\left[H_{\mathrm{inner},P}\;H_{\mathrm{outer}}\right],
\qquad
x_P=
\begin{bmatrix}
x_{\mathrm{inner},P}\\
x_{\mathrm{outer}}
\end{bmatrix}.
\]

The baseline Gaussian proof of concept should use distinct prior groups:

\[
x_{\mathrm{inner},P}
\sim
\mathcal N(\boldsymbol 1,B_{\mathrm{inner},P}),
\qquad
x_{\mathrm{outer}}
\sim
\mathcal N(\boldsymbol 1,B_{\mathrm{outer}}).
\]

The blocks can initially be independent and diagonal, with separate scales
\(\sigma_{\mathrm{inner}}\) and \(\sigma_{\mathrm{outer}}\). The
The local `verification-games` calibration prototype at commit `c0c9bcc` already
records this distinction, using provisional scaling-factor standard deviations
of `1.0` for inner states and `0.5` for fixed outer states. Those values are
external implementation precedents, not calibrated defaults or behavior
provided by this repository; both must be declared and tested for sensitivity.

For the first proof of concept, define the primitive inner prior directly in
root/contrast space. Let \(a=x-1\) be a scaling-factor anomaly, let \(n_G\) be
the number of finest search grid cells under node \(G\), and let
\(G=L\cup R\). Use

\[
a_{\mathrm{root}}
\sim
\mathcal N(0,\tau_{\mathrm{inner}}^2/n_{\mathrm{root}}),
\qquad
\delta_G
\sim
\mathcal N\!\left(
0,
\tau_{\mathrm{inner}}^2
\left(\frac{1}{n_L}+\frac{1}{n_R}\right)
\right).
\]

For each partition, let \(T_P\) decode the active root/contrast vector into
active regional anomalies. Then

\[
x_{\mathrm{inner},P}=\boldsymbol 1+T_Pz_A,
\qquad
B_{\mathrm{inner},P}=T_PB_{z,A(P)}T_P^T.
\]

This choice induces independent active-region means with variance
\(\tau_{\mathrm{inner}}^2/n_G\) under the grid-cell-count mass convention. It
also ensures that the product-space target and analytically integrated Gaussian
oracle use exactly the same prior. Each inactive contrast initially uses the
same normalized Gaussian as its pseudo-prior. A later physical-mass convention
can replace grid-cell counts, but it must redefine both \(T_P\) and the contrast
prior together.

A separate outer prior is not a mathematical requirement for product-space
sampling, but it is the appropriate initial model. Outer states aggregate much
larger and differently supported regions, are always active, and should not
inherit a contrast-depth prior intended for dynamic inner splits. Keeping them
in a distinct block also prevents inner pseudo-prior or partition changes from
silently changing their prior semantics. In the later positive model, each
block can use a lognormal scaling prior with moments matched to its declared
mean and standard deviation.

Outer coefficients are ordinary always-active model parameters, not
pseudo-prior coordinates. Only inactive inner contrasts receive pseudo-priors.
The augmented target therefore factors as

\[
\begin{aligned}
\widetilde\pi(
P,z_{A},z_I,x_{\mathrm{outer}},
\theta_y,\theta_{\mathrm{in}},\theta_{\mathrm{out}},\theta_P
\mid y)
\propto{}&
p(y\mid P,z_A,x_{\mathrm{outer}},\theta_y)
p(z_A\mid P,\theta_{\mathrm{in}})
q_P(z_I\mid z_A,\theta_{\mathrm{in}})\\
&\times
p(x_{\mathrm{outer}}\mid\theta_{\mathrm{out}})
p(P\mid\theta_P)
p(\theta_y)p(\theta_{\mathrm{in}})
p(\theta_{\mathrm{out}})p(\theta_P).
\end{aligned}
\]

### Two validation scales

1. **Exact tiny-tree oracle.** Use a synthetic `1 x 3` inner grid, two fixed
   outer columns, a small number of observations, and diagonal \(R\). Its three
   canonical partitions form a path with proposal degrees `1, 2, 1`, making it
   the smallest case that detects an omitted or incorrectly signed Hastings
   correction. Enumerate every valid inner partition and integrate the Gaussian
   coefficients analytically. Retain a `2 x 2` five-partition case as the first
   branched-tree check. These are the correctness oracles for transition
   probabilities and sampled partition frequencies.
2. **TAC/MHD-shaped synthetic experiment.** Reuse retained TAC/MHD footprint
   times prior-flux sensitivities where available, but draw coefficients and
   observations from the declared model. Apply the InTEM split before building
   the candidate tree, keep fixed outer columns, and optimize/sample only the
   inner tree. The preferred exact scientific oracle aggregates the adapted
   inner rectangle to a `4 x 4` macro grid: one root plus 15 possible split
   contrasts, seven fixed mixed outer coefficients, and 23 continuous
   coordinates in total. That canonical 16-leaf binary tree has 677 valid
   prunings, which can still be enumerated exactly. A proposed deterministic
   observation selection is 32 fitting observations and 16 held-out
   observations, balanced between TAC and MHD. This checks realistic geometry
   and design-matrix scaling without requiring real observations or boundary
   conditions.

Use a recursive split/stop partition prior for the scientific oracle, initially
with split probability \(\rho=0.5\) and sensitivity runs at \(0.25\) and
\(0.75\). This defines prior mass on canonical tree frontiers rather than on
split histories. If \(S(P)\) is the set of split internal nodes and \(U(P)\)
is the set of unsplit but splittable frontier nodes, its probability mass is

\[
p(P\mid\rho)
=
\prod_{v\in S(P)}\rho_v
\prod_{v\in U(P)}(1-\rho_v).
\]

Forced terminal tree leaves contribute no stop factor, equivalently
\(\rho_v=0\) there. This recursion normalizes over the 677 canonical
frontiers. Retain a comparator with \(p(K)=1/16\) and
\(p(P\mid K)=1/N_K\), where \(N_K\) is the number of canonical frontiers with
\(K\) leaves. That comparator separates an intended prior on region count from
the combinatorial multiplicity of partitions. After analytically integrating
the always-active outer block and active inner coefficients,

\[
y\mid P
\sim
\mathcal N\!\left(
H_P\boldsymbol 1,
R
+H_{\mathrm{inner},P}B_{\mathrm{inner},P}
 H_{\mathrm{inner},P}^{T}
+H_{\mathrm{outer}}B_{\mathrm{outer}}H_{\mathrm{outer}}^{T}
\right),
\]

so all 677 normalized partition probabilities are available as a scientific
oracle as well as the smaller transition-matrix oracle.

The generated artifact must record the true partition, true inner and outer
coefficients, the prior groups and scales, \(R\), random seed, and whether
land/ocean classes were applied. Validation should cover partition frequencies,
posterior recovery of inner and outer coefficients, posterior predictions,
coverage over repeated synthetic draws, and sensitivity to pseudo-prior scale.

## Filtering and data flow

The current compatibility weight in `_functions.py` multiplies temporal mean
flux by the measurement-weighted mean footprint. Absolute flux is optional and
off by default. A scientifically common alternative is to form time-resolved
footprint-times-flux contributions, often using absolute flux, and then average
over retained sites and times. These definitions are not generally equivalent
and should be named separately. Under either data-dependent definition, if
observations are filtered after the calculation, rejected observations influence
the basis even though they do not enter the inversion.

**Required ordering:** load data, apply all observation filters, construct the
fine-grid contribution matrix from retained observations, then build multiscale
columns and weights. Any train/holdout split must occur before basis fitting for
the corresponding evaluation.

The prepared experimental input should contain at least:

- retained \(y\), uncertainty/covariance inputs, and observation coordinates;
- the unreduced fine-grid contribution block \(G\);
- fixed design blocks such as boundary conditions;
- grid coordinates, cell area, and hard region classes;
- source/group metadata;
- a record of filtering and any train/holdout membership.

This is intentionally lower level than `fp_all`. Weight builders can adapt
footprints and flux into \(G\), but the partition algorithms should consume
NumPy arrays and explicit metadata.

## Proposed implementation sequence

### Phase 0: public code consolidation

Implement the package, demo, tests, provenance table, and benchmark described
above. Do not add MCMC yet.

### Phase 1: scientifically faithful representation-optimization demo

1. Build multiscale observation columns with the exact declared restriction,
   prolongation, prior, and aggregation-error semantics.
2. Initialize with the current greedy or bisection partitioner.
3. Optimize Equation 45 as adaptive posterior compression, and compare its
   partition with DFS, Fisher, and projected-flux compression objectives.
4. Use exact fixed-count DP as the oracle whenever the tile score is additive;
   retain SLS for richer dictionaries and dense correlated objectives.
5. Record fitting observations separately from held-out predictive or
   compression evaluation.
6. Add a real cooling schedule and best-state tracking where SLS remains
   necessary.
7. Compare against direct recomputation on small problems.
8. Demonstrate optional hard land/ocean and inner/outer classes.

The result is an optimizer and experimental basis generator, not posterior
inference.

### Phase 2a: collapsed fixed-count sampler for a partition-dependent model

This is not a sampler for the exact Bocquet projection model. In that model,
integrating the projected coefficients and exact aggregation residual recovers
the same native innovation distribution for every \(P\), so
\(p(P\mid y)=p(P)\). Before this phase, define and name a reduced model in which
\(P\) genuinely changes the likelihood or prior, such as an explicitly
low-rank regional field model.

1. Implement paired split-and-merge neighbor enumeration with exact proposal
   probabilities.
2. Gather \(H_P\) into a fixed \(K\)-column operator.
3. Compute the exact collapsed Gaussian \(p(y\mid P)\).
4. Compare uniform and DoFS-informed local proposals.
5. Use stochastic local search only for initialization and proposal scores.
6. Validate sampled partition frequencies against exhaustive enumeration.

This phase establishes the partition state, target, proposal, and local-mixing
behavior without coefficient transport or NUTS. The hackathon proof of concept
is a deliberately narrow slice through Phases 0, 1, and 2a rather than a
requirement to finish all earlier productionization work first.

### Phase 2b: non-Gaussian fixed-count sampler

1. Add canonical node-keyed active coefficients and prior whitening.
2. Implement a full-rank reversible proposal for coefficients affected by a
   paired partition move.
3. Compare elliptical slice sampling, pCN, fixed diagonal HMC, and conditional
   native PyMC NUTS for the continuous block.
4. Use the true non-Gaussian likelihood and prior for final acceptance.
5. Diagnose partition switching before investing in structure-specific metrics
   or additional NUTS adaptation machinery.

This phase is optional evidence for the active-only non-Gaussian route. It is
not a prerequisite for product-space or RJMCMC work if Phase 2a already exposes
poor local partition traversal.

### Phase 3: tree-contrast product-space prototype

This phase is split into independently reviewable tracker tasks so the exact
target does not become entangled with PyMC integration or coefficient
transport:

1. **OGI-062, reference:** specify the augmented target, pseudo-prior
   normalization, state representation, inner/outer synthetic model, and PyMC
   boundary in this document.
2. **OGI-063, framework-independent fixed-contrast kernel:**
   - build a tiny variable-count enumerator that provides exact partition
     probabilities for the chosen tree and priors;
   - choose a canonical tree and define one root plus the complete contrast
     coordinate set;
   - specify proper factorized Gaussian pseudo-priors;
   - implement inactive refresh and structure-only partition MH as separate
     kernels;
   - add the synthetic inner/outer Gaussian design with fixed outer columns;
   - verify detailed balance and sampled partition frequencies against exact
     enumeration.
3. **OGI-064, native PyMC adapter:**
   - represent the same target in one static PyTensor graph;
   - wrap the verified partition kernel in a blocked step method;
   - update whitened contrasts, outer coefficients, and continuous
     hyperparameters with native PyMC NUTS;
   - verify step ordering, tuning shutdown, fixed trace shapes, serialization,
     and agreement with the framework-independent oracle.
4. **OGI-065, transported proposals:**
   - add candidate-only Dellaportas-inspired pseudo-prior refreshes under the
     documented common-factor assumptions;
   - implement reversible split/merge coefficient transport with explicit
     forward and reverse auxiliaries;
   - test inverse maps, numerical and analytic Jacobians, and log-ratio
     antisymmetry;
   - compare switching efficiency with the fixed-contrast structure-only
     baseline without changing the target distribution.

After those increments, compare full-vector, active-only, and cached-kernel
execution strategies. Measure partition switching, active and inactive
effective sample sizes, likelihood cost, and sensitivity to pseudo-prior
calibration. Compare posterior results with exact enumeration and Phase 2a
fixed-count results.

### Phase 4: scale-up and alternatives

Only after Phases 2 and 3:

- scale variable-count inference to realistic trees and assess alternative
  partition and leaf-count priors;
- compare product-space with a direct RJMCMC split/merge implementation;
- evaluate delayed acceptance using DFS, contrast, or Laplace surrogates;
- investigate SMC or parallel tempering for multimodal partitions;
- consider pseudo-marginal methods only if an unbiased useful estimator exists;
- support multiple independently optimized groups and time-varying partitions.

## Validation requirements

### Deterministic and algebraic tests

- every valid partition covers each included grid cell exactly once;
- no active node has an active ancestor or descendant;
- every split has a valid inverse merge;
- proposal probabilities normalize on tiny states;
- additive and positive contrast transforms preserve their stated mass;
- every transported map has an exact inverse and its analytic Jacobian agrees
  with numerical differentiation;
- gathered multiscale columns equal direct fine-grid aggregation;
- Python and Numba kernels agree over random small arrays;
- label and grouped-layout conversion preserves stable ordering.

### MCMC correctness tests

- detailed balance for every transition on a tiny enumerated state space;
- sampled partition frequencies agree with exact probabilities in a conjugate
  toy model;
- fixed-partition continuous results agree with the ordinary model;
- product-space marginal probabilities are invariant to normalized
  pseudo-priors, within Monte Carlo error;
- fixed outer coefficients remain always active and retain their declared prior
  group under every inner partition;
- the PyMC compound sampler agrees with the framework-independent tiny-tree
  oracle and preserves the documented partition-step/NUTS order;
- simulation-based calibration or posterior coverage for a small generated
  problem;
- checkpoint/restart reproduces the transition stream from saved RNG state.

### Scientific evaluation

- train/holdout projected-flux compression from OGI-048;
- posterior predictive performance on an outer holdout;
- DFS or expected-information diagnostics under their calibrated Gaussian
  assumptions;
- sensitivity to complexity priors, pseudo-priors, tree orientation, and group
  boundaries;
- land/ocean and inner/outer posterior summaries using group metadata;
- repeated synthetic coverage for separately-prior inner and fixed outer
  coefficients;
- runtime and memory scaling in observations, fine cells, possible tiles, and
  active tiles.

## Risks and exactness boundaries

- A toy pre-summed weight tree is not a faithful replacement for multiscale
  observation columns.
- A fixed indicator does not make a packed variable-length coefficient state
  fixed-dimensional.
- An uncorrected Laplace, DoFS, or projected-flux acceptance rule targets an
  approximation, not the desired posterior.
- Poor pseudo-priors can make a formally correct product-space chain practically
  immobile.
- Applying NUTS to all potential inactive coordinates may remove the intended
  computational advantage.
- Reusing one adapted NUTS geometry across substantially different partitions
  may mix poorly even when the compound chain is valid.
- Allowing arbitrary x/y split histories can assign unintended multiplicity to
  equivalent partitions.
- Dynamic ragged region coordinates do not fit ordinary xarray traces. Store
  fixed node decisions/contrasts or grid-level reconstructed fields during
  sampling, and convert selected partitions to fixed layouts afterward.
- Data filtering after basis construction leaks excluded observations into the
  basis design.

## Open design questions

1. Is the first exact demonstration fixed \(K\) only, or should a tiny conjugate
   variable-\(K\) enumerator be built alongside it as a test oracle?
2. Which canonical dyadic tree best matches the useful partitions: alternating
   binary axes, geometry-chosen axes recorded in the state, or quadtree splits?
3. What is the intended formula and convention for the prototype's inverse-error
   input to `make_weights`?
4. Should multiscale columns be stored densely, chunked by observation/tile, or
   generated through prefix sums and an operator interface?
5. Which variables define \(\mu\) and the contrast prior at each split?
6. Can group-specific trees share a total complexity prior while retaining
   independent weights and priors?
7. Is a custom PyMC step method sufficient, or should partition inference use a
   framework-independent outer sampler that calls cached PyMC conditional
   kernels?
8. Which pseudo-prior family gives acceptable switching for lognormal contrasts?
9. Should the first delayed-acceptance experiment use the existing contrast
   score, a Laplace approximation, or both?
10. What trace representation is needed for posterior maps, group summaries,
    and model averaging without a ragged `region` dimension?
11. For dense correlated \(B\), should the first exact geographic experiment
    preserve the piecewise-regional prolongation or preserve literal aggregate
    restriction semantics?
12. Is Equation 45 evaluated with a strict design/inference split, cross-fitting,
    or as an explicitly descriptive posterior-compression action followed by
    held-out predictive assessment?
13. Does a fixed-count experiment constrain tree leaves, effective supported
    coefficient dimension, or an information target such as retained DFS?
14. Should the first realistic product-space experiment use the packaged InTEM
    map or the CTE-HR/PARIS-adapted InTEM map currently used in
    `verification-games`? The tiny exact oracle must not depend on either file.
15. Should outer coefficients stay independent in the first Gaussian model, or
    should a later experiment add an explicitly declared outer covariance?

## References

### Statistical and inverse-problem references

- Andrieu, C. and Roberts, G. O. (2009). "The pseudo-marginal approach for
  efficient Monte Carlo computations." *The Annals of Statistics*, 37(2),
  697-725. <https://doi.org/10.1214/07-AOS574>.
- Bocquet, M., Wu, L., and Chevallier, F. (2011). "Bayesian design of control
  space for optimal assimilation of observations. Part I: Consistent multiscale
  formalism." *Quarterly Journal of the Royal Meteorological Society*, 137(658),
  1340-1356. <https://doi.org/10.1002/qj.837>.
- Carlin, B. P. and Chib, S. (1995). "Bayesian Model Choice Via Markov Chain
  Monte Carlo Methods." *Journal of the Royal Statistical Society: Series B*,
  57(3), 473-484. <https://doi.org/10.1111/j.2517-6161.1995.tb02042.x>.
- Christen, J. A. and Fox, C. (2005). "Markov chain Monte Carlo Using an
  Approximation." *Journal of Computational and Graphical Statistics*, 14(4),
  795-810. <https://doi.org/10.1198/106186005X76983>.
- Dellaportas, P., Forster, J. J., and Ntzoufras, I. (2002). "On Bayesian model
  and variable selection using MCMC." *Statistics and Computing*, 12(1), 27-36.
  <https://doi.org/10.1023/A:1013164120801>.
- Godsill, S. J. (2001). "On the Relationship Between Markov Chain Monte Carlo
  Methods for Model Uncertainty." *Journal of Computational and Graphical
  Statistics*, 10(2), 230-248.
  <https://doi.org/10.1198/10618600152627924>.
- Green, P. J. (1995). "Reversible jump Markov chain Monte Carlo computation and
  Bayesian model determination." *Biometrika*, 82(4), 711-732.
  <https://doi.org/10.1093/biomet/82.4.711>.

The Dellaportas citation is sometimes reported incorrectly as volume 16, pages
57-68. The verified 2002 article is volume 12(1), pages 27-36. Volume 16,
pages 57-68 corresponds to a different Dellaportas paper.

### Software reference

- PyMC, "Compound Steps in Sampling." The example documents fixed-graph
  Metropolis-within-Gibbs combinations and cautions about changing continuous
  geometry when discrete and NUTS updates are mixed:
  <https://www.pymc.io/projects/examples/en/stable/howto/sampling_compound_step.html>.
- PyMC, `pymc.sample` API. This records the current native and external NUTS
  sampler options and the continuous-model restriction on external samplers:
  <https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html>.
- PyMC, "Using a custom step method for sampling from locally conjugate
  posterior distributions." This demonstrates direct `BlockedStep`
  implementation inside compound sampling:
  <https://www.pymc.io/projects/examples/en/latest/samplers/sampling_conjugate_step.html>.

### Repository reference points

- `docs/plans/mask_constrained_basis.md`
- `docs/plans/state_vector_grouping.md`
- `docs/plans/fixed_outer_regions_grouping.md`
- `docs/plans/ogi_048_basis_algorithm_options.md`
- `openghg_inversions/basis/algorithms/_constrained.py`
- `openghg_inversions/basis/algorithms/_contrast.py`
- `openghg_inversions/basis/layout.py`
- `openghg_inversions/basis/operators.py`
- `openghg_inversions/basis/basis_functions.py`
- `verification-games/src/verification_games/rhime_calibration/basis.py`
  for the experimental fixed InTEM outer/dynamic inner construction
- `verification-games/src/verification_games/rhime_calibration/config.py`
  for the provisional separate inner and outer prior scales
- `verification-games/src/verification_games/rhime_calibration/analytic.py`
  for the existing analytic Gaussian inversion machinery
- branch `codex/basis-prototype-examples`, commit `b6ce565`
