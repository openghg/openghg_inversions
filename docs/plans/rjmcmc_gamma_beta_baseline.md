# Fixed-direction Gamma--Beta RJMCMC baseline

## Purpose and scope

This work is the first alternative structural-mixing baseline after the
moving-Voronoi Stage C profile. It deliberately samples prunings of one
canonical fixed-direction dyadic supertree. It is not the final full-tiling
model and does not represent alternative split orientations.

The baseline has no production-data dependency. Its first acceptance gates are
exact tiny-tree enumeration, pointwise reversible-jump accounting, prior-only
mobility, and a two-cell synthetic likelihood.

## Scientific state and reference measure

The scientific partition is a canonical active frontier \(F\), not a
chronological sequence of splits and not a full vector of active and inactive
split indicators. On one fixed tree, the frontier uniquely determines its
active ancestor splits \(I(F)\).

The active-only continuous coordinates are

\[
z = (F, T, \rho_{I(F)}),
\]

where \(T>0\) is the root total and every \(0<\rho_v<1\) is aligned with a
stable internal-node ID. The reference measure is counting measure on
frontiers times Lebesgue measure on \(T\) and the active fractions.

Mass propagates through an active split as

\[
X_{v_L}=\rho_v X_v,\qquad
X_{v_R}=(1-\rho_v)X_v.
\]

Consequently the active leaf masses always sum to \(T\). Within an unresolved
active leaf, native-cell mass is fixed at the nominal within-leaf proportions.
This removes unresolved contrasts rather than exactly marginalizing them, so
the partition can affect the likelihood. It must not be described as an exact
representation-invariant projection.

The problem's sensitivity matrix is response per unit physical cell mass.
RHIME `fp_x_flux` is normally response to a unit cell scaling coefficient, so
a future real-data adapter must pass \(S_j=G_j/m_j\), where \(G_j\) is the
`fp_x_flux` column and \(m_j\) is nominal cell mass. For node scaling \(a_v\),
\(X_v=a_v\sum_jm_j\) then gives \(D_vX_v=a_v\sum_jG_j\). Passing raw
`fp_x_flux` directly would double-weight the nominal flux.

## Prior

The root uses a normalized Gamma shape--rate density. Active split fractions
use normalized Beta densities. The implementation supports:

1. a fixed-tree validation profile with constant concentration
   \(\kappa\) and mean determined by nominal child mass; and
2. an additive-cell-shape profile in which child Beta shapes are sums of a
   native-cell base measure, the order-consistent route toward a later
   full-tiling prior.

The initial validation profile uses root variance \(0.25\), mean one,
constant \(\kappa=2\), and uniform marginal mass over supported \(K\).
These are validation settings inherited from the tiny product-space oracle,
not a scientific production calibration.

For the structural prior,

\[
p(F)=\frac{p_K(K(F))}{N_{K(F)}},
\]

where \(N_K\) is the exact number of frontiers with \(K\) leaves, computed by
tree-polynomial dynamic programming. Uniform-\(K\) and truncated-geometric
marginals are explicit normalized options.

## Local reversible-jump kernel

Each structural opportunity independently selects split or merge with
probability one half. An unavailable direction remains an explicit
self-transition.

For a split:

1. select one splittable active leaf uniformly;
2. draw its new fraction from the node's normalized Beta prior;
3. replace the leaf by its ordered children; and
4. insert the fraction by canonical node ID.

For a merge:

1. select one mergeable cherry parent uniformly;
2. replace its two active children by the parent; and
3. remove the parent's active fraction.

In root-plus-fraction coordinates, split/merge is coordinate
insertion/deletion and has unit Jacobian. The physical leaf-mass map
\((X,\rho)\mapsto(X\rho,X(1-\rho))\) has Jacobian \(X\); that factor must not
also be inserted into a target already defined in root-plus-fraction
coordinates.

The split ratio retains all terms explicitly:

\[
\log R =
\Delta\log L
+\Delta\log p(F)
+\log p(\rho)-\log g(\rho)
+\log q_{\mathrm{reverse,discrete}}
-\log q_{\mathrm{forward,discrete}}.
\]

The validation proposal uses \(g=p\), so the normalized Beta terms cancel
numerically, but both remain recorded and tested. A merge is the exact reverse
at the removed fraction.

## Implementation boundary

The baseline is a separate NumPy model under
`openghg_inversions.experimental.rjmcmc`. It does not widen the
Voronoi-specific problem, state, proposal, trace, or checkpoint unions.

The first executable schedule is intentionally structural-only: it fixes the
supplied root total and refreshes a fraction only when that split is newly
introduced. It is therefore a topology-mobility chain conditional on its
continuous coordinates, not an ergodic sampler for the full joint posterior.
The independent root and active-fraction refresh proposals exist for the later
compound schedule.

Implemented modules:

- `dyadic_tree.py`: canonical geometry, frontier validation, local topology,
  enumeration, and exact \(N_K\);
- `gamma_beta_tree.py`: Gamma--Beta and partition priors, problem/state
  construction, node design, and prediction;
- `gamma_beta_proposals.py`: split/merge terms and optional independent
  continuous refresh kernels; and
- `gamma_beta_sampling.py`: deterministic seeded orchestration, trace, and
  exact in-memory continuation for the structural-only baseline.

Numba, incremental design updates, masks, alternative orientations, fixed-\(K\)
swaps, and full tilings are deferred until the NumPy target and local kernel
pass exact checks.

## Validation sequence

1. Enumerate the five frontiers of a depth-two binary tree and verify
   \(N_1=1,N_2=1,N_3=2,N_4=1\).
2. Verify exact frontier coverage, canonical ordering, mass conservation, and
   split/merge round trips.
3. Verify normalized Gamma/Beta densities and distinguish rate from scale.
4. Verify pointwise forward/reverse proposal reciprocity, including source and
   candidate eligible-node counts.
5. Assemble the exact prior-only frontier transition matrix and check row
   normalization, detailed balance, and stationarity.
6. Compare empirical prior-only frontier and \(K\) frequencies with the exact
   distribution while recording edge traversal and immediate reversals.
7. Cross-check the active target against the preserved fixed-tree
   product-space oracle.
8. Use a two-cell Gaussian likelihood where the coarse prediction is
   \((2,2)\) and a split with \(\rho=0.25\) predicts \((1,3)\).
9. Add independent root/fraction prior refresh as a separate invariant kernel,
   not only after accepted structural moves.
10. Add Numba and cached node-design changes only after the full NumPy oracle
    remains available.

## Decisions and progress

### 2026-07-23

- Created branch `codex/rjmcmc-gamma-beta-baseline` from the completed
  diagnostics branch while Stage C runs independently.
- Mapped the archived Gamma--Beta/product-space work at `d12b5fd` and the
  sibling `inversions-knowledge` derivations.
- Chose the canonical longer-axis dyadic tree: rows win ties, node IDs use
  deterministic preorder, and odd extents give the second child the extra
  row or column.
- Chose active frontiers as scientific states, uniform \(p(K)\) for the first
  mobility comparison, prior-drawn split fractions, a 50/50 mixed structural
  kernel, and explicit boundary self-transitions.
- Chose root-plus-active-fraction coordinates with unit RJ Jacobian and fixed
  nominal within-leaf allocation as the reduced-model interpretation.
- Kept the active-only implementation separate from both the optimized
  Voronoi sampler and the fixed-dimensional PyMC/product-space code.
- Implemented immutable canonical topology/frontier objects, exact frontier
  enumeration and \(N_K\) counts, normalized Gamma--Beta and partition priors,
  full state reconstruction, native-cell rendering, and explicit target
  caches.
- Implemented pointwise split/merge terms with source/candidate eligible-node
  counts and normalized auxiliary densities. Root and fraction refresh
  proposals are separate invariant kernels; the first sampler intentionally
  schedules only the mixed structural kernel so its mobility can be measured
  in isolation.
- Added an every-opportunity structural trace and exact PCG64 continuation.
  This is an in-memory validation checkpoint, separate from the durable
  Voronoi checkpoint schema.
- Added exact five-state detailed-balance/stationarity checks and a seeded
  30,000-opportunity empirical mobility test. The latter visits every
  frontier, traverses structural edges in both directions, records immediate
  reversals, and recovers each exact frontier probability within 0.025.
- Made bounded structural priors retain and compute \(N_K\) only through their
  declared \(K_{\max}\). A local near-production-size construction
  (23,472 cells, \(K_{\max}=500\)) took 0.13 s for the canonical tree and
  1.36 s for the exact count/prior table; these are development-machine
  timings, not HPC sampling benchmarks.
- Reject zero-target-mass starting/source frontiers explicitly while retaining
  moves from supported states into excluded \(K\) as valid proposals with
  certain rejection.
- Verified 48 focused topology/state/proposal/sampler tests, the full
  experimental RJMCMC suite (570 passed, 2 deselected), Ruff, Pyright, and the
  complete parallel tox compatibility matrix.

## Real-data readiness phase

### Target profile

The first real-data smoke will use the frozen native PARIS-style input seam,
not ordinary reduced-basis `prepare_rhime_inputs` output. The verified dynamic
rectangle is the filled InTEM label-6 slice with shape \(183\times128\)
(23,424 cells), flattened in C order with longitude varying fastest.

The fair comparison profile predicts

\[
y_{\mathrm{pred}} =
y_{\mathrm{BC,fixed}}
+y_{\mathrm{inner,Gamma\text{-}Beta}}
+H_{\mathrm{outer}}x_{\mathrm{outer}},
\]

where the row-aligned archived `YaprioriBC` remains an explicitly selected
fixed offset and the six InTEM outer-region scalings are inferred with
declared lognormal priors. No boundary field is discovered implicitly.

The inner adapter requires an explicitly supplied strictly positive nominal
weight field \(w\). With normalized weights \(\sum_jw_j=1\), it passes
\(S_j=G_j/w_j\), sets the root prior/start around one, and exactly recovers the
all-one inner prediction at the unresolved root. Zero inventory cells are
never silently floored: flux-derived weights with zeros must fail until a
scientific masking/floor policy is chosen. Uniform-cell or area weights are
valid explicit alternative models, not hidden numerical fallbacks.

### Compound schedule

The structural-only sampler remains the correctness/mobility oracle. A
separate posterior sampler uses a deterministic cycle whose individual slots
are each invariant:

1. two internally mixed split/merge opportunities;
2. one independent-prior Gamma root refresh;
3. five independent-prior active-fraction refreshes, with node selection
   uniformly with replacement; and
4. one symmetric Gaussian update for each fixed outer coefficient.

For six outer coefficients this is a 14-transition cycle matching the current
opportunity accounting. Unlike the invalid historical deterministic
one-way structural cycle, each structural slot internally mixes both
directions. Impossible structural directions and fraction slots at \(K=1\)
remain explicit self-transitions.

Independent-prior root/fraction refreshes are tuning-free and establish a
correct irreducible baseline on connected positive \(p(K)\) support. Their
acceptance may be poor under the PARIS likelihood. Log-root and logit-fraction
random walks remain swappable follow-ups only after the smoke test measures
acceptance.

### Readiness gates

Before launching the real-data smoke:

1. cross-check the active target against the archived product-space formulas,
   including the root scaling-to-mass coordinate Jacobian and marginalization
   of inactive Beta coordinates;
2. add fixed offset and inferred outer coefficients to the state and target;
3. verify the full 14-slot compound schedule, every kernel, and exact restart
   across a mid-cycle boundary;
4. reject disconnected or zero-mass structural starts;
5. optimize frontier validation and mergeable-parent discovery without
   changing enumerated topology;
6. verify direct RHIME closure
   \(Gx=S(w\odot x)\), plus boundary and outer terms;
7. persist a strict durable checkpoint, canonical run manifest, and labelled
   trace suitable for segmented Slurm runs;
8. benchmark setup and 100--1,000 cycles at representative \(K\), recording
   per-kernel acceptance and throughput; and
9. retain the full NumPy state rebuild as the correctness oracle for later
   incremental prediction updates.

The initial HPC run is a wiring and acceptance/profile smoke, not a production
convergence run. Numba, incremental candidate predictions, alternative split
directions, correlated errors, and long posterior chains follow only after
these diagnostics.
