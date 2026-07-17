# Dyadic Stochastic Local Search Hackathon Plan

## Status and relationship to the design note

This is the operational plan for extracting the dyadic prototype and producing
a reproducible stochastic local search (SLS) demonstration. The durable
scientific background, prototype inventory, exactness boundaries, and later
partition-inference design remain in
`docs/plans/dyadic_partition_inference.md`.

This plan is intentionally narrower and may be updated as implementation work
progresses. It records work packets, decisions, commands, expected artifacts,
acceptance criteria, and stop conditions. The first result is an optimizer and
basis initializer, not posterior inference.

### Implementation status

As of 17 July 2026, Phase 0A is implemented on
`codex/dyadic-sls-core`:

- exact unpadded rectangular dyadic trees and immutable partition frontiers;
- sum-preserving multiscale design columns and direct-sum parity checks;
- full Gaussian DFS with a partition-dependent covariance-builder boundary;
- explicitly labelled isotropic and historical quadratic benchmark scores;
- greedy, random, and threshold initializers;
- split, merge, and unique fixed-count paired proposals; and
- seeded stochastic local search with a piecewise geometric schedule.

The combined core gate is 49 focused tests plus Ruff and Pyright. The stacked
demo branch adds the TAC/MHD adapter, real-data checks, run manifest, static
figure, trace, and GIF. No production basis interface or `fixedbasisMCMC` code
is touched.

### First executed TAC/MHD result

The stacked `codex/dyadic-sls-demo` branch now contains a reproducible local
run using the committed one-day TAC/MHD fixture:

- sum-preserving factor-8 coarsening: `(47, 293, 391)` to `(47, 37, 49)`;
- fixed `K=32`, seed `20260717`, 100 discarded pilot proposals, and 300 SLS
  evaluations;
- fixed benchmark covariance `diag(max(error, min_error)**2)`, where
  `min_error` supplies a minimum-mismatch floor rather than the inferred
  production total-error process and may understate production total error;
- fine-cell support counts propagated through partial coarsening blocks and
  aggregated for the proxy initializer;
- explicitly benchmark-only isotropic region covariance with `tau=1`; and
- Gaussian benchmark DFS improvement from `6.6213868511` to `6.6969787847`.

Separate diagnostic recomputations also improved for MHD (`2.2306430146` to
`2.2599799610`) and TAC (`4.7197859243` to `4.7464773740`). These site scores
are not additive contributions to combined DFS. The result demonstrates that
the extracted stochastic search can improve its proxy-informed initializer; it
does not establish posterior validity or resolve the covariance-projection
limitation.

Recorded artifacts live in `docs/plans/figures/dyadic_sls/`: the manifest,
bounded CSV trace, static initial/best comparison, and 120-frame GIF.

## Demonstration objective

Produce a runnable example that:

1. constructs a canonical dyadic partition over the EUROPE test grid;
2. builds observation-by-grid contributions from committed TAC and MHD data;
3. starts from a useful greedy or bisection partition rather than one root;
4. improves a declared Gaussian design score with reproducible split/merge SLS;
5. reports the scientific score, region count, temperature, acceptance, and
   best-so-far state separately;
6. emits a static comparison, a bounded diagnostics table, and a GIF; and
7. leaves reusable partition, multiscale-design, move, and test infrastructure
   for the later collapsed partition sampler.

The minimum scientific result is a fixed-region-count comparison. A
variable-count search and complexity trade-off are secondary.

## Scope decisions

### Primary search

Use a fixed region count, initially \(K=64\), and a paired move that merges one
active sibling pair and splits one compatible active leaf. This avoids mixing
an arbitrary region penalty into the principal objective and directly exercises
the move structure needed by the planned fixed-\(K\) partition sampler.

The primary utility is the full Gaussian degrees of freedom for signal:

\[
D(P)
=\operatorname{tr}\left[
B_PH_P^T(H_PB_PH_P^T+R)^{-1}H_P
\right].
\]

For computation with diagonal \(R\), use

\[
A_P=B_P^{-1}+H_P^TR^{-1}H_P,
\qquad
D(P)=K-\operatorname{tr}(A_P^{-1}B_P^{-1}),
\]

evaluated by stable Cholesky solves. The two formulas must agree in small
tests.

The first real-data run may use \(B_P=\tau^2I_K\), but the manifest and figures
must call this an equal-region-multiplier-uncertainty benchmark. The value of
\(\tau\) and the construction of \(R\) are explicit inputs, not hidden
defaults. Sensitivity to at least two plausible \(\tau\) values should be
reported before making scientific claims.

This benchmark does not preserve the covariance transformation used by
Bocquet, Wu, and Chevallier. Their scale-consistent representation starts from
a fine-grid covariance \(B\) and projects it for each partition:

\[
B_P=PBP^T,
\]

where \(P\) denotes the representation projection. Reusing one numerical
covariance form for every \(P\) breaks that assumption. Keep \(B_P\)
construction behind an explicit partition-dependent callable from the first
implementation so a projected covariance can be substituted later without
rewriting the objective or search loop.

At fixed \(K\), SLS accepts improvements and may accept losses using

\[
\Pr(\text{accept loss }L)=\exp(-L/T),
\qquad
L=\max(0,D(P)-D(P')).
\]

This is optimizer behavior. It is not an MH posterior transition, even if the
move implementation also exposes forward and reverse candidate probabilities.

### Secondary scores

Do not combine every diagnostic into one objective. Record:

- combined TAC/MHD full DFS, used by the primary search;
- TAC-only and MHD-only DFS recomputed for diagnostics;
- region count \(K\);
- current and best objective;
- temperature and rolling acceptance;
- optional cumulative split-contrast DFS/EIG diagnostics;
- optional projected-flux compression diagnostics;
- region depth, area, and aspect-ratio summaries; and
- evaluation count and runtime.

TAC-only and MHD-only DFS do not generally add to combined DFS. Label them as
separate recomputations.

The historical score

\[
s_v=\frac{1}{a_v}\sum_i p_i h_{iv}^2,
\qquad
h_{iv}=\sum_{c\in v}G_{ic},
\]

should be retained as a provenance and performance comparison. Here \(p_i\)
must mean observation precision \(1/\sigma_i^2\), and \(a_v\) is an explicitly
documented support/area normalization. Until its prior, prolongation, and
aggregation-error assumptions are reconciled with the Bocquet construction,
call it the **prototype quadratic design score**, not exact DFS.

The existing `split_contrast_score` is another useful diagnostic. It must not
double-apply flux: either pass footprints as the contribution and flux/prior
mass as `cell_weight`, or pass footprint-times-flux contributions with unit
cell weights under an explicitly different coefficient convention.

### Optional variable-count search

After the fixed-\(K\) run works, allow single split or merge moves and optimize

\[
U_\lambda(P)=D(P)-\lambda K(P).
\]

Show \(D(P)\), \(K(P)\), and \(\lambda K(P)\) separately and prefer a small
\(\lambda\) sweep or DFS-versus-\(K\) Pareto plot over selecting one opaque
combined metric. Do not call \(\lambda K\) a posterior prior term.

### Canonical support

Use one deterministic binary dyadic tree:

- tiles have immutable bounds and stable IDs;
- children exactly and disjointly cover their parent;
- split the longer axis, with one documented tie rule for square tiles;
- a partition is an immutable active-leaf frontier;
- a valid partition covers every included grid cell exactly once; and
- only true siblings may merge.

This is less flexible than the historical independent x/y interval graph but
avoids duplicate geometric states and is reusable by exact partition inference.
Flexible split orientation remains an optimizer experiment until equivalent
partition histories are canonicalized.

## Local TAC/MHD data contract

### Committed inputs

Use only repository-owned test data for the first real run:

| Input | File | Relevant content |
| --- | --- | --- |
| Prepared two-site rows | `tests/data/frozen_mhd_tac_make_inv_inputs_hbmcmc.npz` | 47 production-aligned times, sites, observations, errors, and minimum errors |
| TAC footprints | `tests/data/footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc` | `fp(lat, lon, time)`, 168 hourly rows |
| MHD footprints | `tests/data/footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc` | `srr(time, latitude, longitude)`, 168 hourly rows |
| Flux | `tests/data/flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc` | annual `flux(lat, lon)` |
| TAC processed reference | `tests/data/merged_data_test_tac_combined_scenario_v14.nc` | 24-hour production-path comparison |
| Raw observations | `tests/data/obs_*_ch4_*.nc` | optional full-week preparation validation |

The recommended first route uses the frozen one-day rows:

- 23 MHD rows from 1 January 01:00 through 23:00;
- 24 TAC rows from 1 January 00:00 through 23:00;
- site-major order, MHD followed by TAC; and
- prepared `error` and `min_error` vectors in ppb.

This gives an exact, deterministic two-site integration fixture without
reimplementing OpenGHG hourly observation aggregation. The full local week can
later provide 333 rows, but that requires reproducing the production averaging
semantics and is not on the first critical path.

The one-day design has rank at most 47. A (K=64) visualization can still test
the optimizer, but it cannot imply that all 64 regional directions are
independently resolved. Report the achieved DFS against this rank bound and
include a (K=32) sensitivity run if time permits.

The footprint grids are exactly equal. Flux coordinates differ from footprint
coordinates by at most approximately \(7.7\times10^{-6}\) degrees. The loader
must verify equal shapes and `allclose(..., rtol=0, atol=1e-4)`, then adopt one
canonical footprint grid. It must not rely on an accidental exact xarray join.

### Preparation order

1. Load frozen times, site indicators, observations, errors, and minimum errors.
2. Normalize MHD names to `fp`, `lat`, and `lon` and transpose both footprint
   arrays to `(time, lat, lon)`.
3. Select each site's footprint rows using its exact frozen times.
4. Load the annual flux, drop its singleton time dimension, validate coordinate
   tolerance, and adopt the footprint coordinates.
5. Form

   \[
   G_{i,c}=\mathrm{footprint}_{i,c}\,\mathrm{flux}_c\,10^9
   \]

   in ppb per unit fine-cell multiplier.
6. Concatenate MHD and TAC in the frozen site-major order, retaining site and
   time coordinates.
7. Apply any row mask simultaneously to observations, errors, minimum errors,
   \(G\), and optional fixed design blocks.
8. Only then coarsen, pad, construct multiscale columns, or compute scores.

The expected fine-grid array has shape `(47, 293, 391)`. Its row sums can be
checked against the frozen prepared sensitivity row sums. TAC day-one
`fp_x_flux` can also be checked directly against the merged NetCDF reference.

Observed mole-fraction values are not inputs to DFS or the SLS objective. They
remain useful for later synthetic-observation or posterior demonstrations. Do
not fit the real total mole fractions with emissions enhancement \(G\) alone;
that would omit the background/boundary contribution.

For the first covariance benchmark, compare and declare one of:

- \(R=\operatorname{diag}(\mathrm{error}^2)\); or
- \(R=\operatorname{diag}(\max(\mathrm{error},\mathrm{min\_error})^2)\).

The second is a fixed minimum-mismatch-floor benchmark. It is not the same as
RHIME's inferred model-error process and may understate production total error;
"conservative observation covariance" is therefore not an appropriate label.

The first one-day demo should apply no additional scientific filter. If filters
are added later, build the row mask before \(G\), multiscale columns, or scores,
and verify that perturbing an excluded row cannot alter the result.

### Sum-preserving reduction

The native grid is 293 by 391 cells. Coarsen contributions by spatial **sum**,
never mean. The initial candidate is factor-4 coarsening, giving a 74 by 98
grid. The implemented canonical tree supports this exact rectangle, so the POC
does not pad to a power-of-two square.

Carry a coarsened physical-support count or area field so:

- partially supported tiles use declared physical support in normalization;
- rendering clips to the physical grid; and
- direct fine-grid and coarsened total-contribution identities can be tested.

Benchmark factor 4 and factor 8 in a short dry run. Prefer factor 4 if candidate
columns, one DFS evaluation, and a 500-step search fit comfortably within the
event memory/runtime budget; otherwise report and use factor 8.

## Phase 0A: demo-enabling extraction

Phase 0A is the minimum reusable implementation needed by the SLS demo.

### Package layout

Keep provisional code isolated and do not re-export it from
`openghg_inversions.basis`:

```text
openghg_inversions/basis/experimental/
    __init__.py
    dyadic/
        __init__.py
        tree.py
        state.py
        multiscale.py
        objectives.py
        initializers.py
        proposals.py
        search.py
examples/basis/
    dyadic_sls_demo.py
tests/basis/experimental/
    test_dyadic_tree.py
    test_dyadic_multiscale.py
    test_dyadic_objectives.py
    test_dyadic_search.py
```

Responsibilities:

- `tree.py`: tile/node IDs, bounds, parent/children, canonical orientation,
  sibling checks, and exact rectangular support.
- `state.py`: immutable active frontier, exact-cover validation, stable
  ordering, split/merge application, label and boundary rendering data.
- `multiscale.py`: sum-preserving coarsening, candidate observation columns,
  active-column gathering, and direct-aggregation parity helpers.
- `objectives.py`: full Gaussian DFS, explicit partition-dependent covariance
  construction, prototype quadratic score, and structured
  objective/diagnostic results.
- `initializers.py`: canonical greedy, threshold/bucket, random, and optional
  quadtree-style starts.
- `proposals.py`: immutable split, merge, and paired moves; enumeration,
  apply/reverse, candidate counts, and optional `log_q`.
- `search.py`: schedules, seeded SLS runner, bounded trace callbacks, current
  and best states, stop reasons, and manifests.
- `dyadic_sls_demo.py`: local test-data loading, configuration, plotting, CSV
  and GIF output. Core modules perform no file loading or plotting.

### Prototype extraction map

Use the cleaned branch `codex/basis-prototype-examples` at commit `b6ce565` as
a structural scaffold, not as the scientific objective:

- `Tile`, padding, multiscale-axis, label, and loop scaffolding are useful;
- replace scalar pre-summed weights with observation-space candidate columns;
- discard the toy `_tile_information_score` and `_tiling_energy`;
- strengthen merge checks to require true dyadic siblings;
- replace mutable lists/dense indicators with an immutable partition;
- replace the global RNG with a caller-supplied generator; and
- replace the fixed-temperature result with a complete bounded trace.

Preserve the scientific operation from
`~/Documents/basis_functions/basis_fn_ipython_hist_14aug.py`:

\[
h_v=\sum_{c\in v}G_{:,c},
\]

then score \(h_v\). Do not pre-score fine cells and sum scalar scores.

Record exact historical symbol and line provenance in the experimental package
documentation at extraction time. The current repository implementations of
axis-parallel, inertial, priority-queue, and contrast logic should be adapted
where compatible rather than copied from `~/Documents/inversions`.

### Required tests

1. Candidate tile columns equal direct sums of \(G\) over tile cells.
2. A counterexample proves sum-then-square differs from summing cell scores.
3. Gathered \(H_P\) equals direct fine-grid aggregation.
4. Every initial and proposed partition is an exact active frontier.
5. Split followed by the corresponding merge restores identical state.
6. Adjacent non-sibling rectangles cannot merge.
7. Moves and search do not mutate their inputs.
8. Incremental prototype-score changes equal full recomputation.
9. The two Gaussian DFS formulas agree on a small positive-definite case.
10. Leaf ordering changes neither DFS nor rendered coverage.
11. Fixed-\(K\) paired moves preserve \(K\).
12. Seeded SLS produces an identical trace and never loses the best initializer
    score.
13. Zero temperature rejects losses; positive temperature can accept a
    controlled loss.
14. Invalid variance/precision, support, coordinate, and shape inputs fail
    clearly.
15. Partial boundary blocks preserve their physical support and total design
    contribution.

### Local integration tests

1. Assert frozen site counts, ordering, endpoint times, and positive errors.
2. Reject grids outside tolerance and explicitly accept the known flux offset.
3. Assert TAC \(G\) agrees with merged `fp_x_flux`.
4. Assert fine-grid \(G\) row sums agree with frozen prepared sensitivity row
   sums within a declared floating tolerance.
5. Assert every gathered \(H_P\) column equals the corresponding direct cell
   sum.
6. Apply a row mask and assert every observation-aligned array remains
   synchronized.
7. Perturb an excluded row and verify multiscale columns, scores, and seeded SLS
   output are unchanged.

## Phase 0B: deferred consolidation

These items are useful but not on the critical path to the first TAC/MHD GIF:

- parity-tested Numba threshold bisection;
- compile-time versus warm-call benchmarks;
- full-week OpenGHG-compatible observation preparation;
- arbitrary x/y split histories;
- masks and one tree per land/ocean, country, or inner/outer group;
- production data adapters;
- checkpoint/restart;
- public API naming and exports; and
- performance-oriented incremental full-DFS updates.

Promote an item from 0B only if profiling shows it blocks the real demo or it is
needed to avoid implementing semantics that will immediately be discarded.

## SLS demo work packets

### D0: data and score dry run

- Load frozen TAC/MHD rows, raw footprints, and flux.
- Confirm shapes, row order, grid tolerance, units, and memory.
- Validate \(G\) against the merged TAC and frozen row-total references.
- Build factor-4 and factor-8 \(G\) arrays.
- Time one full DFS evaluation at \(K=64\).
- Freeze \(\tau\), covariance choice, coarsening, and target \(K\) in a run
  manifest.

**Gate:** no search implementation should silently compensate for unresolved
units, double flux weighting, or an infeasible DFS cost.

### D1: synthetic reference

- Build a 4 by 4 and an 8 by 8 synthetic contribution field.
- Exercise tree/state, initializers, moves, scores, and SLS.
- Compare full DFS with an independent implementation.
- Save one small deterministic trace used by tests.

**Gate:** all algebraic and state-invariant tests pass before the real run.

### D2: initializers

Implement starts that lie in the same canonical support:

1. canonical greedy: repeatedly split the feasible leaf with greatest declared
   score gain;
2. canonical bucket/threshold bisection;
3. canonical quadtree-style growth represented by two binary levels;
4. random valid growth as a control.

The existing production bucket/quadtree label maps are useful external
baselines but cannot automatically seed the canonical tree if their rectangles
are outside its support. Do not silently relabel or approximate them. Record any
deterministic repair needed to reach exactly \(K\).

The minimum real run requires greedy plus random. Bucket and quadtree are
strongly preferred comparison starts but may follow the first working GIF.

### D3: search schedule

Use a discarded pilot over valid paired moves from all initializer families:

1. collect positive DFS losses from 100-200 proposals;
2. choose \(T_0\) so the median sampled loss has approximately 0.8 acceptance;
3. choose \(T_f\) so it has approximately 0.01 acceptance;
4. freeze the schedule before comparison runs;
5. spend 10% of evaluations at \(T_0\), 80% geometrically cooling to \(T_f\),
   and 10% at zero temperature for polishing.

Run equal evaluation budgets and at least three seeds per available initializer.
Five seeds are preferred for the final comparison. Include zero-temperature
hill climbing and no-search baselines.

If different starts remain in different basins, report that result rather than
continuing to tune temperature indefinitely.

### D4: visualization

Generate:

```text
docs/plans/figures/dyadic_sls/
    tac_mhd_sls.gif
    tac_mhd_sls_summary.png
    tac_mhd_sls_trace.csv
    tac_mhd_sls_runs.csv
    tac_mhd_sls_manifest.json
```

The static summary is the scientific result; the GIF explains the search.

Each frame should contain:

- partition boundaries over a fixed combined sensitivity/design background;
- the most recent accepted split and merge highlighted for one frame;
- current and best DFS traces;
- region count, with the fixed target shown;
- temperature and rolling acceptance;
- iteration, accepted-move count, initializer, seed, and run ID; and
- a concise statement that the objective is Gaussian benchmark DFS.

Use stable boundaries or depth shading rather than recoloring regenerated
integer region labels, which causes flicker. Keep color limits fixed.

Capture the initial state, every new best, every tenth accepted move or every
1% of evaluations, and final/best states. Cap the GIF near 120-150 frames at
8-12 frames per second. Matplotlib's Pillow writer should avoid a new direct
animation dependency. Preserve the diagnostics CSV independently of the GIF.

Tests should verify rendering with the non-interactive `Agg` backend and a few
frames, not compare GIF bytes.

## Acceptance criteria

Phase 0A and the SLS demo are complete when:

- core code has no dependency on private prototype paths or OpenGHG retrieval;
- provenance points from retained behavior to repository-owned symbols/tests;
- all partition and direct-aggregation invariants pass;
- the fixed-\(K\) search starts and ends at exactly \(K=64\);
- full DFS is evaluated under explicit \(B_P\) and \(R\);
- repeated runs from the same manifest reproduce proposals, accepted states,
  best state, and numeric diagnostics;
- combined, TAC-only, MHD-only, and best-so-far diagnostics are distinguishable;
- greedy/random and no-search/hill-climbing controls receive equal budgets;
- the best state is never worse than its initializer;
- runtime and peak-memory summaries are recorded;
- the static figure and GIF identify assumptions and do not use posterior or
  convergence language; and
- focused pytest, Ruff, and configured type checking pass.

## Stop and fallback conditions

Stop and fix correctness rather than polishing output if:

- candidate columns differ from direct grid sums;
- incremental and direct score changes disagree;
- any state fails exact-cover or ancestry invariants;
- TAC/MHD coordinate or row alignment is ambiguous;
- full DFS formulas disagree;
- partial boundary blocks affect total design contribution incorrectly;
- fixed-\(K\) moves cannot connect the demonstrated support; or
- deterministic replay changes.

Narrow transparently if the real run is too expensive:

1. increase sum-preserving spatial coarsening from factor 4 to factor 8;
2. reduce the evaluation budget;
3. reduce \(K\); then
4. use the prototype quadratic score for the animation while retaining sparse
   full-DFS checkpoints, clearly labelling the distinction.

Do not silently replace full DFS with a proxy or add Numba before profiling
identifies the bottleneck.

## Commit and review sequence

Develop on one working branch with logical commits so it can be split later:

1. operational plan and source-provenance update;
2. tree/state/multiscale core with synthetic tests;
3. objectives, initializers, moves, and deterministic SLS tests;
4. local TAC/MHD preparation and real-data integration test;
5. demo script, static report, GIF, and recorded run manifest; and
6. optional performance or initializer comparisons.

The natural review split is:

- **PR A:** experimental Phase 0A core and synthetic tests, with no atmospheric
  file loading;
- **PR B:** local TAC/MHD adapter, demo, diagnostics, and generated evidence;
- **PR C:** deferred Numba/benchmark/generalization only if justified.

PR B may be developed against PR A for the hackathon, but current production
basis interfaces and `fixedbasisMCMC` must remain untouched.

## Effort forecast

- Small synthetic SLS using extracted structure: approximately 1 focused day.
- First technically credible TAC/MHD fixed-\(K\) run and static figure:
  approximately 2-4 focused days.
- Multi-initializer comparison, robust replay, and polished GIF:
  approximately 4-6 focused days total.
- Full Phase 0 including Numba, generalized support, benchmarks, checkpointing,
  and masks: approximately 6-9 focused days.

A faster first animation is possible with the prototype quadratic score, but it
must not be presented as calibrated DFS. The recommended order is correctness
and a static fixed-\(K\) result first, then GIF polish.

## Deferred connection to partition inference

The following Phase 0A work transfers directly to the collapsed fixed-\(K\)
sampler:

- canonical tree and immutable partition state;
- active multiscale-column gathering;
- full Gaussian DFS and collapsed-target linear algebra;
- paired split/merge neighbor enumeration;
- stable node ordering;
- deterministic initializers;
- manifests and trace diagnostics; and
- exhaustive small-state/state-invariant tests.

The SLS acceptance rule and cooling schedule do not transfer into the posterior
target. They remain useful for initialization and for constructing an informed,
fully normalized proposal whose Hastings correction is added later.
