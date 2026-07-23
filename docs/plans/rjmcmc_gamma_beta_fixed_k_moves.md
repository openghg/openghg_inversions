# Gamma--Beta fixed-\(K\) topology-move experiment

## Status

This document records the experiment that follows the native-data Stage C run
of the fixed-direction Gamma--Beta RJMCMC baseline. It is both an implementation
plan and the restart point for future compacted Codex sessions.

- Base branch: `codex/rjmcmc-gamma-beta-baseline`
- Implementation branch: `codex/rjmcmc-gamma-beta-fixed-k-moves`
- Frozen Stage C code revision: `dd687b92abb86ce0080a1c8a713f3eb9a57df3aa`
- Frozen Stage C report revision: `9528f86c0620d45689355b4989948ee4f4113775`
- Frozen input SHA-256:
  `24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044`
- Frozen problem SHA-256:
  `c0cf24065afd567a1647767d697465b5aee21163f0ef39bb8236232780330d46`

The model remains experimental. No result should be written to the production
`PARIS_inversions` archive until the declared convergence gates pass.

## Stage C result and diagnosis

The completed HPC test established software integrity but not posterior
convergence:

- all 40 durable segments completed and passed checkpoint/hash validation;
- restart continuation was byte-identical;
- peak memory was approximately 1.64 GB;
- low-\(K\) starts remained in \(K=36\)--\(115\);
- high-\(K\) starts remained in \(K=192\)--\(285\);
- rank-normalized \(\widehat R(K)=2.886\);
- all 29 monitored summaries failed at least one convergence threshold.

The structural acceptance rates were high, so the immediate failure is not a
simple rejected-proposal problem. Stage C supplied only 2,000 local
split/merge opportunities per chain. A nearest-neighbour random walk needs a
number of accepted structural steps proportional to the square of the
distance it must traverse. Separation of roughly 200 regions is therefore not
surprising after this short mobility run.

Fixed-\(K\) moves cannot change \(K\) directly. Their purpose is to relax the
partition conditional on \(K\), reduce dependence between geometry and the
next split/merge decision, and test whether the dimension-changing walk is
being slowed by locally trapped resolution patterns. The experiment must not
claim that a fixed-\(K\) move alone solves diffusion in \(K\).

## State-space decision

### What remains fixed

The comparison retains the current model exactly:

- one `CanonicalDyadicTree`, with longer-axis splits and row tie-breaking;
- a `DyadicFrontier` that is a pruning of that tree;
- the same normalized \(p(K)/N_K\) structural prior;
- root total \(T\), \(K-1\) active Beta fractions, and fixed outer
  coefficients;
- the fixed concentration \(\kappa=2\) used in Stage C;
- the same frozen PARIS observations, sensitivity, boundary offset, weights,
  and likelihood.

The target and irreducible checkpoint coordinates do not change.

### Why a literal edge flip is not added here

A Cannon--Levin--Stauffer edge flip replaces a shared bisector by its
perpendicular bisector in the state space of full equal-area dyadic leaf
tilings. The current tree assigns each rectangle exactly one child pair.
Almost every perpendicular endpoint is therefore absent from
`CanonicalDyadicTree` and cannot be represented by a `DyadicFrontier`.

Calling a fixed-tree move an edge flip would be misleading. A literal flip
requires a separate leaf-tiling state, a new structural prior and tiling
counter, dynamic rectangle design aggregation, new checkpoint coordinates,
and an allocation law valid across alternative decompositions.

Fixed \(\kappa\) is especially important here. Its product of node-specific
Beta densities is tied to a selected decomposition. If two split trees encode
the same leaf tiling, copying that density to both encodings either changes the
tiling prior through representation multiplicity or requires an explicit
correction and cross-chart Jacobian. An additive-\(\alpha\)
Gamma/Dirichlet allocation is the cleaner order-independent choice for a
future full-tiling model.

The literature boundary is:

- [Cannon, Levin, and Stauffer (2017)](https://doi.org/10.4230/LIPIcs.APPROX-RANDOM.2017.34)
  prove polynomial mixing for a lazy local edge-flip chain on uniform,
  fixed-size, equal-area dyadic tilings. The theorem does not cover adaptive
  unequal-area frontiers, an atmospheric likelihood, or changing \(K\).
- [Janson, Randall, and Spencer (2002)](https://randall.math.gatech.edu/r-rsady.pdf)
  give a unique canonical tree representation of each equal-area tiling and a
  separate chain with rotations of sub-tilings at all scales. Their
  canonicalization directly supports the requirement that split ordering must
  not alter the leaf-tiling law.
- [Angel et al. (2012)](https://arxiv.org/abs/1107.2636) provide recursive
  tiling, friend-pair, chain, and binary-coordinate structure. They do not
  define or analyse an MCMC transition.

The native \(183\times128\) grid is not a strict equal-area dyadic square:
183 is odd. A future literal edge-flip prototype must therefore declare an
adaptive midpoint-rectangle family and cannot inherit the published mixing
bound.

## New fixed-tree kernels

### Resolution relocation

Let \(F\) be the current frontier. Select a mergeable cherry parent \(a\)
uniformly, merge it to obtain the intermediate frontier \(G\), then select a
different splittable leaf \(b\in S(G)\setminus\{a\}\) uniformly and split it.
This preserves \(K\) while moving one unit of resolution.

The forward proposal density is

\[
q(F',\rho_b'\mid F,\rho_a)=
\frac{1}{|C(F)|}
\frac{1}{|S(G)\setminus\{a\}|}
g_b(\rho_b'),
\]

where \(g_b\) is the normalized Beta prior density. The pointwise reverse
merges \(b\), returns to the same \(G\), splits \(a\), and evaluates
\(g_a(\rho_a)\). The two intermediate split-candidate counts are equal, but
the source and candidate cherry counts need not be.

In the declared root-plus-active-fraction reference measure, the move deletes
one fraction coordinate and inserts one auxiliary fraction coordinate. Its
augmented Jacobian is one. At fixed \(K\), the partition-prior ratio is one;
using the Beta prior as the auxiliary proposal cancels the changed fraction
prior. The remaining non-cancelling terms are the likelihood change and the
source/candidate cherry-degree correction.

The trivial merge-then-resplit of \(a\) is excluded. Frontiers with no legal
pair retain the opportunity as an explicit self-transition.

### Bounded subtree retile

Select one of the \(K-1\) active split nodes uniformly as a block root. Let the
current block contain \(m\) frontier leaves.

- If \(m\) exceeds the configured cap, or the canonical subtree has no other
  \(m\)-leaf frontier, retain an explicit self-transition.
- Otherwise draw a different \(m\)-leaf frontier uniformly using exact
  arbitrary-precision subtree counts and deterministic rank/unrank logic.
- Keep the block total, outside fractions, and fraction coordinates common to
  both frontiers.
- Draw fractions at newly active split nodes from their normalized Beta
  priors; the reverse density evaluates the removed source fractions.

The block-root selection probability is \(1/(K-1)\) in both states. The
conditional alternative count is also the same in both directions. Beta
target/proposal terms cancel, the partition prior is unchanged, and the
augmented coordinate swap has unit Jacobian. The non-cancelling term is the
likelihood change.

This is a restricted fixed-tree analogue of multiscale block dynamics, not a
full-tiling rotation. The initial cap is 8 leaves and is persisted as part of
the kernel identity.

## Schedule and durable schema

The v1 14-slot cycle remains reproducible on its base branch. The new schedule
uses 16 atomic slots:

1. two mixed split/merge RJ opportunities;
2. one resolution-relocation opportunity;
3. one bounded subtree-retile opportunity;
4. one root-total refresh;
5. five active-fraction refreshes;
6. one deterministic update of each of the six fixed outer coefficients.

This preserves every Stage C proposal-opportunity count and adds the two
fixed-\(K\) opportunities. Comparisons should match cycles, not raw atomic
transitions.

Planned schedule ID:

`gamma_beta_2_rj_1_relocate_1_subtree_1_root_n_fraction_fixed_sweep_v2`

Persistence changes:

- checkpoint schema 1 to 2;
- manifest schema 2 to 3;
- trace schema 1 to 2;
- persist relocation slots, subtree slots, and subtree leaf cap;
- add a secondary node ID and block leaf count to every-attempt diagnostics;
- reject v1 checkpoints under the v2 sampler rather than attempting RNG-phase
  migration.

The scientific problem fingerprint and state fingerprint remain unchanged
because the posterior target and irreducible state have not changed.

## Correctness plan

### Proposal oracles

- Enumerate every tiny fixed-tree frontier at fixed \(K\).
- Check that every relocation endpoint has its declared pointwise reverse.
- Verify unequal cherry-degree corrections.
- Verify fixed \(K\), conserved root total, immutable sources, fraction
  insertion/deletion identity, and unit augmented Jacobian.
- Cross-check subtree counts and rank/unrank against exhaustive enumeration.
- Check uniform conditional subtree draws.
- Verify common and outside fractions remain unchanged.
- Construct exact prior-only transition matrices and check row sums, detailed
  balance, and the normalized fixed-\(K\) stationary law.

### Sampling and persistence

- Permit singleton positive \(p(K)\) support when at least one fixed-\(K\)
  topology slot is present.
- Verify the exact 16-slot opportunity sequence.
- Consume one acceptance uniform even for invalid topology opportunities.
- Check seeded replay and every-phase in-memory continuation.
- Check durable uninterrupted-versus-split byte identity.
- Round-trip all new settings and attempt metadata through NetCDF.
- Reject schedule, schema, problem, and manifest mismatches explicitly.

### Regression suite

- focused proposal/tree/sampler/I/O tests;
- full experimental RJMCMC tests;
- focused Ruff and Pyright;
- repository `tox -p` before review or HPC handoff.

## HPC experiment

### Stage A: software replay

Use the frozen native input and run:

- one 16-transition cycle;
- an awkward 5+11 continuation split;
- a process-boundary durable restart;
- one segment with zero retained states.

Require byte-identical continuation, matching hashes, finite target
decomposition, and exact slot counts.

### Stage B: fixed-\(K\) topology controls

Run multiple chains at fixed \(K=50\) and fixed \(K=250\), first prior-only and
then with the full likelihood. Compare:

1. relocation only;
2. relocation plus subtree retile.

These runs isolate conditional topology mobility from diffusion in \(K\).
Report acceptance by move, unique frontiers, frontier overlap, active-node
occupancy, topology-indicator ESS, changed-cell fraction, prediction
displacement, immediate reversals, and throughput.

### Stage C: matched variable-\(K\) comparison

Repeat the existing four starts \(50,250,50,250\) with the same number of
cycles, priors, likelihood, retention, and frozen input. The new run has 16
atomic transitions per cycle but the same number of split/merge, root,
fraction, and fixed-coefficient opportunities as the 14-slot baseline.

The original 1,000-cycle Stage C run is a software/mobility screen, not a
convergence-length experiment. If integrity passes, extend the comparison
enough to observe several plausible low-to-high and high-to-low \(K\)
traversals before interpreting \(\widehat R\) or ESS.

Primary comparison outcomes:

- \(K\) round trips and rank-normalized \(\widehat R\);
- overlap of chain-specific \(K\) distributions;
- topology mixing conditional on \(K\);
- posterior prediction and outer-coefficient diagnostics;
- accepted structural displacement per wall hour;
- peak memory and transition throughput.

No pooled concentration or flux estimate is defensible unless the convergence
gates in the existing HPC plan pass.

## Future full-tiling track

The literal paper-derived experiment remains separate:

1. represent the scientific state as a canonical set of leaf rectangles;
2. enumerate the seven order-two equal-area tilings without counting
   decomposition histories;
3. reproduce the Cannon rectangle-side proposal and its constant
   \(1/(2K)\) off-diagonal probability;
4. add friend-pair discovery and all-scale compatible block moves;
5. define an adaptive native-grid structural prior;
6. use an order-independent additive-\(\alpha\) allocation, or derive and
   test the fixed-\(\kappa\) canonical-chart density and Jacobians explicitly;
7. only then attach the atmospheric likelihood.

This track should not be hidden behind the fixed-tree API: its state,
reference measure, geometry catalogue, target, and persistence boundary are
different.

