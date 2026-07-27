# MH-guided local partition search: S0 synthetic result

## Decision

The bounded S0 experiment **passed** at the predeclared factor-four budget.
Starting from the same data-conditioned deterministic basis, adding
Metropolis--Hastings (MH) partition moves substantially improved finite-run
recovery in both deliberately misaligned one-move cases and did not materially
degrade the aligned case.

This is evidence for a finite-budget stochastic local-search algorithm. It is
not evidence for convergence, correct Bayesian marginalization over
partitions, or inference of partition probabilities.

The immutable passing run is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_synthetic/e9e422fe3ab973898cffbd38df00b689efe212b8/harness-2d9dc06812ab0802a3723c4cb7ef6e66612106d791a924b5558b3f49570f7106
```

Source revision:
`e9e422fe3ab973898cffbd38df00b689efe212b8`.

## What was tested

The scientific state is a positive, dimensionless scaling field on a
\(2\times4\) native grid with \(K=4\) rectangular regions. The initial
partition \(P_0\) is the deterministic balanced `largest-nominal` basis; it is
not a random tiling or a draw from a structural prior.

Three planted truths were used:

1. `aligned`, with planted partition \(P_\star=P_0\);
2. `edge-one`, where \(P_\star\) is exactly one valid edge flip from \(P_0\);
3. `relocation-one`, where \(P_\star\) is exactly one valid resolution
   relocation from \(P_0\).

Within each \(P_\star\), truth is piecewise constant, positive, and normalized
to native-grid mean one. Four paired observation-noise replicates were run per
scenario.

The model used:

- \(T\sim\operatorname{Gamma}(4,4)\);
- a globally additive Dirichlet allocation with total concentration
  \(2K=8\);
- fixed independent Gaussian observation error,
  \(R=0.05^2I\), in scaling units;
- likelihood power one;
- no outer coefficients, boundary term, hierarchy, aggregation error, or
  inferred error parameter.

Training data comprised 16 noisy direct-cell observations: all eight native
cells observed twice in a fixed permutation. The primary score used ten
held-out adjacent-cell-average operators, each distinct from the direct
training rows, evaluated against their noiseless synthetic truth. These
held-out values were sealed from initialization, conditioning, proposals,
acceptance, and tuning.

## Samplers, iterations, burn, and thinning

For each paired replicate, a fixed-basis run first conditioned the continuous
state on \(P_0\). That exact state was then forked into fixed and mobile arms.

The passing factor-four profile used:

- 8,000 conditioning cycles, excluded from all reported finite-run averages;
- 20,000 production cycles per fixed, mobile, and oracle arm;
- all 20,000 post-cycle states retained;
- thinning interval one cycle;
- one root slice and five allocation-pair updates per cycle in both arms;
- two additional structural MH attempts per mobile cycle.

Thus a fixed production run used 120,000 transition slots and a mobile run
used 160,000. The finite-run estimator is the mean of all retained native
fields at complete cycle boundaries. It is a residence-weighted algorithm
output, not a claimed converged posterior expectation.

The conditional reference checks used four fixed-topology local chains, each
with 20,000 retained cycles after the same 8,000-cycle conditioning regime.
Their reference NUTS fits used four chains, 1,000 warm-up draws and 1,000
retained draws per chain, no thinning, diagonal mass, and target acceptance
0.9. The homogeneous factor-four profile lengthened the practical, oracle,
and local fixed-topology runs while reusing the already-passing NUTS
references.

## Primary results

The table reports medians across the four paired noise replicates. RMSE is in
dimensionless scaling units. The mobile/fixed ratios use the paired fixed arm
as denominator, so values below one favor MH-guided movement. “Oracle/fixed”
instead uses a separately labelled fixed run on the planted \(P_\star\) as
numerator.

| Scenario | Fixed held-out RMSE | Mobile held-out RMSE | Mobile/fixed held-out | Mobile/fixed native | Equal-wall mobile/fixed held-out | Oracle/fixed held-out |
|---|---:|---:|---:|---:|---:|---:|
| Aligned | 0.016685 | 0.016157 | 0.968952 | 0.984337 | 0.972202 | 1.003553 |
| Edge-one | 0.104861 | 0.015363 | 0.145762 | 0.138931 | 0.145498 | 0.142022 |
| Relocation-one | 0.065363 | 0.017333 | 0.268064 | 0.217191 | 0.267989 | 0.273480 |

All four mobile/fixed held-out ratios were below one in every scenario. The
aligned difference is continuous Monte Carlo variation: no structural move
was accepted there. The equal-wall comparison uses the longest complete
100-cycle prefixes fitting within common sampler time and preserves the
one-move advantage.

In every `edge-one` run, the mobile algorithm accepted exactly one edge flip,
first reached \(P_\star\) at cycle 12, 7, 5, or 12, and did not return to
\(P_0\). Its \(P_\star\) residence fraction was 0.99945--0.99980.

In every `relocation-one` run, it accepted exactly one relocation, first
reached \(P_\star\) at cycle 78, 146, 106, or 48, and did not return to
\(P_0\). Its \(P_\star\) residence fraction was 0.99275--0.99765.

Each mobile run made 40,000 structural attempts and witnessed valid proposals
of both move types. These traces show a successful posterior-targeted local
search followed by residence at the better partition; they do not show
round-trip structural mixing.

## Reference and correctness gates

The exact \(2\times3,\ K=4\) structural oracle passed:

- 54 valid moves with bit-exact topology, authoritative log-mass, and
  discrete-selection reverse recovery; continuous accounting agreed within
  the declared tolerance, with a maximum observed discrepancy of two ULP and
  accepted-flow equality differing by at most \(9.09\times10^{-13}\);
- 261 invalid proposals represented as explicit self-transitions.

That exact result applies only to the enumerated eight-tiling catalogue and
does not establish connectivity on the production tiling space.

All five factor-four local-versus-NUTS conditional references passed:

- zero NUTS divergences;
- worst \(\hat R=1.002571\), for `root_total`;
- minimum bulk ESS \(=2044.36\), for
  `leaf_mass[r0:1_c2:4]`;
- minimum tail ESS \(=1691.08\), for the same leaf mass;
- worst local MCSE/posterior-SD ratio \(=0.02943\), for `top_half`;
- worst half-window difference/posterior-SD ratio \(=0.06074\), for
  `top_right`;
- worst local-versus-NUTS tolerance use \(=0.60038\), for `bottom_right`.

The 5,000-cycle primary local checks had legitimately triggered the
predeclared homogeneous factor-four rescue: four of five conditional cells
failed either a half-window or MCSE gate. The rescue was authorized from
sealed conditional-reference evidence, not by inspecting the utility scores.
No further tuning grid was used.

## Operational history and integrity

An earlier source revision,
`ef27efd58c58afaa077d7b1b915a2d4498fbb751`, produced complete primary
scientific artifacts but stopped at the conditional launcher because the
authorization parent directory had not been created. Preserve that run as
incomplete evidence; it was neither promoted nor reused.

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_synthetic/ef27efd58c58afaa077d7b1b915a2d4498fbb751/harness-96bdfe118c6de4440efb26099ad83037ec227c3d8ad7cd706b9be9ad74d575f2
```

The repaired revision created only the authorization parent, preserving the
create-only token contract and fail-closed replay behavior. On the passing
run:

- the focused preflight passed 69 tests plus Ruff, formatting, and Pyright;
- all 48 factor-four Slurm tasks completed with exit code zero;
- no `*.failed.json` artifact exists;
- 75 index-referenced external hashes replayed with zero mismatches;
- aggregate and independent replay decisions and reports are byte-identical;
- root `complete.json` was written last.

The repaired-primary jobs were `18187568`--`18187576` and `18187661`.
Factor-four jobs were `18187702`, `18187704`--`18187708`, `18187803`, and
`18187805`.

Key hashes are:

| Artifact | SHA-256 |
|---|---|
| Frozen harness | `2d9dc06812ab0802a3723c4cb7ef6e66612106d791a924b5558b3f49570f7106` |
| Pixi lock | `4ed1244c33ffb7ef929bad73d8bd9944e49ed9b36b51fa05163b59b2a5b2f564` |
| Final index | `a67555ce935a712e66ff4610f70d5af5d369dfd275479116e93ae0dc72c930f0` |
| Decision | `2cef819c704f0d062cdb38dc09111fa08e230cf2d21ff4b9ba1dd059df1803ef` |
| Root completion | `cdeda8440bfd71119f0509529620ebc5be48a06d37b3d18665357103185491f8` |

## Interpretation

For these predeclared high-signal, one-move synthetic cases, MH acceptance
provided the intended posterior-targeting signal: the search rejected
structural movement when \(P_0\) was already aligned and accepted the single
useful local correction when it was not. The resulting finite-run field
averages were much better than holding the known-misaligned basis fixed and
closely tracked the planted-partition oracle.

The experiment does **not** establish:

- a converged partition posterior or valid marginalization over partitions;
- partition probabilities, structural uncertainty, or inference over \(P\)
  or \(K\);
- connectedness or irreducibility of the production tiling graph;
- uncertainty calibration or interval coverage;
- superiority over fixed bases beyond these four seeds and planted local
  truths;
- success for multi-move, larger-\(K\), atmospheric-footprint, real-data,
  correlated-error, hierarchical, or aggregation-error problems.

The next scientifically useful test is an atmospheric-like synthetic screen
with a reasonable deterministic starting basis and planted truths a small
number of witnessed legal moves away. It should retain the same fixed/mobile
common-state fork and oracle-learnability controls. A multi-move test should
be framed as search utility and path discovery, not as partition-posterior
mixing. S1 and S2 were not attempted in this run.
