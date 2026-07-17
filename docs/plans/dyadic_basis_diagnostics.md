# Dyadic Basis Diagnostic Experiments

## Purpose

This operational note follows the mathematical background in
`docs/reports/rhime_bocquet_reduced_gaussian.md`. It separates three questions
that were confounded in the first SLS demonstration:

1. does stochastic local search find a good partition in its own dictionary;
2. does that dictionary compare well with an existing basis algorithm; and
3. is the TAC/MHD input and error model suitable for an inversion?

All work remains under `basis.experimental.dyadic` or `examples/basis`. It does
not change production RHIME, `run_hbmcmc.py`, or `fixedbasisMCMC`.

## Terminology correction

The current demo's command-line search block width of 8 means an
\(8\times8\) sum-preserving spatial coarsening. The native 293 by 391 grid is
represented by a 37 by 49 search grid, with partial edge blocks retained.

This is distinct from all of the following:

- the private prototype name `dyadic8`, which means dyadic depth 8 and a
  \(2^8\times2^8\) finest padded grid;
- the private spatial precomputation, whose tensor-product dyadic array is
  about four times the padded finest spatial grid, or about 9.1 times the
  unpadded 293 by 391 grid after padding to 512 squared; and
- Bocquet, Wu, and Chevallier's up-to-eightfold storage statement for a fully
  precomputed hierarchy that is multiscale in two spatial dimensions and time.

Future reports should say **8x8-block search grid** rather than only
**factor-8**.

## Prototype search provenance

The private stochastic search did not calibrate temperature to a 50% overall
acceptance rate. It used fixed-temperature chunks, often 10,000 or 100,000
iterations, and manually explored temperatures around 0.001, 0.002, 0.005,
0.01, and sometimes 0.02 under its historical quadratic tile score.

The main target was approximately 250 regions. The above-target linear penalty
was also hand tuned against observed successful split gains. Recorded values
include 0.58 for the depth-8 padded problem and roughly 0.10 to 0.15 for the
depth-9/full-domain problem. The final transition logic approximated
\(\lambda\max(0,K-250)\), but missed the reward on the threshold-crossing
merge from 251 to 250. This provenance supports tuning a penalty on the scale
of local gains, but the exact prototype rule should not be copied as a
probabilistic prior.

The current repository calibration is newer: it takes the median positive
pilot loss \(L\) and sets \(T=-L/\log p\). A setting of `p=0.5` means the
median pilot loss has 50% acceptance at the initial temperature. It does not
mean that half of all proposals should be accepted.

## Implemented diagnostics

### Exact emissions-only additive oracle

`basis.experimental.dyadic.dynamic_programming` computes the global optimum
for every fixed region count in a canonical dyadic tree. It is the correct
reference for the present emissions-only projected-Gaussian DFS score when the
baseline is known or has already been removed. It is not an oracle for the
joint problem with uncertain boundary coefficients.

The current implementation keeps complete active-node tuples for each
subproblem and examines feasible left/right count pairs. This is convenient and
fast at the tested \(K\) values, but its worst-case work is quadratic in the
region limit per internal node. A larger production solver should store scores
and compact backpointers.

### Arbitrary labelled partitions

`basis.experimental.dyadic.partition_diagnostics` evaluates any positive
integer-labelled search-grid partition under the same projected prior as the
dyadic model. It provides:

- regional summed design columns and \(\tau^2/n_k\) variances;
- directly accumulated aggregation covariance;
- innovation-covariance closure checks;
- projected DFS;
- an emissions-only precision-weighted compression score; and
- an analytic Gaussian posterior mean with an optional explicit baseline
  design and training subset.

This permits a like-for-like score for the existing quadtree output without
claiming that quadtree and dyadic search use the same partition dictionary.

### Synthetic inversion experiment

Run:

```console
python examples/basis/dyadic_basis_diagnostics.py
```

The experiment uses the aligned one-day TAC/MHD native emissions sensitivity
and the frozen 32-column boundary sensitivity. It holds out 12:00 through
17:00 UTC at both sites, constructs bases from training rows only, and compares
at requested \(K=31\):

- greedy dyadic splitting;
- exact dyadic dynamic programming;
- fixed-count SLS with representative-loss acceptance targets 0.5 and 0.1;
- the existing quadtree algorithm, constructed from its existing cellwise
  precision-weighted proxy and then evaluated with the common projected score;
  and
- the native no-reduction Gaussian reference.

Three synthetic truths are evaluated: the prior mean, a seeded Gaussian draw,
and a smooth non-basis-aligned scaling field. The baseline is represented by
the actual frozen boundary design, not by an intercept hidden in the error
covariance. Results are repeated for observation-only error, additive 5, 10,
and 20 ppb mismatch, and the existing percentile-floor rule recomputed from
training rows only.

For this one-day split, that legacy rule produces 9.48 ppb at MHD and 4.21 ppb
at TAC, not 43 ppb. These still exceed the respective median supplied errors of
3.17 and 1.97 ppb. The much larger value seen in the earlier setup demonstrates
that this empirical range statistic is dataset-dependent; it remains a
sensitivity case rather than a recommended model-error specification.

Artifacts are in `docs/plans/figures/dyadic_basis_diagnostics/`.

## Initial findings

Under observation error plus 5 ppb mismatch and the smooth truth:

- exact DP training DFS is 0.252226 at \(K=31\);
- greedy DFS is 0.252223, within \(3.2\times10^{-6}\) of the optimum;
- neither 2,000-step SLS schedule improves the greedy state in this run;
- existing quadtree DFS is 0.170797 at an actual and effective \(K=31\);
- held-out compression quality is 0.0887 for DP, 0.0885 for greedy, and
  0.0712 for quadtree; and
- known-baseline held-out emissions RMSE is about 6.58 ppb for DP/greedy and
  6.69 ppb for quadtree.

These are proof-of-concept numbers, not a ranking of production algorithms.
The quadtree gap reflects both a different construction objective and a
different partition dictionary. The synthetic metric is based on only 47
observation rows.

The blocked holdout also exposes a baseline limitation. Some boundary
directions used in the held-out time block have no training support, so their
posterior means remain at the prior. Total modeled-mole-fraction RMSE is
therefore dominated by boundary extrapolation. Emissions RMSE and compression
quality must remain separate from total RMSE.

## Frozen boundary-fixture diagnostic

The stored one-day real observations should not be combined with the frozen
boundary contribution in this experiment because that boundary fixture is
known to be corrupted, especially for late TAC rows. At unit emissions and
boundary coefficients:

- prior emissions range from about 2.8 to 49.7 ppb;
- prior boundary contribution ranges from about 95.9 to 1772.0 ppb; and
- the observation-minus-prior residual ranges from about 128.5 to 1846.0 ppb,
  with RMSE about 589 ppb.

This is far beyond the stored observation errors and floors, but it is not
evidence that the observations themselves are invalid. It shows that the
frozen boundary contribution cannot support a real posterior or total
observation holdout score.

A bounded MHD-only diagnostic could estimate a constant marine baseline from
western winds. [ICOS describes 180--300 degrees as Mace Head's broad North
Atlantic clean sector](https://icos-atc.lsce.ipsl.fr/panelboard/MHD/), while a
[methane mass-balance study used 240--300 degrees plus trajectory
screening](https://acp.copernicus.org/articles/19/3043/2019/acp-19-3043-2019.html).
In this fixture both sectors select the same 16 first-day MHD hours and give a
mean of about 1933.9 ppb. This is suitable only as an explicitly rough
diagnostic constant, not a replacement for time-varying boundary conditions.
Emissions-only holdout compression does not require any baseline and remains
the preferred next experiment.

## Scale and performance check

A separate full-week fixed-\(K=250\) run on the 8x8-block search grid used
10,000 paired proposals with an initial representative-loss acceptance target
of 0.1. The common Gaussian model took 0.79 seconds to construct. Measured
end-to-end and partition-solver timings were:

- SLS: 68.1 seconds end-to-end, including a model rebuild, greedy
  initialization, temperature pilot, and search;
- exact DP: 1.09 seconds end-to-end using the common model construction, of
  which 0.30 seconds was the partition recurrence;
- accepted proposals: 3023 of 10,000;
- greedy initializer DFS: 0.490334523;
- SLS best DFS: 0.490334613; and
- exact DP DFS: 0.490338939.

The candidate-level timing fields in the generated manifest have explicit
scope strings and should not be compared without those scopes. The scale check
shows that the current code can search at the prototype's region count, but
this additive Gaussian objective is both easier and much faster to solve
exactly. SLS performance becomes the relevant research target only for a
non-additive score or constraints that break the tree recurrence.

## Completed resolution and holdout experiment

Run:

```console
python examples/basis/dyadic_resolution_sweep.py
```

The experiment uses the full-week emissions sensitivities without fitting or
scoring observed mole-fraction targets and without boundary conditions. The
fixed observation-error weights do include within-hour variability estimated
from observed mole fractions, so this is not complete statistical independence
from concentration measurements. Five folds hold out January 2 through 6 in
turn at both TAC and MHD, and remove a 24-hour buffer from either side of the
training set. It compares fixed \(K=16,31,64,250\) at native-cell block widths
8, 4, and 2. Exact DP is capped at \(K=64\) on the finer grids; the greedy
search still reaches \(K=250\). A bounded comparison also evaluates the
no-mask axis-parallel and quadtree algorithms at width 4 and \(K=64,250\).

The all-leaf representation ceiling is the clearest result. Median training
native-grid DFS retained is about 0.10 at width 8, 0.25 at width 4, and 0.52 at
width 2. Median holdout compression at the corresponding all-leaf partitions
is about 0.16, 0.37, and 0.61. Thus the original 8x8-block grid is not merely a
runtime approximation: it excludes most of the native-grid information from
the partition dictionary. Increasing from \(K=64\) to 250 at width 8 has
essentially no effect because the available representation has already
saturated.

Single-cell influence is material. Across the five unthinned folds, the most
influential native cell contributes about 3.8--7.1% of native DFS and the ten
most influential cells contribute about 30--39%. Such cells cannot be isolated
inside an ordinary 8x8 leaf. The generated cell-DFS map and CSV retain the
native coordinates of the largest cell for each fold.

Greedy splitting is usually close to exact DP on its training objective, but
the oracle does not always have the best holdout compression. At low \(K\) on
the finer grids, exact DP can select a training-optimal dictionary that
generalizes worse than greedy splitting. The DP result is therefore a useful
optimizer-loss reference, not an independent validation score. DP gaps are
reported only for algorithms in the same dyadic dictionary.

For the central fold, thinning training rows to every sixth hour across all six
wall-clock phases changes held-out compression very little. This supports the
spatial conclusions against one simple reduction in closely spaced data. It
does not simulate temporal correlation: it reduces repeated information while
leaving the diagonal likelihood unchanged. Explicit non-diagonal covariance
remains a separate model extension.

Artifacts are in `docs/plans/figures/dyadic_resolution_sweep/`. Candidate and
resolution CSVs include both per-fold compression and the weighted covariance
traces used to pool compression across folds. The report gives median/range and
pooled summaries separately. The manifest records hashes for every input
fixture and local source file used by the experiment.

## Next experiments

1. Test adaptive native-cell refinement around high-influence cells. A width-1
   global tree is the clean reference, but a mixed-resolution tree is more
   likely to be computationally useful.
2. Repeat selected width-2 and adaptive cases after removing or capping the
   highest-influence cells. This distinguishes broad resolution gain from a
   small number of dominant receptors/grid cells.
3. Add observation-error and fixed model-mismatch sensitivity to the blocked
   compression sweep. Keep `mf_error` visible and report the additive model
   error separately.
4. Compare six-hour bin averages with the existing six-hour phase thinning.
   Aggregation can reduce short-lag dependence and gives each time block a
   different sensitivity row, but it is a robustness approximation rather
   than a simulation or estimate of non-diagonal temporal covariance.
5. Use explicit land/sea or country masks in a separate partition-dictionary
   experiment. The no-mask axis-parallel comparison is not evidence about the
   value of physical constraints.
6. Replace full active-node tuples with compact DP backpointers before running
   exact \(K=250\) frontiers on width 4 or 2.
7. Diagnose and reconstruct valid boundary contributions only before fitting
   stored real observations or reporting total holdout prediction. The rough
   MHD western-wind constant is sufficient as a bounded diagnostic fallback.
8. Defer explicit temporal covariance until the emissions-only spatial tests
   need it or an inversion-quality synthetic experiment demonstrates a failure
   that blocked holdouts and thinning cannot diagnose.
