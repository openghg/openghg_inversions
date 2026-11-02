# OPE-184 implementation evidence

The owner-approved design addendum records the profile and strategy decision
required by tasks 1.2–1.3. This note retains the caller inventory and raw
profile runs for tasks 1.1–1.3.

## Direct-use inventory

| Caller | Source and flux need | Time, eager, and map need |
| --- | --- | --- |
| `BasisOperator.interpolate` | Linear grid scaling, optional caller weights; gathered multisource weights yield a total | Multisource result is eager; single-source remains lazy |
| `BasisFunctions.interpolate` | Same action with the retained signed flux when `flux=True` | Delegates the operator's legacy conversion |
| Basis postprocessing in `_basis_products.py` | Reconstructs total flux and scaling statistics; selected sector bases use their retained flux | Product labels use `flux_time`; completed outputs materialize before consumers and writers |
| `MultiSourceBucketBasisOperator.sensitivity` | Pairs source-labelled footprint-times-flux with gathered retained states | Uses an expanded map internally to keep source identity during the grid contraction |
| `AffineFluxMap` | Uses the bucket map in its separate centred equation | Needs a labelled expanded map with one native source axis; retains public `prolongation` |
| Covariance tests and profiling scripts | Inspect an expanded labelled source map | Internal map access only; no other production caller was found |

The repository search found no other direct production use of the map adapter.
External direct use cannot be measured from this checkout. The map adapter was
internalized now; `interpolate` stays deprecated through the next release cycle,
with removal to follow a downstream usage review.

## Calculation profile

Run `scripts/profile_ope184_reconstruction.py` with each `case` and `strategy`.
The Dask fixture uses a 48 × 64 grid, two chains, 20 draws, 24 × 32 grid
chunks, and nonlexicographic sources `zeta`, `alpha`, `mu`. The ragged state
has 7, 13, and 5 regions. Each source-preserving result is explicitly summed
and compared with the historical gathered total. Results below are one warm
local run per command on 2026-09-25; timings include graph construction,
execution, and the parity check. Peak RSS includes Python imports and is
therefore most useful for comparing runs in the same environment.

| Case | Strategy | Seconds | Peak RSS MiB | Dask tasks | Largest dense / sparse chunk MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Single | gathered slices | 0.474 | 340.5 | 37 | 0.469 / 0.012 |
| Single | per-source | 0.545 | 340.7 | 37 | 0.469 / 0.012 |
| Single | expanded | 0.482 | 340.7 | 37 | 0.469 / 0.012 |
| Single | legacy baseline | 0.442 | 344.3 | 37 | 0.469 / 0.012 |
| Shared state, source flux | gathered slices | 0.508 | 349.2 | 109 | 1.406 / 0.012 |
| Shared state, source flux | per-source | 0.478 | 351.2 | 109 | 1.406 / 0.012 |
| Shared state, source flux | expanded | 0.474 | 350.4 | 109 | 1.406 / 0.012 |
| Shared state, source flux | legacy baseline | 0.554 | 358.5 | 109 | 1.406 / 0.012 |
| Ragged multisource | gathered slices | 4.883 | 408.9 | 197 | 1.406 / 0.035 |
| Ragged multisource | per-source | 0.832 | 358.9 | 259 | 1.406 / 0.047 |
| Ragged multisource | expanded | 3.482 | 389.2 | 213 | 1.406 / 0.079 |
| Ragged multisource | legacy total | 0.903 | 357.5 | 637 | 0.469 / 0.035 |

The initial per-source candidate rebuilt a bucket operator during each call.
That construction computed chunked basis data before returning a lazy result,
so it was replaced with per-source matrices retained from the operator's
existing construction step. The single and shared cases use the same direct
contraction for all strategy labels because no source-specific state map needs
splitting or expansion.
Separating graph construction from Dask execution in a second pass gave:

| Case and strategy | Graph construction s | Execution s |
| --- | ---: | ---: |
| Single, direct | 0.036 | 0.508 |
| Shared state, direct | 0.038 | 0.473 |
| Ragged, gathered slices | 0.075 | 4.602 |
| Ragged, per-source | 0.081 | 0.494 |
| Ragged, expanded map | 1.586 | 1.376 |
| Ragged, legacy total | included in eager call | 0.664 |

The legacy ragged call computes eagerly within the method, so construction
and execution cannot be timed separately at that API boundary.

After retaining the already-built per-source matrices, the same ragged fixture
was rerun in separate processes. These are one-run measurements, so the RSS
figures are comparable within this pass, not with the earlier pass above.

| Strategy | Graph s | Execution s | Peak RSS MiB | Dask tasks | Largest dense / sparse chunk MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gathered slices | 0.032 | 4.726 | 1748.6 | 197 | 1.406 / 0.035 |
| Retained per-source matrices | 0.029 | 0.524 | 1690.4 | 193 | 1.406 / 0.012 |
| Expanded map | 1.335 | 1.397 | 1741.4 | 213 | 1.406 / 0.079 |
| Legacy total | included in eager call | 0.631 | 1741.5 | 637 | 0.469 / 0.035 |

The retained per-source calculation remains the fastest source-preserving
option in this fixture, avoids the expanded `(native_source, grid, state)` map,
and did not execute Dask tasks while constructing the result. This strategy
applies to all ragged states regardless of provenance; shared and single states
use direct contraction. Existing total-grid products are summed at the
postprocessing boundary. Memory cost should be reassessed if production draw
counts or grids greatly exceed this fixture.
