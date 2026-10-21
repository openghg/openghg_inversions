# OPE-152 native amplitude prototype: performance evidence

Measured on 21 October 2026, following the
[amplitude kernel and CVM work](ope152_amplitude_nuts_performance.md).
This is dated engineering evidence for
[OPE-152](https://linear.app/openghg-inversions/issue/OPE-152/validate-vg-sampling-and-hpc-operation-for-specialised-co-runners),
not a supported backend or a convergence certificate.

The corrected native prototype reduced median compiled-call time by **7.24%**
across six fixed conditioning anchors. In two short chains per variant,
retained transitions took **4.74% less time**, but startup cost increased and
amplitude bulk effective sample size per second (ESS/s) fell in 11 of 12
coordinates. These results establish a modest execution improvement on this
target, without establishing a general sampling-efficiency improvement.

## Target, method and provenance

All comparisons used 4,557 observations, rank 256 plus the preserved diagonal
covariance tail, 264 physical flux-scaling coefficients, four boundary
coefficients and 12 site amplitudes. The fixed OU timescale was 5 h; amplitude
priors were HalfNormal(0.75 ppm), boundary priors were Normal with mean 1,
standard deviation 0.01 and lower bound 0, and there was no offset. Data, priors,
covariance representation, cache refresh, adaptation settings and state
parameterization stayed fixed.

The local prepared-input SHA256 was
`22313f482e0eb2143804f2a290cddc5f247ad293db11776977b75d631df06353`;
its original dense-source SHA256 was
`7ab7cf733eba7b12dd1fea7a5c8812236eda1d579486b03e5e2d27feab8866bf`.
The six historical residual/amplitude anchors had SHA256
`08cddffb167f48e478439ea3036c5eb82da1f6e1010fcfee0820a41ba44cf1c1`.
They are selected diagnostic points, not posterior-weighted timing samples.

| Evidence round | Source revision | Slurm job | Purpose |
| --- | --- | --- | --- |
| r1 | `7b4fa5f88be280e996c892c56bb90a58683ab136` | 19595655 | First native kernel and compiled graph |
| r2 | Same as r1 | 19596237 | Diagnose Gram and reduction costs |
| r3 | `5c771e448cc7632a55a98aaf62a2f4b78e65a705` | 19597063 | Corrected native kernel and graph |
| r4 | Same as r3 | 19597739 | CVM versus native conditional sampling |

The frozen numerical oracle was CVM revision
`e2bb62465f43b287b08e394bdc552b8e4aa599cc`. Microbenchmarks compared that oracle,
the extracted shared Python kernel under CVM, and the native lowering. Sampling
used the **current shared kernel in both arms**; it did not sample the old
revision against the new one. All performance runs used Python 3.13.7,
NumPy 2.2.6, SciPy 1.15.3, PyMC 6.3.2, PyTensor 3.3.2 and Numba 0.63.1,
float64, and one thread in each NumPy/SciPy BLAS pool.

The [shared kernel](../../openghg_inversions/models/fixed_ou.py) retains one
likelihood implementation. The [prototype](../../scripts/ope152_native_amplitude.py)
provides a genuine `numba_funcify` lowering with explicit numerical inputs:
amplitudes, transformed residual, covariance factor, mode eigenvalues, site
indices and base log determinant. It neither captures the prepared target in
a callback nor calls Python `Op.perform` during native evaluation. Chain-local
residual installation, public evaluator behavior and general residual gradients
are preserved. The production amplitude constructor remains CVM.

## Why the first native version was slower

Each microbenchmark timed 12 repetitions per anchor and method, balancing all
six orders of the three variants. Compilation and warm calls were separate.
Outer clocks measured either the numerical kernel or the complete amplitude
graph, including its prior and transform Jacobian; no inner native timer ran.

In r1, pooled compiled-call median increased from **44.712 to 55.428 ms**;
native was 23–25% slower at every anchor. Native execution removed the large
Python callback payload, but that alone did not make its numerical code faster.
Warmed native returns used only 32-byte dtype-boxing callbacks, with zero
`NativeAmplitudeOp.perform` calls. The direct kernel showed the same regression.

The r2 single-anchor probe isolated two implementation costs. NumPy's
same-buffer Gram operation used symmetric rank-k multiplication (SYRK), while
the initial native matrix product used general multiplication (GEMM). Comparing
both operations within SciPy's BLAS also showed the difference, rather than
relying only on NumPy/SciPy library comparisons. Native squaring and axis-zero
reduction also handled the triangular solve's Fortran-layout result poorly.

| r2 diagnostic, median ms | Python/reference | Initial native | Fused native reduction |
| --- | ---: | ---: | ---: |
| Gram, NumPy SYRK / native GEMM | 8.342 | 13.144 | — |
| Gram, same SciPy BLAS SYRK / GEMM | 8.487 | 13.220 | — |
| Triangular solve | 27.272 | 27.086 | — |
| Squared column norms, known Fortran layout | 2.524 | 13.475 | 1.690 |
| Solve plus column norms | 29.964 | 41.735 | 29.083 |
| Full numerical kernel | 45.460 | 54.358 | — |

The probe retained actual Cholesky/solve layouts and separately checked the
combined solve/reduction. Standalone stages used precomputed inputs; their
times are diagnostic comparisons, not an additive accounting of the full kernel.

The corrected prototype uses a native SYRK helper with triangle mirroring and
a fused column-norm loop. Ordinary Python helper expressions are unchanged.
It reuses installed native Cholesky/triangular solver wrappers, keeps fastmath
and parallel execution disabled, and uses a strict linker when graphs are
constructed or rebuilt. Private PyTensor APIs and a process-local SciPy BLAS
pointer remain prototype limitations: persistent native-code caching is disabled.

In r3, pooled compiled-call median fell from **43.981 to 40.798 ms**, or
**7.24%**; individual-anchor reductions were 6.95–7.41%. Shared Python/CVM
was 43.984 ms. Direct-kernel medians were 43.852 ms frozen Python,
43.793 ms shared Python and 40.742 ms native. Comparing r1 and r3 absolute
times across separate allocations is less informative than each round's
balanced comparisons. All 42 r3 numerical checks passed; maximum absolute
log-density and amplitude-gradient differences were 1.27e-11 and 6.04e-14.
Dynamic shared-array updates, invalid inputs, ownership and absence of a
Python fallback were checked independently of timing.

## Sampling: faster transitions, more startup work

Job 19597739 ran four fresh processes in ABBA order: CVM seed 152202610,
native seed 152202610, native seed 152202611, CVM seed 152202611. Each ran
one chain with 300 warmup and 200 retained draws, its own empty compiler cache,
amplitude/state target acceptance 0.9/0.95, maximum tree depth 10 and early
warmup depth 8. The job requested one CPU and 4 GiB, completed in 20m53s
on `bp1-compute068`, and recorded sampled MaxRSS of 939,332 KiB.

The density timer wraps the actual integrator callable after compilation and
warm calls. It includes likelihood, prior, transform Jacobian and unravel work;
an `Op.perform` timer would miss native execution. Density, trajectory and
wrapper timers are nested. Only amplitude wrapper plus state trajectory are
disjoint, and their sum defines **transition time**. All 2,000 sweeps passed
the density-count check (`tree_steps + 1`) and refresh/statistic reconciliation.

| Quantity, pooled over two chains | CVM | Native |
| --- | ---: | ---: |
| Warmup transition time | 367.399 s | 342.456 s |
| Retained transition time | 178.239 s | 169.785 s |
| Warmup / retained density calls | 6,702 / 3,184 | 6,566 / 3,192 |
| Mean warmup density call | 45.708 ms | 42.570 ms |
| Mean retained density call | 45.839 ms | 42.634 ms |
| Three-part startup | 44.940 s | 67.196 s |
| `pm.sample` wall time | 547.238 s | 514.523 s |
| Recorded harness elapsed | 607.095 s | 594.506 s |

Warmup transitions took 6.79% less time and retained transitions 4.74% less;
retained savings were 4.56–4.93% per seed. Mean retained density-call cost fell
6.99%. Neither seed produced identical draws or identical overall tree work;
small arithmetic differences can propagate through adaptation. The same-seed
runs measure execution and differing trajectories, without establishing the
specific cause of each trajectory difference. Six-anchor checks provide the
direct numerical-parity evidence.

Startup comprises model construction, sampler construction and initial
density/lazy compilation. Experiment-only oracle checks (1.428/2.806 s) and
serialization (3.517/1.513 s), CVM/native totals, are separate from startup and
transition timing but included in recorded harness elapsed. That elapsed time
fell only 2.07%. External process wall totaled 641.42/609.93 s, but the first
CVM process spent 28.03 s outside the harness clock versus 6.30–8.00 s for the
other arms. Its apparent 4.91% reduction is order-sensitive, not evidence for
a universal cold-start or whole-job improvement.

Native retained time remained 80.50% amplitude trajectory (80.15% density),
16.89% cache refresh, 0.75% residual installation and 1.54% state trajectory.
The unchanged cache cost 68.549/71.693 ms per refresh, with 399/400 refreshes;
the cause of its increased time was not isolated. Step 2 amplitude evaluation
therefore remains the larger cost than Step 3 state updates.

## Diagnostics, validation and disposition

Both variants had zero retained divergences or depth flags in either block.
Each had four amplitude and six state warmup divergences, with no depth flags.
Retained amplitude trees used three or seven steps (median seven); state trees
used 31 or 63 (median 31). Amplitude step sizes were 0.468–0.498 for CVM
and 0.440–0.582 for native. These observations give no new reason to change
the metric, parameterization or tree caps.

The following descriptive ESS/s estimates divide by pooled **retained transition
time**, excluding warmup and startup. They use only two 200-draw chains per arm.

| Variables | Maximum rank R-hat, CVM / native | Median bulk ESS/s, CVM / native | Median tail ESS/s, CVM / native |
| --- | ---: | ---: | ---: |
| Amplitudes | 1.0166 / 1.0108 | 4.613 / 3.715 | 1.748 / 2.031 |
| Physical flux scaling | 1.0259 / 1.0315 | 2.268 / 2.892 | 1.640 / 1.815 |
| Latent flux coordinates | 1.0338 / 1.0307 | 2.289 / 2.856 | 1.683 / 1.816 |
| Boundary coefficients | 1.0278 / 1.0061 | 1.719 / 2.900 | 1.469 / 1.881 |

Amplitude bulk ESS/s decreased in 11/12 coordinates; the median coordinate-wise
native/CVM ratio was 0.7925. Physical-flux group medians improved, but 44/264
bulk and 79/264 tail coordinate rates decreased. Group medians hide this
variation. Some R-hat values exceed 1.01, and these short traces neither certify
convergence nor establish an intrinsic efficiency improvement or regression.

Validation job 19597060 passed 254 tests on Python 3.13.7 with the runtime
versions above, including 43 prototype tests. Coverage includes likelihood and
amplitude-gradient parity, invalid inputs, shared-array updates, borrowed ownership,
residual refresh, C/Fortran/strided layouts, and strict flags through copying,
pickling and spawn. Independent frozen-oracle checks found 300 exact general-
evaluator comparisons; 150 native comparisons had maximum likelihood/gradient
errors 8.88e-16/6.66e-16. This does not establish native compatibility with
Python 3.12 or older dependency versions. Spawn tests covered adapters and
compiled functions, not full parallel `pm.sample` execution. The prototype's
temporary constructor-patching context is unsafe for concurrent Python threads.

Keep this as a prototype: the measured gain is modest beyond the simpler CVM
implementation, startup is more expensive, and private-API maintenance remains.
The comparison does not justify a production backend switch or a full metric.
Any adoption or longer efficiency study remains a separate OPE-152 decision.

## Reproduction and retained evidence

The [microbenchmark driver](../../scripts/profile_ope152_amplitude_native.py)
accepts `--prepared`, `--expected-sha256`, `--anchors`, `--out` and `--repeats 12`.
Use source revision r1 or r3 above, the identified input/anchors, float64 and
one BLAS thread from startup. Production-sized work belongs on Slurm.
The [timing CSV](ope152_native_amplitude_timings.csv) records exact comparison
values, units, source revisions, job IDs and SHA256 hashes of their evidence.
Its `candidate_fraction_less` is `1 - candidate/reference`; a positive value
means a smaller value, not necessarily an improvement for ESS or R-hat.

Raw manifests, frozen scripts, traces, runtime hashes and independent audits
remain in local `inversion_outputs/ope152-native-amplitude-20261021-r1/` through
`-r4/`. They are ignored artifacts, not files distributed with this repository.
r1/r3 use `micro/manifest.json`; r2 uses `stages/manifest.json`; r4 retains
`profile_sampling.py`, `profile.sbatch`, four arm manifests/traces, `analysis.json`,
`independent-native-r4.json` and `diagnostic-audit.json`. Its frozen analyzer
SHA256 is `6d0901252f340f13b9ec3002ffcd341fc8f9ab7f5f29984ad2d331a20be3c2eb`.
These records preserve per-coordinate diagnostics and alternative time
denominators; none of the earlier experiment artifacts were overwritten.
