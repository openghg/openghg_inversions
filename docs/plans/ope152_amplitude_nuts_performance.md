# OPE-152 amplitude-conditional NUTS performance

Measured on 21 October 2026, following the
[guarded sum-of-squares cache work](ope152_sum_squares_performance.md).
This records a bounded implementation and sampling comparison for
[OPE-152](https://linear.app/openghg-inversions/issue/OPE-152/validate-vg-sampling-and-hpc-operation-for-specialised-co-runners).
It is engineering evidence on one target, not production convergence or a
general sampler-performance guarantee.

The final compiler candidate `6560e9d145e11b80687064ddd9f654de197ed0d0`
reduced pooled retained transition time from 271.670 to 172.534 seconds,
**36.49% less**, against the already optimized `2a440c98` numerical kernel.
Warmup took 37.24% less time. This comparison and its short-trace limitations
are recorded in the final section below.

The earlier numerical candidate `2a440c9827aa2906c79b3b1056fe0952f27e15be` reduced pooled retained
transition time from 282.004 to 265.044 seconds, **6.01% less**, against PR #926
at `9acd0c579b1ce61247f07892965fa1f44105e258`. One seed produced identical draws
and work counts, with a 5.04% reduction; the other followed a different
adaptation path. The amplitude likelihood itself took 11.28% less time per
call. Amplitude trajectories still consumed 88.50% of retained transition time.

## Implementation and numerical checks

During an amplitude trajectory, flux and boundary-condition coefficients stay
fixed, so the observation residual stays fixed. The candidate transforms that
residual into prepared mode coordinates once when installing the conditional,
then reuses it throughout the trajectory. Its private evaluator returns the
likelihood and amplitude gradient without calculating the unused observation-
residual gradient and its backward transform. The public evaluator retains its
full outputs and rejection behavior. Covariance, priors, cache refresh,
factorizations, NUTS adaptation and depth limits are unchanged.

An interleaved microbenchmark checked nine prior-state/amplitude combinations
and six heterogeneous posterior anchors. All 15 likelihood and amplitude-gradient
comparisons with the pinned baseline were exactly equal in that run. Across the
six posterior anchors, median direct evaluation time fell from 50.386 to
44.503 ms. Installing the transformed residual cost about 3 ms once per
trajectory; the sampled timings below include this cost.

The anchors came from an existing trace of the same numerical target. All 40
shared prepared-input datasets were checked bitwise despite different file
serialization. Residuals used physical flux and boundary-condition coefficients;
the reconstructed completed mean agreed with the stored mean within 1.14e-13 ppm.
These six fixed anchors are diagnostic cases, not weighted posterior timing samples.

## Representative sampling protocol

Job `19587951` ran four fresh processes in ABBA order on one allocation:
baseline seed `152202610`, candidate seed `152202610`, candidate seed
`152202611`, baseline seed `152202611`. Each process ran one chain with 300
warmup and 200 retained draws. Both NumPy and SciPy BLAS pools used one thread
from process startup. The job requested one CPU and 4 GiB; Slurm allocated two
logical CPUs. It completed in 30m12s with exit status zero on `bp1-compute121`,
using Python 3.13.7, NumPy 2.2.6, SciPy 1.15.3,
PyMC 6.3.2 and PyTensor 3.3.2.
Maximum process peak RSS was 1,879,480 KiB; Slurm's sampled MaxRSS was
1,814,760 KiB.

Both variants used the same public prepared artifact: 4,557 observations,
rank 256 plus its preserved diagonal tail, 264 flux coefficients, four boundary
coefficients and 12 site amplitudes. Its SHA256 was
`22313f482e0eb2143804f2a290cddc5f247ad293db11776977b75d631df06353`.
The fixed OU timescale was 5 h; amplitude priors were HalfNormal(0.75 ppm),
boundary priors were lower-truncated Normal(1, 0.01), and there was no offset.
Amplitude/state target acceptance stayed at 0.9/0.95, with maximum tree depth
10 and early-warmup depth 8. Both variants retained PR #926's cache refresh.

The [sampling profiler](../../scripts/profile_ope152_amplitude.py) records
nested timers. Only the amplitude-step wrapper and state trajectory are
disjoint; their sum defines **transition time** below. Likelihood evaluation
is inside the amplitude trajectory, which is inside the wrapper. Adding all
timer columns would double-count work. Startup/compilation and `pm.sample`
wall time are recorded separately. Serialization is outside transition timing
and has no dedicated timer. Fresh processes share the job's compilation caches,
so these results do not establish cold-start savings.

## Timing and remaining cost

Totals pool two chains per variant. The
[per-chain timing CSV](ope152_amplitude_timings.csv) retains every arm and phase.

| Quantity | Baseline | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| Warmup transition time | 579.765 s | 547.968 s | 5.48% less; 600 sweeps each |
| Retained transition time | 282.004 s | 265.044 s | 6.01% less; 400 sweeps each |
| Retained likelihood calls | 3,188 | 3,180 | Work differs in one seed |
| Retained likelihood time per call | 49.521 ms | 43.934 ms | 11.28% less |
| Retained amplitude trajectory | 252.536 s | 234.556 s | Includes likelihood and trajectory overhead |
| Retained residual installation | 0.013 s | 1.205 s | Includes the moved transform |
| Retained cache refresh | 25.881 s | 25.723 s | 400 versus 399 refreshes |
| Retained state trajectory | 3.047 s | 3.034 s | About 1.1% of transition time |
| Total `pm.sample` wall time | 862.860 s | 813.746 s | Warmup and retention, both chains |

For seed `152202611`, all 500 warmup/retained draws and work counts matched
bitwise; retained transition time fell from 140.126 to 133.066 s (5.04%).
For seed `152202610`, initial residuals and amplitudes matched, followed by
small state-side numerical differences during warmup and eventually different
tree counts. Its retained reduction was 6.98%, but it is not an identical-work
comparison. The pooled result is an observed reduction, without a confidence
interval from these two seeds; it should not be presented as a universal speedup.

Candidate retained time was 88.50% amplitude trajectory, 9.71% cache refresh,
0.45% residual installation and 1.14% state trajectory, with the remainder in
the wrapper. The amplitude trajectory exceeded its measured likelihood work
by 94.847 s, or 29.826 ms per likelihood call. This remainder includes compiled
graph, callback and NUTS bookkeeping; the timer does not identify their shares.

## Sampling diagnostics and conditional geometry

Both variants had zero retained divergences and zero maximum-depth flags in
either block. During warmup each had four amplitude and six state divergences,
with no maximum-depth flags. Retained amplitude trees used three or seven
steps (median seven, depth at most three), with adapted step sizes 0.482–0.590
across the four arms. State trees used 31 or 63 steps. Early warmup reached
127 amplitude and 255 state steps, which is included in the warmup totals.

The following estimates use two chains of 200 retained draws per variant.
R-hat is rank-normalized; effective sample size (ESS) is the minimum across
coordinates. ESS/s divides by **retained transition time**, 282.004 s for
baseline and 265.044 s for candidate, excluding warmup and startup.

| Variables | Maximum R-hat, baseline / candidate | Minimum bulk ESS/s, baseline / candidate | Minimum tail ESS/s, baseline / candidate |
| --- | ---: | ---: | ---: |
| Site amplitudes | 1.0180 / 1.0166 | 1.227 / 1.921 | 0.883 / 0.979 |
| Physical flux coefficients | 1.0239 / 1.0344 | 0.821 / 0.840 | 0.488 / 0.276 |
| Latent flux variables | 1.0304 / 1.0344 | 0.842 / 0.757 | 0.580 / 0.276 |
| Boundary coefficients | 1.0198 / 1.0349 | 1.292 / 1.537 | 0.806 / 0.802 |

Including warmup increases the denominators to 861.768/813.012 s; amplitude
minimum bulk ESS/s becomes 0.402/0.626, and physical-flux minimum tail ESS/s
becomes 0.160/0.090. Minimum physical-flux tail ESS fell from 137.7 to 73.0.
These short traces do not certify convergence or a reliable ESS/s improvement:
adaptation paths differ, estimates are noisy, and the variants share one identical chain.
Longer scientific sampling still needs joint amplitude, flux and boundary
convergence checks. An earlier four-chain, 1,000-retained-draw run on the same
numerical target had at most seven retained amplitude steps, no retained
divergences/depth flags and amplitude maximum R-hat 1.0032. That historical
trace supports investigating evaluation cost before changing the metric; it
is not validation of the new implementation.

The 12-dimensional amplitude conditional fixes the completed mean from flux
and boundary coefficients. Its geometry is distinct from full joint geometry;
low-rank covariance can still couple sites. To isolate the effect of changing
that completed mean, the [geometry diagnostic](../../scripts/profile_ope152_amplitude_geometry.py)
held log-amplitudes at one common reference and varied six posterior residuals.
It included the amplitude prior and log-coordinate Jacobian, using centered
gradient differences at steps 0.001 and 0.0005 (288 evaluations).

All twelve negative Hessians were finite and positive definite. Relative
Frobenius disagreement between step sizes was below 1.00e-7; raw antisymmetry
fell approximately fourfold when the step halved. At the finer step, spectral
condition numbers were 4.33–4.49, or 1.217–1.225 after diagonal scaling.
Largest absolute inverse-Hessian correlations were 0.0750–0.0806. Across the
six residuals, maximum relative curvature change was 2.34%, and the largest
per-site diagonal-curvature span was 7.55% of that site's minimum.

This is modest local coupling and modest sensitivity to flux/boundary
conditioning at the chosen reference. The inverse-Hessian correlations are
local Gaussian approximations at a non-mode, not exact conditional correlations.
Six residuals at one log-amplitude point do not cover tails, near-zero amplitudes
or another month. Together with the short retained trees, the evidence gives
no present reason for a dense-metric experiment or a lower tree-depth cap.

## Validation, reproduction and next priority

Job `19587953` passed all 210 relevant tests on each of Python 3.12 and 3.13
through the repository's Slurm tox runner. Coverage includes density/gradient
parity, nonfinite rejection, input ownership, residual installation and cache
lifecycle, alongside the existing covariance stability and public model tests.
The job used two CPUs and 8 GiB, completed in 11m19s with exit status zero,
and recorded sampled MaxRSS of 4,998,412 KiB.
Microbenchmark checks and the fixed-coordinate geometry diagnostic also passed.

The sampling profiler pins baseline `9acd0c57` and records current source
hashes, input hashes, configuration, native sampler statistics and traces.
To repeat the measured protocol, use candidate `2a440c98`, the prepared artifact
identified above and one BLAS thread from startup. First run `--mode micro`
with `--anchors` pointing to verified residual/amplitude anchors. Then run
four separate `--mode sample` processes in the ABBA seed order above, each
with `--chains 1 --tune 300 --draws 200`. Each invocation needs `--prepared`
and a distinct `--out` directory. Production-sized runs belong on Slurm.
Frozen sources, manifests, traces and analysis are retained in
`inversion_outputs/ope152-amplitude-20261021-r1/` and
`inversion_outputs/ope152-amplitude-20261021-r2/`; those local artifacts are not
distributed with the source repository.

Before the compiler follow-up below, the next performance question was the
amplitude trajectory's remaining cost.
A separate tiny synthetic PyTensor/Numba experiment found that increasing an
unused captured array from zero to 8 MiB increased time outside a constant
Python Op from about 0.021 to 1.604 ms per call, despite memoized unpickling.
This motivated investigating callback payload handling in the installed runtime.
At that stage the production callback payload had not been measured, so the
synthetic result did not explain the whole 29.826 ms remainder. The follow-up
measured its payload and warmed outer/inner call times before implementing a
compiler candidate. Evaluation cost had stronger supporting evidence as a
priority than metric adaptation.

## Compiler follow-up — 21 October 2026

Job `19590092` compared the actual frozen `2a440c98` amplitude graph under Numba
and PyTensor's C virtual machine (CVM), at one verified posterior anchor from
the same input. Twelve balanced AB/BA pairs per stage followed compilation and
five warmups per graph. Both BLAS pools used one thread. The job requested one
CPU and 4 GiB, received two logical CPUs, and completed in 56 s with exit status
zero. Process peak RSS was 1,749,408 KiB; sampled Slurm MaxRSS was 1,752,840 KiB.

Stock compiled-call median fell from **70.517 to 41.094 ms, 41.72% less**.
Every CVM call was faster than every Numba call in that stage. Each Numba call
passed a 79,638,136-byte callback buffer plus four 32-byte buffers, even though
every unpickle lookup was a memo hit. The buffer pattern was identical before
and after adding the inner timer; CVM used no Numba callbacks.

| Instrumented quantity | Numba median (ms) | CVM median (ms) |
| --- | ---: | ---: |
| Whole compiled call | 71.087 | 42.169 |
| Numerical likelihood Op | 42.005 | 42.103 |
| Outside the numerical Op | 29.233 | 0.0646 |

The stock Op was compiled and warmed before instrumentation. Numba reused that
graph; CVM's timed graph was recompiled and warmed outside the clock because
it binds `perform` during compilation. Both stages include callback-counter
bookkeeping. The nearly equal numerical work and large difference outside it
support avoiding the callback path; they do not attribute every microsecond to
payload copying. Both modes matched direct likelihood plus prior/Jacobian:
likelihoods were exactly equal and maximum gradient difference was 3.55e-15.
Input/source hashes and numerical array ownership checks passed.

The candidate selects `Mode(linker="cvm", optimizer="fast_run")` only for the
private amplitude NUTS graph. Constructing the mode explicitly also supports
older PyTensor versions without the `CVM` string alias; `FAST_RUN` alone selects
Numba under this runtime's automatic linker. State compilation, global settings,
priors, covariance, metric and adaptation stay unchanged. No runtime option was
added. The implementation passed 26 focused tests and 13 tests with C++
compilation disabled. Subsequent validation and sampling are recorded below.

This one-anchor graph result is not a sampled-chain speedup. The paired
run below isolates `2a440c98`/Numba against the CVM candidate, preserving the
earlier `9acd0c57`-to-`2a440c98` evidence separately. Frozen graph sources,
48 call records, manifests and the independent audit remain under the local
`inversion_outputs/ope152-amplitude-20261021-r3/` artifact directory.

## CVM sampling comparison and final priority

Job `19593208` compared frozen compiler candidate `6560e9d1` with the exact
amplitude-step constructor from `2a440c98`. An AST check verified that only the
private compiler mode and its import differed; both shared numerical modules
and earlier profiling helpers were byte-identical. It used the same target,
runtime, one-thread BLAS settings, ABBA order, seeds and 300/200 warmup/retained
protocol described above. It completed successfully in 24m44s on
`bp1-compute057`, requesting one CPU and 4 GiB and receiving two logical CPUs.
Maximum recorded process peak RSS was 1,880,980 KiB; Slurm sampled MaxRSS was
1,813,768 KiB. Every source/input hash and every
per-sweep likelihood and refresh count passed the independent audit.

Six historical conditioning anchors each used twelve balanced AB/BA pairs.
Median compiled evaluations took 75.08–77.07 ms with Numba and 44.28–44.95 ms
with CVM, **40.94–41.81% less time** at the corresponding anchors. Likelihoods
matched exactly and the largest transformed-amplitude gradient difference was
3.55e-15. These deliberately chosen anchors are not posterior-weighted timing
samples. This run omitted the earlier callback-size counter.

The [compiler comparison CSV](ope152_amplitude_cvm_timings.csv) preserves all
eight per-chain phase rows separately from the earlier numerical comparison.

| Quantity | Optimized Numba baseline | CVM candidate |
| --- | ---: | ---: |
| Warmup transition time, 600 sweeps | 557.348 s | 349.795 s |
| Retained transition time, 400 sweeps | 271.670 s | 172.534 s |
| Retained likelihood calls | 3,188 | 3,180 |
| Retained numerical likelihood time per call | 44.570 ms | 44.481 ms |
| Retained amplitude trajectory | 240.648 s | 142.144 s |
| Retained residual installation | 1.222 s | 1.246 s |
| Retained cache refresh, 400 each | 26.078 s | 25.610 s |
| Retained state trajectory | 3.201 s | 3.000 s |
| Total `pm.sample` wall, both chains | 830.161 s | 523.055 s |

Retained reductions were 38.63% and 34.23% for the two seeds; pooled retained
and warmup reductions were 36.49% and 37.24%. The unchanged numerical kernel
took nearly the same time per call. Time outside it within the amplitude
trajectory fell from 30.915 to 0.218 ms per likelihood call. This remainder
includes graph and trajectory bookkeeping, not just payload copying.

Neither seed has identical trajectories: tiny initial rounding differences
grow during adaptation and eventually change work counts. The fixed-point
oracle and compiler-scope checks establish numerical agreement independently
of those trajectories. These two seeds provide no reliable confidence interval
or universal speedup estimate. Do not multiply this reduction by the earlier
6.01% result and present it as a directly measured combined gain.

Startup was measured separately: model construction, sampler construction and
initial density compilation/evaluation totalled 19.034/8.035 s for the two
Numba processes and 2.169/2.125 s for CVM. Processes shared the job's compilation
caches, so this is execution-order-dependent startup evidence, not a cold-start
guarantee. The retained transition denominator continues to exclude startup,
warmup and serialization; `pm.sample` wall includes its own setup and recording.

Both variants had zero retained divergences or maximum-depth flags in either
block; each had four amplitude and six state divergences during warmup.
Retained amplitude trees remained three or seven steps, with step sizes
0.482–0.590 for Numba and 0.513–0.515 for CVM. State trees used 31 or 63 steps.
No metric, acceptance target or tree-depth limit changed.

| Variables | Maximum R-hat, Numba / CVM | Minimum bulk ESS/s, Numba / CVM | Minimum tail ESS/s, Numba / CVM |
| --- | ---: | ---: | ---: |
| Site amplitudes | 1.0180 / 1.0096 | 1.274 / 2.902 | 0.917 / 1.153 |
| Physical flux coefficients | 1.0239 / 1.0397 | 0.852 / 1.069 | 0.507 / 0.862 |
| Latent flux variables | 1.0304 / 1.0402 | 0.874 / 1.068 | 0.603 / 0.914 |
| Boundary coefficients | 1.0198 / 1.0100 | 1.341 / 1.523 | 0.837 / 1.191 |

ESS/s here uses 271.670/172.534 s of pooled retained transition time.
The minimum bulk ESS over physical flux coordinates fell from 231.4 to 184.4, and the
largest R-hat increased to 1.0402. Short-chain adaptation variability therefore
matters despite the better group-minimum ESS/s values. Two chains of 200
retained draws do not demonstrate convergence or improved mixing. Longer
scientific runs still require joint amplitude, flux and boundary checks.
Efficiency did not improve at every coordinate: bulk ESS/s fell for 46/264
physical flux coefficients, 47/264 latent flux variables, 1/12 amplitudes and
1/4 boundary coefficients. Tail ESS/s fell for 9/264 physical flux coefficients,
8/264 latent variables and 1/12 amplitudes, with no boundary decreases.

CVM retained runtime was 82.39% amplitude trajectory, 14.84% cache refresh,
0.72% residual installation and 1.74% state trajectory. Further performance
work should prioritize the amplitude numerical kernel over the flux/boundary
update. The local geometry evidence above and short amplitude trees still give
no reason to prefer a full metric. Native Numba support could remove Python
callbacks too, but material extra savings over CVM would require improving numerical
work or allocations; no native-kernel speedup has been measured.

The [compiler profiler](../../scripts/profile_ope152_amplitude_cvm.py) reproduces
the six-anchor microbenchmark and separate sampling processes with the same
arguments and ABBA seed order as the earlier profiler. It pins baseline
`2a440c98`, checks the compiler-only source difference and records complete
source snapshots. CVM's inner timer is bound before graph construction; Numba
is compiled and warmed with stock `perform` before its runtime timer is
installed. Both modes warm the same initial densities without advancing the
chain or RNG. Startup inner-call counts are not comparable between modes.
Frozen original harnesses, manifests, traces and analysis remain under the
local `inversion_outputs/ope152-amplitude-20261021-r4/` directory.

Validation job `19593205` passed all 211 related tests on each of Python 3.12
and 3.13. Its PyMC 6.0 check passed 30 cases and exposed a test-setup omission
in twelve direct density/gradient oracle cases: the older wrapper requires
`set_extra_values(point)` before a direct call. Production sampling cases
passed. Isolated-snapshot job `19593979` validated the one-line test-only
correction: all twelve affected cases passed again on each current Python
version, and all 42 cases passed with PyMC 6.0 / PyTensor 3.0.2. The applied test
file is byte-identical to that tested snapshot. The full documentation build
also succeeded with twelve existing duplicate-object/ambiguous-reference
warnings; the earlier isolated-build missing-download warnings were eliminated
by including the repository's OpenSpec download targets. No specification changed.
The retry completed in 9m22s on `bp1-compute084`, using two CPUs and 8 GiB,
with sampled MaxRSS 5,801,560 KiB.

The documentation build used the frozen implementation snapshot. Subsequent
performance-report text, CSV and profiling-script additions passed link and
table checks separately. The public compiler profiler also passed normalized
source/AST equivalence with the frozen benchmark harness, Ruff, command-line
help, microbenchmark parity and a bounded two-chain sampling smoke check.
