# BP1 chunked projected-bank execution addendum

## Scope

This addendum freezes the choices needed to execute G0--G3 of
[`rjmcmc_chunked_projected_bank_hpc_test_plan.md`](rjmcmc_chunked_projected_bank_hpc_test_plan.md).
It was written before any G1--G3 result was inspected.

The target remains the conditional pushforward of one common native
Gamma--Dirichlet root model.  It is not an RJ likelihood, a comparison among
partitions, or a source of data-dependent basis weights.  The leading
projected bank is an approximation to the non-Gaussian residual marginal;
spectrum coordinates beyond the stored rank remain the analytic Gaussian
moment-closure complement.

## Engineering-only concentration

The common additive Dirichlet total is fixed to

```text
eta = 100
```

for G0--G3 only.  This is the lower of the two historical resource-probe
scales and is chosen a priori as the higher-variance engineering stress case.
No comparison with \(\eta=500\) is authorized.  This choice is dimensionless
and is not atmospheric mole-fraction concentration.

It is not a scientific concentration lock.  G4 remains barred until a
scientific common native \(\eta\) is supplied independently of results.

## Software-only execution repairs

The first G1 attempt on source
`9fd067885c8c14e62970613fa0f8b99009ba43f0`, Slurm job `18194997`, failed
before publishing a G1 report because this addendum initially imposed
bitwise equality across different \(P\) values.  That overconstraint
contradicted the already-committed background contract: fixing \(P\) makes
the result invariant while varying \(C\), but changing \(P\) changes the BLAS
row shape and can change final rounding.  The repair below requires exact
replay at each fixed \(P\) and applies the already-frozen float64 parity
tolerance across \(P\).  No new numerical threshold was chosen, and no PARIS
input or realized residual had been accessed by G1.

The earlier job `18192814` never started because it inherited Slurm account
`default`, which the `compute` partition denies.  The committed batch scripts
now name the established BP1 account `chem007981` explicitly.  Both failures
and their logs remain preserved under their original run root.

The first G3a attempt on source
`ddad33bf886fc94981e328806456ecab305b663b`, Slurm job `18195041`, stopped
before constructing either projected bank or publishing a G3a report.  G3a
authenticated and loaded the frozen PARIS input and authoritative spectrum,
then the strict reader rejected the otherwise valid G1 report: its
projection-size parity map had been emitted from integer Python keys, whose
ordering changes after the mandatory JSON conversion to strings.  The repair
emits those control keys as strings before canonical serialization.  It
changes no numerical path, threshold, scientific input, or stage decision.
The failed job and its logs remain preserved under their original run root.

## Numerical controls

The G1 projection-microbatch ladder is

```text
P = 64, 128, 256
```

Each candidate is run three times on the same moderate synthetic operator.
Every candidate must replay bitwise at its fixed \(P\).  Across different
\(P\) values, projected arrays must meet the frozen v2/v3 float64 parity
tolerance below; different BLAS row shapes are not required to be bitwise
identical.  This matches the background contract, which guarantees bitwise
invariance when \(P\) is held fixed while \(C\) varies.  The locked value is
the lowest median elapsed time among parity-passing candidates, with smaller
\(P\) as the exact-tie break.  For deliberately tiny G1 cases with \(S<P\),
the effective microbatch is \(\min(P,S)\), held fixed across allocation
chunks for that case.

The v2/v3 parity tolerance is frozen as

\[
32\,\epsilon_{64}\,n_{\rm cell}
\max(1,\max |z_{\rm v2}|).
\]

Here juxtaposition in the executable formula is multiplication:

```text
32 * float64_epsilon * native_cell_count * max(1, max_abs(v2))
```

Maximum absolute and representable-float (ULP) differences are both
reported.  Allocation-chunk comparisons at fixed \(P\) must be bitwise
identical.

G3a uses \(S=256\), \(q=32\), seed 731, and the predeclared allocation-chunk
ladder \(C=64,128,256\).  Only candidates satisfying \(C\ge P\) are legal
after the G1 lock; all legal candidates are run and must be bitwise
identical.

G3b uses \(S=65536\), \(q=128\), seed 731, and
\(C=1024,2048,4096,8192\).  One \(S=4096,C=1024\) warm-up is excluded from
selection.  Every G3b candidate has three serial authoritative timed
replicates.  A chunk passes when all three jobs:

- complete normally;
- use zero swaps;
- have Slurm `MaxRSS` at most 12 GiB;
- finish the constructor within 45 minutes; and
- publish internally replayed binary arrays.

All completed candidates must have one common projected-array and binary-file
digest.  At least one allocation chunk must pass.  The selected \(C\) has the
lowest median of its three constructor times among passing chunks, with
smaller \(C\) as the exact-tie break.

## Spectrum audit

G2 publishes one authoritative complete spectrum.  A second-node
construction is an audit only.  Context arrays and native identities must be
exact.  The audit compares eigenvalues and reconstructed covariance, not raw
basis columns, because nearly degenerate eigenspaces may rotate.

The frozen absolute tolerances are:

```text
eigenvalues:
128 * float64_epsilon * r * max(1, max_abs(authoritative eigenvalues))

reconstructed covariance:
256 * float64_epsilon * observations
    * max(1, max_abs(authoritative covariance))
```

The audit bundle cannot replace the authoritative bundle even if it passes.

## Stop rule

No G4 source/science lock will be attempted in this execution.  The required
observation-blind G4 threshold supplement is absent, and the scientific
common native \(\eta\) has not been supplied.  A successful G3 is therefore
the terminal engineering result for this run.  No extra rank, sample, chunk,
component, mixture, flow, NLE, or posterior screen may be selected from these
results.
