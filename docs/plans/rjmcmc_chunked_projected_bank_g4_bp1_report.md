# BP1 calibrated projected-bank G4 report

## Decision

The calibrated G4 development experiment is a valid scientific failure for
the finite equal-weight source-bank likelihood at the frozen envelope
\(S\le65536\), \(q\in\{16,32,64,128\}\).  No retained rank passed, so no
development source lock exists.  The three confirmation seeds, G5 clustering,
and G6 likelihood/posterior screen were not run.

This decision does not rely on the small moment and tail misses at the larger
ranks.  Those misses could reasonably be described rather than enforced under
a pragmatic reading of the staged gates.  The decision-driving failure is
likelihood instability by tens to thousands of nat, against the predeclared
0.05-nat median and 0.20-nat 99th-percentile leakage limits.

The result is a hard stop for this finite empirical-mixture envelope, not for
the common native Gamma--Dirichlet model or for simulation after a fixed basis
projection.  It indicates that a raw equal-weight bank is not a usable density
estimator in the retained dimensions at this sample count.

## What was tested

The native model is one common positive Gamma--Dirichlet scaling model.  It is
simulated on the complete native grid and then projected through the frozen
PARIS operator.  The first \(q\) spectrum coordinates are represented by an
equal-weight finite mixture of simulated source locations.  Coordinates
\(q+1{:}r\) use an analytic Gaussian moment-closure complement; this complement
is an approximation diagnostic, not a different native model.

The scientific calibration was fixed before G4:

- modeled European-domain physical-total CV: 0.20;
- GBR physical-total CV: 0.50;
- common native concentration:
  \(\eta=528.618161317525\); and
- root variance: \(0.022861001527515423\), corresponding to root CV
  \(0.15119855001790006\).

The modeled European domain is the complete \(183\times128\) inner grid, not
political EU membership.  The GBR aggregate uses the frozen
`country_fraction` mask.

The likelihood comparator consists of 256 deterministic, observation-blind
prior-predictive states.  Held-out allocation residuals, root masses,
measurement noise, and six outer coefficients come from independent frozen
catalogues.  There is no ground-truth flux field and no observed-data score in
this report.  The realized PARIS `mf` was not read.  The 0.05/0.20-nat limits
measure approximation leakage between finite banks and retained ranks; they
are not absolute error bounds against an unavailable continuous PARIS oracle.

## Provenance

- Branch: `codex/rjmcmc-chunked-projected-bank`
- Scientific execution revision:
  `189427d5ccca9187618ab8be1cc2cf7d7105b216`
- G3 substantive-adjudication revision:
  `1735d0cb5fafd561f33bcb39473c5e0927945ba3`
- G4 serialization-recovery revision:
  `ed93bc37e51a3a0f2dab94df54d75215ab9a45d0`
- Scientific detached worktree:
  `/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc_chunked_projected_bank_189427d5ccca9187618ab8be1cc2cf7d7105b216`
- Recovery detached worktree:
  `/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc_chunked_projected_bank_ed93bc37e51a3a0f2dab94df54d75215ab9a45d0`
- Preserved run root:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_chunked_projected_bank/189427d5ccca9187618ab8be1cc2cf7d7105b216`
- Frozen input SHA-256:
  `24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044`

The reporting-only revisions did not change the scientific input, calibrated
native model, spectrum, source catalogue, \(S\) or \(q\) ladders, validation
grid, or numerical thresholds.  The protected catalogue remained sealed, no
production output was written, and nothing was written to `PARIS_inversions`.

## Upstream engineering status

G0, G1, G2, and the substantive G3 engineering question passed.  G1 selected
projection microbatch \(P=64\).  The G3 adjudication selected allocation chunk
\(C=1024\) using complete earlier evidence plus authenticated current-source
repeats.  Allocation chunk controls temporary memory only and cannot change
the projected result.  The selected projected-bank file SHA-256 is
`aec20f2d3fd1c93c6ba52c2fbc4a84986121debafbad48e3b7e07943911a33a7`.

The interrupted exclusive-node G3 recovery array is not interpreted as a
scientific failure.  Its purpose and the reuse of complete prior evidence are
recorded in
[`rjmcmc_chunked_projected_bank_g3_certifier_recovery.md`](rjmcmc_chunked_projected_bank_g3_certifier_recovery.md).

## What happened in G4

The observation-blind grid passed.  The first development job completed every
expensive calculation but failed while serializing a NumPy Boolean in the
final JSON report.  Its projected-bank and likelihood arrays were preserved.
The narrowly repaired runner wrote Python scalar types without changing any
scientific calculation.  The retry reproduced both expensive files exactly,
published its report, and placed its seed completion marker last.

The table records scheduler results.  MaxRSS is Slurm batch-step MaxRSS.

| Stage | Job | Node | State | Elapsed | MaxRSS |
|---|---:|---|---|---:|---:|
| G4 observation-blind grid | `18213816` | `bp1-compute097` | completed `0:0` | 00:00:39 | 2,187,068 KiB |
| G4 development, preserved serialization failure | `18213953` | `bp1-compute095` | failed `1:0` | 00:55:51 | 2,044,916 KiB |
| G4 development, serialization retry | `18214252` | `bp1-compute051` | completed `0:0` | 00:57:17 | 1,984,500 KiB |

All jobs used ordinary shared nodes.  No whole-node exclusivity was requested
for G4.

## Decision-driving likelihood results

For each \(q\), the first two result columns compare nested prefixes of the
same source bank with the \(S=65536\) bank.  The third compares each
\(q<128\) likelihood with \(q=128\) at \(S=65536\).  Every entry is
median / 99th-percentile absolute log-likelihood change in nat; the limits are
0.05 / 0.20 nat.

| \(q\) | \(S=16384\) vs \(65536\) | \(S=32768\) vs \(65536\) | \(q\) vs 128 |
|---:|---:|---:|---:|
| 16 | 10.896 / 1523.708 | 0.693 / 126.013 | 288.711 / 3021.077 |
| 32 | 26.248 / 1849.592 | 0.693 / 214.463 | 199.901 / 1101.384 |
| 64 | 31.623 / 2121.445 | 0.693 / 249.866 | 83.015 / 560.294 |
| 128 | 34.050 / 2223.190 | 0.693 / 346.490 | 0 / 0 |

The repeated \(0.693147\approx\log 2\) median after doubling \(S\) is
diagnostic.  For at least half of the validation states, the added half-bank
contributes negligible density, so the unchanged accumulated mixture mass is
divided by twice as many equal-weight components.  At other states, rare new
components dominate and move the likelihood by hundreds or thousands of nat.
Exact prefix identity and exact weight normalization rule out a missing-row or
normalization implementation error.

## Secondary diagnostics and technical controls

The \(q=16\) sample moments and all nested tail checks passed.  At larger
ranks, the \(S=65536\) normalized-mean maximum was `0.0236437` against `0.02`;
the covariance diagnostic was `0.0659014` at \(q=64\) and `0.0757023` at
\(q=128\), against `0.06`.  The only nested-tail failure was the
\(+3\)-standardized-coordinate probability difference for
\(S=16384\) versus \(32768\): `0.00241089` at \(q=32\) and `0.00268555`
at \(q=64,128\), against `0.002`.  All \(S=32768\)-versus-\(65536\) tail
comparisons passed.

These secondary misses are small enough that they are not used as the reason
to stop.  By contrast, the likelihood leakage is qualitatively and
quantitatively decisive.

The following controls all passed:

- exact selected-G3 rebuild, including binary file identity;
- exact nested row and coordinate-prefix identities;
- finite likelihood and positive-variance support;
- exact equal-weight normalization;
- zero original-versus-translated likelihood difference, against tolerance
  `2.357670041290327e-8` nat;
- strict canonical JSON replay after the scalar-serialization repair; and
- explicit `realized_mf_used=false`,
  `protected_catalogue_accessed=false`, and
  `production_output_written=false`.

The create-only development certifier recorded `passed=false`,
`passing_suffix=[]`, `selected_rank=null`, and
`next_gate=terminal-G4-development-hard-stop`.  It correctly did not publish
`g4/development/G4_DEVELOPMENT_COMPLETE.txt`.

## Primary artifact checksums

The table identifies the decision-driving artifacts under the preserved run
root.

| Artifact | SHA-256 |
|---|---|
| `g3/prior_complete_g3_recertified.json` | `d4825a511f06c93e6f6dcbe3beee1c2b890f9633b8e2fec0b60a5794741cff3e` |
| `g3/g3_decision.json` | `8ce88a1240e7734e765c2ef38b07a248f8b5d6cc3bf5a5ce44c8e12dbdd00451` |
| `g3/G3_COMPLETE.txt` | `164e7ac2ca5bc78a0f245143a5d720fa1a6e3f32786a56362afec5c7138c7551` |
| `g4/grid/grid_manifest.json` | `846bbdc9cb6991ed2facc595c13a0c2132a655cf0050ffe188b232352ddf7f5f` |
| `g4/grid/G4_GRID_COMPLETE.txt` | `ba287a4e811d30df4672f26f1acde14697353c1ed7380be61e0b6d94e9af6df5` |
| `g4/development/seed731_retry1/projected_locations.npy` | `aec20f2d3fd1c93c6ba52c2fbc4a84986121debafbad48e3b7e07943911a33a7` |
| `g4/development/seed731_retry1/log_likelihood.npy` | `500772c940b5804db42f878159be3996392aae564ff00758da03db18aee9b74f` |
| `g4/development/seed731_retry1/seed_report.json` | `338a161a3dc269018b27a5de25ebce4e08131476c31ff4f894d11a11b922c8bc` |
| `g4/development/seed731_retry1/G4_SEED_COMPLETE.txt` | `b75f4c53c5382a376d58f644cb997a53281939c6e601a923815ef13ae1ecd2aa` |
| `g4/development/development_decision.json` | `7f48b6505b1936feb6eee05de178100699bdb831f9bc45d898da9f3ea6b5c1fb` |

## Interpretation and follow-up

The simulation and projection machinery worked.  Low-order empirical moments
and most distributional diagnostics were reasonable.  The failure occurs
when the resulting point cloud is used directly as a high-dimensional
equal-weight density estimator.  Increasing \(q\) reduces Gaussian-complement
leakage but worsens sparse-mixture likelihood convergence; decreasing \(q\)
improves bank convergence only superficially and leaves very large
\(q\)-truncation leakage.  No tested rank reconciles the two.

Therefore:

1. do not spend three more node-hours on confirmation seeds;
2. do not cluster this unstable empirical likelihood or run the G6 screen;
3. preserve the successful constructor as a simulator for the projected
   native marginal; and
4. treat any learned-density or neural-likelihood continuation as a new,
   observation-blind, predeclared architecture rather than an extension of
   this failed source-bank lock.

The exact scientific limit remains invariant to computational partition and
\(K\).  Approximation differences remain leakage diagnostics and must not
become data-dependent basis weights.
