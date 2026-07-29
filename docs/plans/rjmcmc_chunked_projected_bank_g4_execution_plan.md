# BP1 execution plan: scientific calibration through G4

## Scope

This is the pre-result execution addendum for
[`rjmcmc_chunked_projected_bank_hpc_test_plan.md`](rjmcmc_chunked_projected_bank_hpc_test_plan.md).
It authorizes a fresh G0--G4 execution of the single-root
Gamma--Dirichlet model using the scientific calibration and numerical gates
frozen in
[`rjmcmc_chunked_projected_bank_g4_threshold_supplement.md`](rjmcmc_chunked_projected_bank_g4_threshold_supplement.md).

The complete pushed Git commit containing this plan is the candidate.  Its
40-character SHA gets a clean detached worktree and a new create-only run root:

```text
/group/chem/acrg/brendan_for_codex/
rjmcmc_chunked_projected_bank/<full-SHA>
```

No artifact from an earlier \(\eta=100\) engineering run is copied, linked, or
relabelled.  The only external scientific input is the authenticated frozen
NetCDF with SHA-256
`24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044`.
Nothing is written to `PARIS_inversions`, and no protected catalogue is
opened.

## Frozen scientific controls

```text
modeled European-domain total CV: 0.20
GBR total CV:                     0.50
common native eta:                528.618161317525
root variance:                    0.022861001527515423
root Gamma shape and rate:        43.742615510366136

S ladder:                         16384, 32768, 65536
q ladder:                         16, 32, 64, 128
development seed:                 731
confirmation seeds:               1877, 4099, 8317
```

The driver recomputes the calibration from exactly aligned
`nominal_weight`, `prior_flux`, `grid_cell_area`, and frozen GBR
`country_fraction` fields.  The complete \(183\times128\) inner grid is the
“modeled European-domain total”; it is not a political EU aggregate.

## Ordered hard gates

1. Run G0 on a quiet login node with `run_preflight.sh`.
2. Submit G1 with `run_g1.sbatch` on an exclusive node so its timing-only
   \(P\) selection is not confounded by unrelated workloads.
3. Submit the authoritative G2 spectrum with `run_g2.sbatch`.
4. Submit `run_g2_audit.sbatch` on a node distinct from the authoritative G2
   node.
5. Submit G3a with `run_g3_prefix.sbatch`.
6. Submit the excluded warm-up with `run_g3_warmup.sbatch`.
7. Prepare directories with `prepare_g3_array.sh`, then submit one sequential
   exclusive-node resource array, `--array=0-11%1`, with
   `run_g3_bank.sbatch`.
8. Submit `run_g3_certify.sbatch`.  Stop if it produces no passing G3
   certificate.
9. Create the G4 prior-predictive grid with `run_g4_grid.sbatch`.
10. Resolve `repeat0` for the G3-selected \(C\), set its manifest as
    `BANK_G3_REFERENCE`, and submit `run_g4_development.sbatch`.
11. Submit `run_g4_development_certify.sbatch`.  A nonzero exit with a
    complete decision report is the predeclared terminal scientific hard stop
    if no all-larger passing \(q\) suffix of length at least two exists.
12. Only after a passing development certificate, submit one confirmation
    array, `--array=0-2`, with `run_g4_confirmation.sbatch`.
13. Submit `run_g4_certify.sbatch`.  A passing all-seed certificate publishes
    `G4_SOURCE_LOCK.txt` last; any individual or cross-seed gate failure is a
    terminal G4 hard stop.

Every potentially long Slurm submission is registered through
`slurm-wakeup`.  Homogeneous repetitions use the two arrays above.  Each
subsequent stage is submitted only after the preceding wake event has been
diagnosed.

## Resource declarations

| Stage | CPUs | Memory | Time | Isolation |
|---|---:|---:|---:|---|
| G1 | 1 | 4 GiB | 30 min | exclusive node |
| G2 authoritative/audit | 1 | 4 GiB | 30 min | allocated cores |
| G3a/warm-up | 1 | 4 GiB | 30 min | allocated cores |
| G3 resource array element | 1 | 16 GiB | 1 h | exclusive node |
| G3 certifier | 1 | 2 GiB | 30 min | allocated cores |
| G4 grid | 1 | 4 GiB | 30 min | allocated cores |
| G4 source-seed array element | 1 | 16 GiB | 2 h | allocated cores |
| G4 certifiers | 1 | 4 GiB | 30 min | allocated cores |

G3 chooses \(C\) only from the frozen resource ladder and requires identical
projected outputs.  G4 consumes that selected \(C\) and the G1-selected
projection microbatch \(P\); neither changes the mathematical source
catalogue.

The isolated timing recovery is specified in
[`rjmcmc_chunked_projected_bank_g3_certifier_recovery.md`](rjmcmc_chunked_projected_bank_g3_certifier_recovery.md).
The create-only seed-731 JSON publication recovery is specified in
[`rjmcmc_chunked_projected_bank_g4_serialization_recovery.md`](rjmcmc_chunked_projected_bank_g4_serialization_recovery.md).

## Interpretation and stop rule

G4 compares nested finite-source prefixes, retained-rank prefixes, and
independent scrambles on an observation-blind prior-predictive grid.  Its
0.05-nat median and 0.20-nat 99th-percentile limits are finite-approximation
stability limits, approximately 5% and 22% likelihood-ratio changes.  They are
not absolute accuracy claims against a nonexistent exact 23,424-cell PARIS
quadrature oracle.

No failed gate permits changing \(\eta\), the root prior, \(S\), \(q\), seeds,
thresholds, or the grid.  No flow, neural likelihood, extra component/rank
ladder, posterior, clustering, or RJ calculation is authorized by this
execution plan.
