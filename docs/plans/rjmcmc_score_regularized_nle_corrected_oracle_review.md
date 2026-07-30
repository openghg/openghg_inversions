# Corrected score NLE oracle and metric review

## Scope

Independent read-only numerical review of the boundary reference,
nonboundary oracle bundle, completion/authentication chain, scientific grid,
and merger treatment of numerically non-interpretable metrics.

## Initial findings and disposition

The primary and independent boundary references were numerically consistent,
but the first certificate did not gate the independent \(-80\) to \(-120\)
lower-tail refinement or primary posterior-summary refinement. The first E2
metric grid also refined only exact evidence and used a finite-grid mask as if
it described continuous retained mass.

The certificate now gates:

- primary order-32/order-64 evidence, posterior mean, SD, median, and interval
  endpoints;
- independent native-coordinate \(-80\)/\(-120\) evidence, mean, and SD;
- primary-versus-independent evidence, mean, and SD;
- outer and maximum inner scaled quadrature error;
- retained prior/posterior mass and posterior-mode inclusion.

Previously audited numerical margins were:

- primary log-evidence refinement: about \(3.69\times10^{-11}\) nat;
- primary/independent agreement: about \(6.25\times10^{-10}\) nat;
- independent lower-tail refinement: about \(8.37\times10^{-7}\) nat.

The E2 grid now refines exact and learned evidence, posterior summaries, and
pointwise errors at 1024, 2048, 4096, and 8192 midpoint prior-CDF strata.
Learned scientific metrics are marked interpretable only if all final
refinement checks pass and the exact grid agrees with the adaptive reference.
Continuous support accounting comes from the adaptive oracle; the merger
excludes non-interpretable scientific values from authoritative aggregation.

Oracle publication is create-only. Its final completion marker binds the
canonical payload, exact pretty-JSON report bytes, full source SHA, and report
path. The attempt loader verifies the exact selected-case catalogue, nested
summary identities/hashes, boundary certificate, and completion binding.

## Final verdict

**PASS for commit, push, and launch.**

Run the corrected oracle first. Submit the compile-canary array only after the
oracle has published a valid exact-byte completion marker from the same clean
detached full-SHA source.
