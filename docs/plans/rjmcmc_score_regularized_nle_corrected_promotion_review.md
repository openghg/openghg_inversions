# Corrected score NLE E3 prelaunch review

## Evidence supporting predeclaration

The corrected E2 experiment shows that NLL-only is generally accurate but
that the observation-score objective can produce a comparably accurate fit
when initialization is selected by an independent held-out NLL. In the
two-case `S=4096` canary, this rule selected:

- near-Gaussian initialization 3: evidence error about `0.00523` nat,
  posterior-weighted p99 about `0.0669` nat, gradient error about `0.0718`;
- skewed initialization 2: evidence error about `0.0336` nat,
  posterior-weighted p99 about `0.109` nat, gradient error about `0.00821`.

This is credible enough to predeclare an all-six promotion experiment, but
the near-Gaussian gradient result is above the historical `0.05` gate.
Promotion therefore remains unearned pending the frozen experiment.

## Review findings closed before launch

Independent code, simulator, and oracle reviews initially held the live E3
diff. The implementation now closes the concrete blockers:

- legacy E2 matrix identities and the three-case v1 loader remain distinct;
- all promotion cases, sizes, seeds, optimizer controls, and roles are bound
  in promotion-only matrix identities;
- the generic array launcher derives promotion controls from that committed
  identity rather than historical CLI defaults;
- workers compute model-selection metrics only; reporting and oracle science
  are evaluated only for the selected artifact;
- every selected row preserves exact report/artifact hashes and a manifest of
  all losing starts;
- observation-score improvement must exceed five pooled MCSEs on both
  held-out domains;
- posterior quantiles use within-bin interpolation and each metric has its
  own numerical interpretability flag;
- the all-six oracle has predeclared primary and independent routes plus an
  exact metric-grid preflight;
- per-matrix summaries and cross-matrix certificates are create-only and bind
  exact file bytes, paths, source SHA, matrix IDs, and tags;
- promotion attempts separately bind package/lock runtime identity and
  JAX/XLA/thread/autodiff execution identity;
- exact grids used by selected-artifact scoring and the certifier must replay
  the v2 preflight hashes;
- gradient interpretability gates allocation-order refinement, a three-step
  exact/learned finite-difference ladder, and final-step stability;
- independent column/native numerical errors and support accounting are hard
  gates, and the loader enforces nested case/order/method semantics;
- one shared validator runs before oracle publication and again at promotion
  load, recomputing every nested numerical diagnostic, exact Boolean check
  map, case pass, and top-level pass rather than trusting rehashed flags;
- the certifier accepts exactly the two development summaries or all five
  development/confirmation summaries.

Focused tests cover legacy identity preservation, wrong seeds/specs,
missing/duplicate starts, non-finite NLL, tie handling, selection invariance
to reporting/oracle fields, exact-byte tampering, all-six completeness, the
frozen confirmation seed set, and 25 coherently rehashed semantic/numerical
oracle tamper routes. The final five-module suite passed 68 tests; Ruff,
focused Pyright, shell syntax, and `git diff --check` also passed. Independent
code/provenance and numerical-oracle re-reviews both returned PASS on the
exact diff.

## Verdict

**PASS to commit and run the new all-six oracle, followed by the frozen
development matrices if and only if the oracle passes.**

This is not a promotion verdict. The candidate must pass every predeclared
development and confirmation gate before any protected-data work is
considered.
