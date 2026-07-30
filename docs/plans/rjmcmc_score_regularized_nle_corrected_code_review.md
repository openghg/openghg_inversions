# Corrected score NLE E2 code and provenance review

## Scope

Independent read-only review of the exploratory objective, autodiff schedule,
loss calibration diagnostics, frozen arrays, attempt identity, artifact
publication, merger authentication, and initial Slurm resources.

## Initial findings and disposition

The initial implementation was held for these launch blockers:

- observation-score training used reverse-over-reverse differentiation;
- scalar partial-score risk was divided by retained rank;
- component variances and parameter-gradient norms were not measured before
  applying weights;
- a duplicated vectorized likelihood lacked parity with the authenticated
  public artifact evaluator;
- completion markers did not bind actual report bytes;
- the \(S=16384\) mappings and exact array-task provenance were absent.

The corrected observation score uses coordinate `jax.linearize`; the outer
parameter gradient is reverse-over-forward. The four-task compile canary
covers partial and observation objectives for both the \(q=1\)
near-Gaussian and \(q=3\) skewed architectures. Scalar partial risk is
Fisher-scaled without division by \(q\); observation risk uses
per-coordinate Fisher scales and then averages coordinates.

Every attempt records initialization row means/variances and parameter
gradient norms for NLL, partial score, and observation score before auxiliary
weights are applied. The candidate identity contains the rule that produces
the scales, not realized sample-dependent values. Optimizer streams are
paired across ablations by case, initialization, and stage position.

The vectorized scientific evaluator must match the public artifact evaluator
on representative tail and central inputs. Array identity binds matrix name,
task index/count, and exact frozen row. The merger revalidates that mapping,
source SHA, attempt tag, output path, exact completion schema, report payload,
report file, artifact metadata, and serialized artifact file.

The committed catalogue contains:

- four compile-canary tasks;
- 16 small overfit tasks;
- 36 \(S=4096\) tasks;
- eight optional observation-score tasks;
- separate 12-task \(S=16384\) NLL, partial-score, and curriculum matrices.

## Final verdict

**PASS for commit, push, and launch.**

The first array should be submitted explicitly as `--array=0-3` with one
epoch and zero patience. One CPU, 8 GiB, one hour, shared-node execution is a
reasonable upper bound for this canary; its accounting evidence must guide
later requests.
