Fixed-OU experiment evidence
============================

The maintained algorithm explanation is :doc:`fixed_ou_likelihood`. The
following files are archived at the immutable pre-cleanup revision
`3d765dbc <https://github.com/openghg/openghg_inversions/tree/3d765dbc216636c2986e315332e50feac59a7506>`_.
They retain the experimental methods, input/source hashes, measured results
and limitations, but are not maintained benchmark tools. No general benchmark
driver is retained: these scripts extract and patch historical implementations.

The tracked evidence is public. Large prepared inputs, raw chain traces and
some one-off cluster harnesses were never committed; their provenance is
recorded in the reports and they remain cluster artifacts. The archive does
not claim a self-contained public replay of those data.

Reports and timing tables
-------------------------

* `ope152_amplitude_cvm_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_amplitude_cvm_timings.csv>`_
* `ope152_amplitude_followups.md <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_amplitude_followups.md>`_
* `ope152_amplitude_nuts_performance.md <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_amplitude_nuts_performance.md>`_
* `ope152_amplitude_stability_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_amplitude_stability_timings.csv>`_
* `ope152_amplitude_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_amplitude_timings.csv>`_
* `ope152_cached_ou_performance.md <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_cached_ou_performance.md>`_
* `ope152_cached_ou_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_cached_ou_timings.csv>`_
* `ope152_sum_squares_guarded_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_sum_squares_guarded_timings.csv>`_
* `ope152_sum_squares_implementation_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_sum_squares_implementation_timings.csv>`_
* `ope152_sum_squares_performance.md <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_sum_squares_performance.md>`_
* `ope152_sum_squares_timings.csv <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/docs/plans/ope152_sum_squares_timings.csv>`_

Historical profiling scripts
----------------------------

* `profile_ope152_amplitude.py <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/scripts/profile_ope152_amplitude.py>`_
* `profile_ope152_amplitude_cvm.py <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/scripts/profile_ope152_amplitude_cvm.py>`_
* `profile_ope152_amplitude_geometry.py <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/scripts/profile_ope152_amplitude_geometry.py>`_
* `profile_ope152_fixed_ou.py <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/scripts/profile_ope152_fixed_ou.py>`_
* `profile_ope152_sum_squares.py <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/scripts/profile_ope152_sum_squares.py>`_
* `profile_ope152_sum_squares_interleaved.py <https://github.com/openghg/openghg_inversions/blob/3d765dbc216636c2986e315332e50feac59a7506/scripts/profile_ope152_sum_squares_interleaved.py>`_

Review and prototype records
----------------------------

* `#922 <https://github.com/openghg/openghg_inversions/pull/922>`_: full-width cache solve.
* `#926 <https://github.com/openghg/openghg_inversions/pull/926>`_: guarded sum-of-squares cache.
* `#928 <https://github.com/openghg/openghg_inversions/pull/928>`_: amplitude residual reuse and compiled density.
* `#942 <https://github.com/openghg/openghg_inversions/pull/942>`_: inherited evaluator stability fix and repeat rank-256 timings.
* `#939 <https://github.com/openghg/openghg_inversions/pull/939>`_: closed, unmerged native Numba prototype. Its implementation, tests, reports and timing data remain at
  `5ed1ff90 <https://github.com/openghg/openghg_inversions/tree/5ed1ff90d47cca4ea58ac5c21a262331b669c1b2>`_.

The archived timings compare specific machines, ranks, priors and conditioning
states. They do not establish general sampling efficiency or override the
regression tests retained in the current tree. Backend-independent tests
recovered from #939 remain in the maintained evaluator tests; native-only
compiler/lifecycle tests remain with that prototype.
