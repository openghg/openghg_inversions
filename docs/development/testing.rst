Keeping the test suite fast
===========================

The default suite has about 5,100 cases. Most of them are cheap: in a serial
run on 11 November 2026 (Python linker, one process), 4,125 cases took under
0.05 s each and 31 s in total. Most of the runtime came from a few dozen tests:

.. list-table::
   :header-rows: 1

   * - Per-test cost
     - Cases
     - Share of the 634 s run
   * - 5 s or more
     - 24
     - 49 %
   * - 1 s to 5 s
     - 45
     - 14 %
   * - under 0.05 s
     - 4,125
     - 5 %

The expensive tests fall into three groups:

* Tests that start a fresh Python process. Each child pays the package import,
  up to about 9 s on a laptop, because ``openghg_inversions.rhime`` imports
  PyMC, the CO2 recipes and parts of OpenGHG. About 35 such tests took 136 s
  (22 %).
* Installed-command workflows, which run several CLI subprocesses with real
  sampling.
* Statistical-accuracy checks, which draw thousands of samples to compare
  posterior moments against an independent oracle.

The rules below keep new work out of those groups unless it needs to be
there, and keep the cheap tests from multiplying without adding protection.

Running tests while iterating
-----------------------------

Run the tests that own the code you are changing, not the whole suite. Run
the relevant broader groups before handing off a change, and leave the full
suite and tox to the final check (see the Testing section of ``AGENTS.md``).

.. list-table::
   :header-rows: 1

   * - Area
     - Paths
   * - Configuration and request validation
     - ``tests/test_rhime_configuration.py tests/test_rhime_resolved_run.py
       tests/test_public_run_api.py tests/test_unsupported_tracer.py``
   * - Model components and likelihoods
     - ``tests/models``
   * - Basis functions
     - ``tests/test_basis_*.py tests/test_constrained_basis_algorithm.py
       tests/test_weighted_basis_algorithm.py tests/test_bc_basis_*.py``
   * - Surface acquisition and preparation
     - ``tox -e py312-surfacePilot`` lists the owning modules
   * - Stage handoffs and replay
     - ``tests/test_*handoff*.py tests/test_staged_workflow.py``
   * - CO2 and CO2/O2 recipes
     - ``tests/test_co2_*.py tests/test_rhime_co2_*.py``
   * - Postprocessing and outputs
     - ``tests/postprocessing tests/test_output_contract.py``

Useful pytest options while iterating:

* ``-x --lf`` reruns the last failures and stops at the first new one.
* ``-k <expression>`` selects tests by name within a group.
* ``--durations=15`` reports the slowest cases, so you notice a new expensive
  test before review does.

PyTensor backend and precision
------------------------------

Pytest uses PyTensor's Python linker by default (``tests/_pytensor_defaults.py``
adds ``linker=py`` to ``PYTENSOR_FLAGS`` when no linker is set). This matches
CI and avoids native compilation, which made graph-heavy modules about 40 %
slower. Compiled coverage is explicit: tests marked ``compiled_backend`` are
rerun with ``PYTENSOR_FLAGS=linker=numba`` in a fresh serial process, and any
explicit ``linker`` entry in ``PYTENSOR_FLAGS`` is kept. See the README for the
exact command.

Focused runs can currently fail under the package's float32 default, depending
on which modules import PyTensor first (`#770
<https://github.com/openghg/openghg_inversions/issues/770>`_). If a test that
passes in the full suite fails in a focused run with a float32/float64
tolerance mismatch, that is #770, not your change. Do not make a test pass by
forcing float64 globally; precision-sensitive tests should state their
precision explicitly.

Writing tests that stay cheap
-----------------------------

**One rule, one test, at the layer that owns it.** Test a validation rule
where it is implemented, usually configuration or request loading. Do not
re-test it through each public runner with data acquisition patched out. A
runner test only needs to show that validation runs before work starts, and
one input is enough for that.

**Do not multiply independent parametrize axes.** A product such as
route × bad value × file format runs the same branch many times. Before adding
an axis, check that each value reaches different production code. Then:

* test every input against one route, and every route against one input;
* use one representative per failure class (for example negative, ``nan`` and
  wrong type), not several values that fail the same check;
* for numerical tests with three or more axes, use a small hand-picked table or
  pairwise coverage rather than the full product;
* put a case that expects a different error in its own test.

**Keep retired-feature guards in one table.** When an option or alias is
removed, add it to one parametrized "removed names are rejected" test per
layer instead of writing a new test for each removal. Remove such guards once
the retirement has shipped in a release, unless the removed name is likely to
reappear in users' configuration files.

**Assert behaviour, not structure.** Do not assert ``__module__``, ``__all__``
contents, absent private names, signatures or mock call order when a returned
value, saved file or raised error can be checked instead.

**Check file formats and metadata without real sampling.** Tests of labels,
units, manifests, provenance or output layout should use
``tests.helpers.make_trace`` or saved samples. Keep real sampling for tests
whose subject is the sampler or the posterior.

**Share expensive setup deliberately.** Build an expensive, read-only result
(a preparation, a compiled function, a fitted trace) once in a module-scoped
fixture, and copy it into ``tmp_path`` when a test writes to it. Compile a
PyTensor function once per test, not once per evaluation point. Do not widen
fixture scope for mutable inputs.

**Justify per-family copies.** A rule shared by the standard, multisector,
nested, CO2 and CO2/O2 families belongs in one test over the shared code,
parametrized by family where needed. A per-family test should check something
that only that family implements. Where the families still implement the same
rule separately, prefer moving the rule into shared code over copying its
test again.

Tests that are expensive on purpose
-----------------------------------

Some tests need real processes or many draws. Keep them, but make the cost
visible and bounded:

* **Fresh-process tests.** State in the docstring which import-time or
  process-boundary behaviour the test protects. Check several modules or
  conditions in one child process rather than starting one per parameter.
* **Statistical-accuracy tests.** Use one representative case in the default
  run and mark the rest ``slow``. Draw counts above about 100, quadrature
  oracles and MCSE gates are signs that a test belongs in this group. Say in the
  docstring roughly how long it takes.
* **Installed-command workflows.** Keep one real run per distinct entry point
  and route, and check detailed output contracts in cheaper tests.

``slow`` tests are excluded by default (``addopts`` in ``pyproject.toml``) and
run with ``-m slow``.

Reviewing test changes
----------------------

When a change adds tests, run its new tests with ``--durations`` and check:

* each new parametrize axis reaches distinct production code;
* any new case slower than about 1 s has a reason in its docstring;
* no new fresh-process, sampling or installed-command test duplicates an
  existing one.
