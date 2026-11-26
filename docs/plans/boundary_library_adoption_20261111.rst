:orphan:

Boundary-library adoption record (11 November 2026)
===================================================

This dated record preserves review evidence from adopting the boundary
component library in the CO₂ and linked CO₂/O₂ recipes
(`#1013 <https://github.com/openghg/openghg_inversions/pull/1013>`_, recorded
at revision ``952949d9``). The counts and probes describe that revision; they
are not maintained and will drift as the builders change. The durable
extension and locality rules are in
:doc:`../development/rhime_model_development`.

Builder signatures
------------------

AST inspection of the concrete builder signatures gave the following counts
before and after the adoption:

.. list-table:: Builder parameters, including keyword-only arguments
   :header-rows: 1
   :widths: 55 15 15 15

   * - Builder
     - Before
     - After
     - Boundary inputs
   * - ``build_co2_model``
     - 23
     - 15
     - 9 → 1
   * - ``build_co2_cached_sigma_model``
     - 24
     - 16
     - 9 → 1
   * - ``build_co2_o2_model``
     - 23
     - 15
     - 9 → 1
   * - ``build_co2_o2_cached_sigma_model``
     - 23
     - 15
     - 9 → 1

To reproduce a count, parse each builder module with ``ast.parse``,
select its top-level ``FunctionDef`` and count ``posonlyargs + args +
kwonlyargs``. The only boundary parameter is ``boundary``; unrelated arguments
are unchanged.

Extension probes
----------------

A dry-run extension with a new supported boundary prior reached these owners:

* A new module under ``models/boundary/``: define the choice (with ``bind``) and
  the bound record (with ``add`` and ``outputs``), inheriting
  ``CenteredBoundary`` for a centered family. ``Choice`` and ``Bound`` are
  structural protocols, so no shared list of families changes.
* ``rhime/co2/configuration.py`` and, for channel-specific syntax,
  ``co2_o2_configuration.py``: resolve user selections into component inputs.
* Scientific component tests and the affected configuration tests: establish
  the equation, fixed-state behavior, units, full density and gradients.

The exercised extension probes are
``tests/test_co2_boundary_affine_integration.py`` and
``tests/test_boundary_output.py``. The frontend handoff is exercised in
``tests/test_co2_bound_frontends.py`` with raw binding forbidden, absent raw
operators, shared delayed payloads and independent concentration equations for
ordinary/cached single/linked fits. The embedding probes exercise a one-dimensional native response
and a matrix with unsorted coefficient labels into interleaved joint rows,
reject reversed observation rows before reading lazy cache payloads, and accept
independent coefficient blocks with different labels on identically named axes.
The cached conversion aligns only observation rows; it does not align separate
coefficient axes. Independent curtain corrections replay through the same
scientific correction role. Unsampled additive anomalies are registered as
zero model data, so replay reads every declared role from the saved draws or
constants.

A runnable dry-run temporarily added ``StudentT(nu, mu, sigma)`` to the
configuration prior table in memory. Existing component prior construction
handled it directly. Ordinary and cached graphs had finite full log densities,
and reconstructed boundary contributions closed for three prior draws and two
posterior draws. The only temporarily extended module was
``rhime/co2/configuration.py``; no likelihood, cache or output module changed.
This probe does not add Student-t to the released configuration choices.
