Nested-domain model family
==========================

The nested-domain family runs one emissions source across two atmospheric
transport grids: a surrounding outer domain and a higher-resolution inner
domain. RHIME masks their spatial overlap, constructs an independent basis and
state vector on each native grid, and fits both contributions to one set of
observations.

The outer and inner names describe transport domains, not groups within one
basis. Use :doc:`grouped_basis_layout` when one transport grid instead needs
basis regions of different sizes or roles.

Current support
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - Nested-domain RHIME
   * - Intended use
     - One emissions source represented by overlapping outer and inner
       footprint grids at different spatial resolutions.
   * - Public entry points
     - :func:`openghg_inversions.rhime.run_nested` for a complete run,
       and ``run_nested(config=config, prepared=nested)``
       for explicit prepared outer and inner inputs.
   * - Configuration and command line
     - Supports the RHIME INI schema and
       ``rhime run --model nested``. The generic staged commands do
       not currently accept this family.
   * - Outputs
     - ``output_format="none"`` returns ``NestedOutput`` with separate outer
       and inner scientific views without writing files or formatted
       products. ``output_format="paris"`` can additionally write separate
       outer- and inner-grid flux products plus one shared concentration
       product. Other output formats are rejected because they assume one
       spatial grid.
   * - Current model boundary
     - One standard flux sector. When boundary conditions are enabled, their
       contribution comes from the outer domain; multisector nested runs and a
       second inner boundary condition are not supported.
   * - Shared spatial support
     - Every retained site has both domains, with the same indexed outer grid
       and the same indexed inner grid across sites. Outer-only sites remain
       deferred in GitHub issue #897 pending a scientific decision about their
       interior emissions states and output interpretation.

Continue with the nested-domain guide for configuration, basis construction,
time alignment, prepared inputs, and output details.

.. toctree::
   :maxdepth: 1

   nested_domains
