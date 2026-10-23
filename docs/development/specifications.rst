Development guidance and behavioral specifications
=====================================================

Use :doc:`rhime_model_development` and :doc:`architecture_principles` to decide
how to keep scientific code readable and responsibilities local. Use scoped
capability specifications for observable behavior that a caller or artifact
consumer can rely on. Usage documentation explains how to use the implemented
interfaces. These sources serve different reader needs.

To find the requirements for a behavior you are changing:

1. Locate the boundary in the map below and read the specification's scope,
   requirements, and scenarios alongside its usage reference.
2. Check the proposal, review record, and delivery tasks. Review status and
   implementation status are separate: an accepted or finalized design can
   precede implementation.
3. Confirm the supported route against the implementation and tests for the
   version you are changing. A proposed requirement or a checked planning task
   alone does not establish delivered behavior.

Durable contracts under ``openspec/specs/`` describe delivered behavior. Proposed
requirements and amendments remain under ``openspec/changes/`` until implemented
and validated, even when their design is approved. Delivery status and review
history belong in the change records and the map below, not in durable specs.

Current contracts and proposals
--------------------------------

The following map was reviewed against ``devel`` revision ``7269f50f`` on
12 October 2026. It covers RHIME selection, composition, configuration, and
reconstruction boundaries; it is not a complete package API specification.
The configuration entry was amended on 13 October 2026 during review of PR 867.
The resource-I/O and HBMCMC entries were updated the same day for the accepted
0.8 implementation and durable-spec promotion in PR 869.
The 23 October 2026 #876 amendment separates configuration choices, numerical
input behavior and saved handoff guarantees below. Its code is implemented on
the working branch; CI acceptance remains pending. The broader handoff change
remains active and is not complete merely because these owners are identified.

.. list-table:: Behavioral ownership and delivery status
   :header-rows: 1
   :widths: 25 45 30

   * - Boundary
     - Authoritative requirement source
     - Delivery and reader reference
   * - Public model selection and aliases
     - :download:`RHIME model selection
       <../../openspec/specs/rhime-model-selection/spec.md>`
     - Baseline of delivered behavior; :doc:`/usage/rhime` and
       :doc:`/usage/cli`. The :download:`CO2 selector/runner amendment
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/rhime-model-selection/spec.md>`
       includes the merged `PR 886 <https://github.com/openghg/openghg_inversions/pull/886>`_
       runner and the OPE-214 stack for model-driven fixed-OU inference and shared handoffs.
   * - Registered coordinates and custom likelihood composition
     - :download:`RHIME model composition
       <../../openspec/specs/rhime-model-composition/spec.md>`
     - Baseline of the named delivered interfaces;
       :doc:`/usage/customising_rhime`. Covariance support remains recipe-specific.
   * - Requested options and phase resolution
     - :download:`Resolve RHIME configuration
       <../../openspec/specs/rhime-configuration/spec.md>`
     - Delivered grouped configuration and independent phase resolution,
       implemented in `PR 866 <https://github.com/openghg/openghg_inversions/pull/866>`_.
       The #876 amendment retains option ownership, defaults, accepted choices
       and prepared-fact binding here; numerical behavior and artifact guarantees
       have the owners below.
       Current usage: :doc:`/usage/rhime_configuration`.
       Configuration serialization/INI writing remains with `issue 814
       <https://github.com/openghg/openghg_inversions/issues/814>`_.
   * - Scientific preparation and model-input operands
     - :download:`RHIME scientific inputs
       <../../openspec/specs/rhime-scientific-inputs/spec.md>`
     - Owns observation selection/error preparation, retained alignment, raw
       spatial BC inputs, model-time BC periods and consumed minimum-error
       operands. #876 code is implemented on the working branch; CI acceptance
       is pending. Review and delivery evidence stays in the
       :download:`alignment change tasks
       <../../openspec/changes/move-alignments-to-model-boundary/tasks.md>`.
       Current explanations: :doc:`observation_uncertainty`,
       :doc:`numerical_array_ownership` and :doc:`/usage/rhime_configuration`.
   * - External resources and basis exports
     - :download:`RHIME resource I/O
       <../../openspec/specs/rhime-resource-io/spec.md>`
     - Delivered runner-owned path resolution, selected-resource checks, BC
       loading and optional basis exports, preserving exact destinations.
       Delivery history: :download:`resource-I/O archive
       <../../openspec/changes/archive/2026-10-13-move-rhime-resource-io-to-runners/tasks.md>`.
       Manifest references and checkpoint completion stay with the handoff specs.
   * - Retained-state scientific reconstruction
     - :download:`Retained-state reconstruction
       <../../openspec/specs/retained-state-reconstruction/spec.md>`
     - Delivered scientific capability; implemented routes and limitations in
       :doc:`/usage/retained_state_reconstruction`.
   * - HBMCMC dependency direction and retirement
     - :download:`HBMCMC isolation contract
       <../../openspec/specs/hbmcmc-isolation-retirement/spec.md>`
     - Delivered 0.8 dependency boundary and legacy-only resource translation.
       Runner frontends own deprecated spellings; canonical configuration has
       no alias translation or warnings. Complete 0.9 removal remains in the
       :download:`active retirement change
       <../../openspec/changes/isolate-and-retire-hbmcmc/proposal.md>`.
       Path conversion and resource loading stay with resource I/O.
   * - Saved CO2 affine reconstruction
     - :download:`CO2 affine output reconstruction
       <../../openspec/specs/co2-output-reconstruction/spec.md>`
     - Delivered scientific capability retaining its owner-consent requirement;
       :doc:`/usage/affine_flux_map` and :doc:`/usage/co2_model_family` describe
       current support and conditional uncertainty.
   * - Scientific execution across routes
     - :download:`Recipe execution
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/recipe-execution/spec.md>`
     - Owns supported-route parity and execution behavior, consuming the
       configuration, scientific-input and handoff contracts. Family routes are
       implemented; remaining programme acceptance stays in the active change.
   * - Saved preparation and compatible reuse
     - :download:`Prepared recipe handoff
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/prepared-recipe-handoff/spec.md>`
     - Owns prepared formats, authoritative saved facts, raw-input reuse and
       incompatible baked-preparation rejection. Standard/multisector routes
       are implemented; CO2 retains coherent prior-dependent reuse constraints.
       The #876 amendment awaits CI acceptance; the capability remains active.
   * - Saved inference and fitted-operand replay
     - :download:`Sampled recipe handoff
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/sampled-recipe-handoff/spec.md>`
     - Owns sample formats, exact fitted BC/floor persistence and graph-free
       replay without recalculation. Standard/multisector replay is implemented;
       CO2 handoffs were adopted in PRs 886 and 900 with later fixed-OU work.
       The #876 amendment awaits CI acceptance; broader acceptance remains open.
       Current routes: :doc:`/usage/staged_workflow` and the family pages.

The older :download:`workflow configuration proposal
<../../openspec/changes/archive/2026-10-11-unify-recipe-workflow-configuration/proposal.md>`
is archived as superseded, without promoting its requirements. Delivered
configuration and reconstruction changes are also archived after reconciliation
into their current specs. The replacement in PR 802 remains active.
Neither proposal makes a common checkpoint interface a prerequisite for all
recipes. Do not silently merge their pending behavior into a delivered baseline.
PR 867 separated the configuration and resource-I/O amendments into
separate changes. Resource I/O and grouped configuration are promoted and
archived. The active handoff programme consumes those owners without duplicate
deltas. The durable resource contract preserves PR 840's exact destinations.

Planning records
-----------------

Plans preserve dated decisions, proposals, research, and validation evidence.
They do not establish current behavioral requirements or supported features.
The `repository-rendered planning-record inventory
<https://github.com/openghg/openghg_inversions/blob/27faed88c64517456dc5f4d176dc9232bef4c2d3/docs/plans/README.md>`_
records their status, maintained documentation, and owning issues as of
12 October 2026.
