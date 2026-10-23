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

Durable contracts under ``openspec/specs/`` describe accepted behavioral
requirements. A retention promise can be amended before its permitted removal
is implemented, as in #961; authorization does not establish delivery. Proposed
implementation changes remain under ``openspec/changes/`` until delivered and
validated, even when their design is approved. Delivery status and review
history belong in the change records and the map below, not in durable specs.

Current contracts and proposals
--------------------------------

The following map was reviewed against ``devel`` revision ``7269f50f`` on
12 October 2026. It covers RHIME selection, composition, configuration, and
reconstruction boundaries; it is not a complete package API specification.
The configuration entry was amended on 13 October 2026 during review of PR 867.
The resource-I/O and HBMCMC entries were updated the same day for the accepted
0.8 implementation and durable-spec promotion in PR 869.
The compatibility entries were reconciled on 23 October 2026 for the
`#961 breaking-development amendment
<https://github.com/openghg/openghg_inversions/issues/961>`_. It supersedes
obsolete compatibility schedules without implementing removals or deciding a
release number. See :doc:`/usage/legacy_and_migration` for the affected promises,
retained capabilities, evidence gate and remaining owners.

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
       :doc:`/usage/cli`. #961 removes the blanket retention timetable;
       unresolved public aliases remain with #945. The :download:`CO2 selector/runner amendment
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/rhime-model-selection/spec.md>`
       includes the merged `PR 886 <https://github.com/openghg/openghg_inversions/pull/886>`_
       runner and the OPE-214 stack for model-driven fixed-OU inference and shared handoffs.
   * - Registered coordinates and custom likelihood composition
     - :download:`RHIME model composition
       <../../openspec/specs/rhime-model-composition/spec.md>`
     - Baseline of the named delivered interfaces;
       :doc:`/usage/customising_rhime`. Covariance support remains recipe-specific.
   * - Requested configuration and preparation ownership
     - :download:`Resolve RHIME configuration
       <../../openspec/specs/rhime-configuration/spec.md>`
     - Delivered grouped configuration and independent phase resolution,
       implemented in `PR 866 <https://github.com/openghg/openghg_inversions/pull/866>`_.
       Current usage: :doc:`/usage/rhime_configuration`.
       #961 permits bounded retirement of temporary runtime translation under
       the affected-evidence gate, while preserving canonical configuration
       and scientific contracts.
       Configuration serialization/INI writing remains with `issue 814
       <https://github.com/openghg/openghg_inversions/issues/814>`_.
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
     - Delivered dependency boundary and transitional legacy-only resource translation.
       Runner frontends own deprecated spellings; canonical configuration has
       no alias translation or warnings. #961 authorizes development removals
       after affected evidence preservation and regression migration; finished
       converters/launchers are not prerequisites. Implementation remains with #719 and the
       :download:`active retirement change
       <../../openspec/changes/isolate-and-retire-hbmcmc/proposal.md>`.
       Its `delivery checklist
       <https://github.com/openghg/openghg_inversions/blob/devel/openspec/changes/isolate-and-retire-hbmcmc/tasks.md#migration-evidence-checklist-and-delivery-order>`_
       records the pinned comparison, per-PR work, owners and delivery order;
       these implementation instructions are separate from the durable evidence guarantees.
       Path conversion and resource loading stay with resource I/O.
   * - Saved CO2 affine reconstruction
     - :download:`CO2 affine output reconstruction
       <../../openspec/specs/co2-output-reconstruction/spec.md>`
     - Delivered scientific capability retaining its owner-consent requirement;
       :doc:`/usage/affine_flux_map` and :doc:`/usage/co2_model_family` describe
       current support and conditional uncertainty.
   * - Reusable recipe execution and prepared/sample handoffs
     - :download:`Reusable handoff proposal
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/proposal.md>`
       in `PR 802 <https://github.com/openghg/openghg_inversions/pull/802>`_
     - Standard prepared/sample interfaces are implemented, including graph-free
       replay and the two-model/two-output parity example. CO2 preparation and outputs
       were adopted in PRs 886 and 900; the OPE-214 stack completes inferred fixed-OU
       handoffs. Remaining programme work stays in the active change. Current routes are documented in
       :doc:`/usage/staged_workflow` and the family pages.

The older :download:`workflow configuration proposal
<../../openspec/changes/archive/2026-10-11-unify-recipe-workflow-configuration/proposal.md>`
is archived as superseded, without promoting its requirements. Delivered
configuration and reconstruction changes are also archived after reconciliation
into their current specs. The replacement in PR 802 remains active.
Neither proposal makes a common checkpoint interface a prerequisite for all
recipes. Do not silently merge their pending behavior into a delivered baseline.
PR 867 separated the configuration and resource-I/O amendments into
separate changes. Resource I/O is now promoted and archived; grouped
configuration remains active. PR 802 consumes those owners without duplicate
deltas. The durable resource contract preserves PR 840's exact destinations.

Related scientific amendment and delivery order
-----------------------------------------------

For `#876 / PR 964
<https://github.com/openghg/openghg_inversions/issues/876>`_ and its
`#954 / PR 971 prerequisite
<https://github.com/openghg/openghg_inversions/pull/971>`_, the maintainer agreed
the bounded contract placement on 23 October 2026:

* ``rhime-configuration`` owns accepted options, defaults, anchors and compatibility checks.
* A focused ``rhime-scientific-inputs`` capability owns numerical preparation,
  selection/imputation, nested retained populations, BC alignment and retained-row floors.
* The existing prepared/sample handoff contracts own saved-operand reuse and exact replay.
  Concrete helper calls and wrapper instructions belong in development guidance/design;
  the PyMC-composition contract keeps its scope.

Those amendments will be delivered with their owning PRs; their placement is agreed,
not awaiting another API decision. #876 supersedes preservation of pre-filter
imputation and tests the retained population. #954 owns explicit rejection of
retired aggregation filters without a replacement statistic; its scenario is
``Reject a retired aggregation filter``. Nested population/alignment, common-site
selection and ``averaging_error`` configuration are separate claims and scenarios.
Historical witnesses remain evidence rather than required modern expectations.

Complete #719, including the separately identified special ``use_tracer``
compatibility retirement, before #964. #971 remains a separate prerequisite for
#964's selection-first imputation change. None of #964's scientific delivery,
#971's implementation or M2 tooling is a prerequisite for #719. The current
branch still has the transitional adapters and acquired-population error policy;
this contract amendment implements neither removal nor the new science.

Planning records
-----------------

Plans preserve dated decisions, proposals, research, and validation evidence.
They do not establish current behavioral requirements or supported features.
The `repository-rendered planning-record inventory
<https://github.com/openghg/openghg_inversions/blob/27faed88c64517456dc5f4d176dc9232bef4c2d3/docs/plans/README.md>`_
records their status, maintained documentation, and owning issues as of
12 October 2026.
