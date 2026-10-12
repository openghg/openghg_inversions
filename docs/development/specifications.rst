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

Capability baselines live under ``openspec/specs/``. Some reviewed contracts
remain under ``openspec/changes/``; directory location alone does not establish
approval or delivery. The map identifies both kinds of source.

Current contracts and proposals
--------------------------------

The following map was reviewed against ``devel`` revision ``42a32f62`` on
12 October 2026. It covers RHIME selection, composition, configuration, and
reconstruction boundaries; it is not a complete package API specification.

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
       :doc:`/usage/cli`.
   * - Registered coordinates and custom likelihood composition
     - :download:`RHIME model composition
       <../../openspec/specs/rhime-model-composition/spec.md>`
     - Baseline of the named delivered interfaces;
       :doc:`/usage/customising_rhime`. Covariance support remains recipe-specific.
   * - Requested configuration and preparation ownership
     - :download:`Resolve RHIME configuration
       <../../openspec/changes/resolve-rhime-configuration/specs/rhime-configuration/spec.md>`
     - Delivered configuration slice; :doc:`/usage/rhime_configuration`.
       Export/writer work remains with `issue 814
       <https://github.com/openghg/openghg_inversions/issues/814>`_.
   * - Retained-state scientific reconstruction
     - :download:`Retained-state reconstruction
       <../../openspec/changes/unify-retained-state-reconstruction/specs/retained-state-reconstruction/spec.md>`
     - Final reviewed scientific change; implemented routes and limitations in
       :doc:`/usage/retained_state_reconstruction`.
   * - Saved CO2 affine reconstruction
     - :download:`CO2 affine output reconstruction
       <../../openspec/changes/persist-co2-affine-output-reconstruction/specs/co2-output-reconstruction/spec.md>`
     - Final reviewed scientific change with an explicit owner-consent gate;
       :doc:`/usage/affine_flux_map` and :doc:`/usage/co2_model_family` describe
       current support and conditional uncertainty.
   * - Reusable recipe execution and prepared/sample handoffs
     - Current replacement proposal in `PR 802
       <https://github.com/openghg/openghg_inversions/pull/802>`_
     - Proposed interfaces and configuration regrouping; implementation remains
       outstanding at this review date. Current routes are documented in
       :doc:`/usage/staged_workflow` and the family pages.

The older :download:`workflow configuration proposal
<../../openspec/changes/unify-recipe-workflow-configuration/proposal.md>` in this
checkout is a planning record whose replacement is under review in PR 802.
Neither proposal makes a common checkpoint interface a prerequisite for all
recipes. Do not silently merge their pending behavior into a delivered baseline.

Planning records
-----------------

Plans preserve dated decisions, proposals, research, and validation evidence.
They do not establish current behavioral requirements or supported features.
The `repository-rendered planning-record inventory
<https://github.com/openghg/openghg_inversions/blob/27faed88c64517456dc5f4d176dc9232bef4c2d3/docs/plans/README.md>`_
records their status, maintained documentation, and owning issues as of
12 October 2026.
