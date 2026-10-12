Development guidance and behavioral specifications
=====================================================

Use :doc:`rhime_model_development` and :doc:`architecture_principles` to decide
how to keep scientific code readable and responsibilities local. Use scoped
capability specifications for observable behavior that a caller or artifact
consumer can rely on. Usage documentation explains how to use the implemented
interfaces. These sources serve different reader needs.

An accepted design can precede implementation. A proposed requirement does not
establish supported behavior, and a checked planning task does not replace
source and validation evidence. Check the review status, scope, and delivery
record of an active change before using it as an implementation baseline.

Current contracts and proposals
--------------------------------

The following map was reviewed against ``devel`` revision ``42a32f62`` on
12 October 2026. It records the sources relevant to this guide cleanup; it is
not a complete package API specification.

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

This change adds two narrow main capability baselines and leaves the existing
configuration and scientific changes in their reviewed locations. It does not
sync or rewrite those deltas. Broader OpenSpec lifecycle, discoverability, and
setup decisions remain tracked in `issue 707
<https://github.com/openghg/openghg_inversions/issues/707>`_.

Disposition of the former guide material
----------------------------------------

The review distinguished implementation facts and scientific assumptions from
general development advice. Imperative wording alone did not justify making a
new behavioral requirement.

.. list-table:: Guide review at the same source revision
   :header-rows: 1
   :widths: 26 38 36

   * - Material
     - Disposition
     - Reason and maintained owner
   * - Procedural runners, scientific order, ordinary callables, explicit
       resolved inputs, small duplication, locality
     - Retain as development principles.
     - Explain how scientists read and modify recipes; they are not a universal
       public calling protocol.
   * - Public selector, concrete names, established aliases
     - Establish the model-selection baseline; link it from the guide.
     - Observable supported dispatch and compatibility behavior, verified in
       ``tests/test_public_run_api.py``.
   * - Coordinate registry, custom-likelihood inputs, explicit options and
       callable provenance
     - Establish the model-composition baseline; keep examples in usage docs.
     - Real consumer guarantees with focused source/test evidence. The common
       seam does not imply identical equations or covariance support.
   * - Constructor fields, INI translation, resolution order, preparation policy
     - Link the existing configuration change and usage reference.
     - Already owned requirements. Remove the guide's separate configuration
       layout mandate; PR 802 proposes a later regrouping.
   * - Prepared-array and private staged-module inventories
     - Remove duplicated layout instructions; link current prepared/staged usage.
     - An implementation inventory is not a stable schema or a required module
       arrangement for every recipe.
   * - Likelihood, covariance, sampler and prior discussion
     - Retain scientific design reasoning; link concrete equations and limits.
     - Scientific targets, cross-channel units and sampler coupling need
       recipe-specific review. No universal covariance or sampler contract added.
   * - Scientific roles and reconstruction
     - Retain the rationale; link existing scientific specifications and usage.
     - Quantity meaning and conditional uncertainty already have scientific
       owners. Finalized artifacts are unchanged.
   * - Borrowing, labels, sparse payloads and numerical boundaries
     - Retain principles; extract enduring explanation from the dated plan into
       :doc:`numerical_array_ownership`.
     - Execution and storage are distinct. A historical Zarr-cache experiment
       is not an adopted API or architecture requirement.
   * - Future handoffs, family landing and acceptance checklists
     - Link reviewed proposals and owning issues; inventory the plans separately.
     - Proposed support and incomplete downstream acceptance remain explicit.

Planning records
-----------------

The :download:`28-record planning inventory <../plans/README.md>` records
proposed destinations, maintained documentation, and existing issue owners.
Plans retain dated decisions, research, and validation evidence; they do not
define current engineering policy, feature support, priorities, or release
gates. This first slice corrects the strongest authority conflicts and preserves
the numerical explanation without moving the other records.

Remaining migrations are tracked in `issue 853
<https://github.com/openghg/openghg_inversions/issues/853>`_ and can land in
small coherent groups. Review scientific assumptions before extracting a
contract, and transfer missing task details to existing issues before retiring
a checklist. Preserve dated evidence and its linked assets when it remains
useful; an archived proposal must not become a second current specification.
