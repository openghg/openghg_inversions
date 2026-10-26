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
requirements. Specifications describe the accepted target behavior, including
changes not yet implemented. As in #961, an agreed retirement requirement does
not establish that removal has landed. Active change specs need not match the
transitional implementation; ``design.md`` explains how to reach their target.
Delivery status and review history belong in the change records and the map
below, not in durable specs.

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
obsolete compatibility schedules without deciding a release number. On the
current #719 branch, namespace/adapters and the approved finite names are removed;
review and archival remain pending. The 26 October reconciliation places positive
canonical construction with scientific inputs; historical deletion maps remain
in the migration guide and evidence ledger. See :doc:`/usage/legacy_and_migration` for the affected promises,
retained capabilities, evidence gate and remaining owners.

The #876 amendment separates configuration choices, numerical input behavior
and saved handoff guarantees below. The restacked implementation and its scoped
review and validation are complete. The :download:`alignment change tasks
<../../openspec/changes/move-alignments-to-model-boundary/tasks.md>` record the
26 October full-suite diagnostics and successful bounded correction retry,
including exact tested revisions and limits; the retry is not a fresh full-suite
pass. PR 964 remains draft for maintainer review. The broader handoff change
remains active with its separate programme acceptance still open.

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
       the finite :ref:`CLI command retirement <rhime-cli-command-retirement>`
       and :ref:`five Python alias removals <rhime-python-alias-retirement>`
       are implemented by #719. Revision-specific acceptance belongs to its
       task/PR records. Canonical
       behavior and scientific artifact contents remain protected; supported
       artifact formats/versions are specified independently. Broader Python
       naming remains with
       #663/#945 and a #776 revival. The :download:`CO2 selector/runner amendment
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
       #961 specifies canonical-only options and ordinary unsupported-option
       rejection of every explicit ``use_tracer``. #719 implements runtime retirement
       with affected migration evidence; canonical scientific contracts remain.
       The active BC selector amendment adds explicit ``bc_basis_algorithm="NESW"``
       beside exact ``bc_basis_paths``; revision-specific acceptance and promotion
       are tracked in the retirement change.
       Configuration serialization/INI writing remains with `issue 814
       <https://github.com/openghg/openghg_inversions/issues/814>`_.
   * - Scientific preparation and model-input operands
     - :download:`RHIME scientific inputs
       <../../openspec/specs/rhime-scientific-inputs/spec.md>`
     - Owns canonical scientific construction, values/units, labelled metadata,
       coordinates, provenance and borrowed/lazy arrays, together with observation
       selection/error preparation, retained alignment, raw spatial BC inputs,
       model-time BC periods and consumed minimum-error operands. The #876
       restack is implemented; review and delivery evidence stays in the
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
     - PR 869 delivered isolation; the current #719 branch removes the transitional
       namespace and translation. The positive contract requires normal execution
       through canonical interfaces without migration tooling. Affected evidence preservation and regression migration
       gate runtime removal; finished converters/launchers are not prerequisites.
       Implementation and fresh acceptance are tracked by #719 and the
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
       The #876 restack is implemented; the capability remains active.
   * - Saved inference and fitted-operand replay
     - :download:`Sampled recipe handoff
       <../../openspec/changes/replace-recipe-workflow-with-reusable-handoffs/specs/sampled-recipe-handoff/spec.md>`
     - Owns sample formats, exact fitted BC/floor persistence and graph-free
       replay without recalculation. Standard/multisector replay is implemented;
       CO2 handoffs were adopted in PRs 886 and 900 with later fixed-OU work.
       The #876 restack is implemented; broader programme acceptance remains open.
       Current routes: :doc:`/usage/staged_workflow` and the family pages.

The older :download:`workflow configuration proposal
<../../openspec/changes/archive/2026-10-11-unify-recipe-workflow-configuration/proposal.md>`
is archived as superseded, without promoting its requirements. Delivered
configuration and reconstruction changes are also archived after reconciliation
into their current specs. The replacement in PR 802 remains active.
Neither proposal makes a common checkpoint interface a prerequisite for all
recipes. Do not silently merge their pending behavior into a delivered baseline.
PR 867 separated the configuration and resource-I/O amendments into
separate changes. Resource I/O is now promoted and archived; grouped
configuration is promoted and archived. PR 802 consumes those owners without
duplicate deltas. The durable resource contract preserves PR 840's exact destinations.

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

The scientific-input capability combines canonical construction and preparation
with the agreed selection/imputation/nested amendments implemented in #876.
#876 supersedes preservation of pre-filter imputation and tests the retained
population. #954 removes aggregation filters without a replacement statistic;
unsupported active filter names use ordinary registry validation. Global scalar
and list requests validate even without observation sites; absent-site mapping
entries remain deferred. Nested population/alignment, common-site
selection and ``averaging_error`` configuration are separate claims and scenarios.
Historical witnesses remain evidence rather than required modern expectations.
Provider acquisition averaging precedes inversion-side error combination; the
retained-population amendment does not reverse that order. Supplied ``mf_error``
is preserved. Aggregation is a separate scientific operation with its own
uncertainty propagation, not an observation-selection filter.

#719 supplies canonical interfaces and removes the special ``use_tracer``
compatibility handling. #971 supplies aggregation-filter retirement and generic
registry validation. This #964 restack builds on both prerequisites and implements
selection-first imputation, retained-row floors and model-time BC alignment.
Scoped implementation, review and validation are complete, with revision-specific
evidence and limits in the alignment change tasks linked above. None of #964's
scientific delivery, #971's implementation or M2 tooling is a prerequisite for #719.

Planning records
-----------------

Plans preserve dated decisions, proposals, research, and validation evidence.
They do not establish current behavioral requirements or supported features.
The `repository-rendered planning-record inventory
<https://github.com/openghg/openghg_inversions/blob/27faed88c64517456dc5f4d176dc9232bef4c2d3/docs/plans/README.md>`_
records their status, maintained documentation, and owning issues as of
12 October 2026.
