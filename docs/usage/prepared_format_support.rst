Prepared artifact format support
================================

A format version describes a storage contract; a representation describes the
scientific contents. Generic preparations use the unchanged identifier
``openghg_inversions.rhime_prepared_inputs`` and current version **3**.
The root ``representation`` attribute is required and is either ``ordinary``
or ``centered``. Ordinary is the preparation default. Centered inputs require
an explicit preparation choice and the complete ``H_bc_anomaly``,
``H_bc_mass`` and scalar ``bc_reference`` operands, with matching labels and
concentration units and dimensionless mass. The inner boundary marker remains
``boundary_representation="centered-v1"``; it is an operand convention, not an
artifact version. Model boundary periods are chosen when binding the preparation.

Supported numerical envelopes
-----------------------------

.. list-table:: Finite identifier/version map
   :header-rows: 1
   :widths: 45 12 43

   * - Identifier
     - Version
     - Supported contents
   * - ``openghg_inversions.rhime_prepared_inputs``
     - 3
     - Explicit ``ordinary`` or ``centered`` representation.
   * - ``openghg_inversions.rhime_prepared_inputs``
     - 2 (read only)
     - Known ordinary raw inputs from the reusable input lifecycle. No centered
       marker or operands, no representation field, and any ``H_bc`` must be
       marked ``raw-spatial`` without fitted boundary periods.
   * - ``openghg_inversions.co2_prepared_inputs``
     - 1 / 2
     - Version 1 supports ordinary CO₂ coherent reduction; version 2 additionally
       supports centered operands. Writers use 1 / 2 for ordinary / centered.
       The embedded
       ``rhime_inputs`` tree follows the generic format rules above.
   * - ``openghg_inversions.co2_o2_prepared_inputs``
     - 1 / 2
     - Version 1 supports linked absolute boundaries; version 2 additionally
       supports independently labelled centered channel operands. Writers use
       1 / 2 for absolute / centered.

The CO₂ and linked envelopes have independent histories. Their versions do not
follow the generic version number. Coherent-reduction covariance remains part of
the CO₂ scientific contract.

Supported stage manifests
-------------------------

.. list-table:: Manifest identifiers and accepted versions
   :header-rows: 1
   :widths: 80 20

   * - Identifier
     - Version
   * - ``openghg_inversions.standard.prepare``
     - 2
   * - ``openghg_inversions.standard.sample``
     - 2
   * - ``openghg_inversions.multisector.prepare``
     - 2
   * - ``openghg_inversions.multisector.sample``
     - 2
   * - ``openghg_inversions.co2.prepare``
     - 1
   * - ``openghg_inversions.co2.sample``
     - 1
   * - ``openghg_inversions.co2_o2.prepare``
     - 1
   * - ``openghg_inversions.co2_o2.sample``
     - 1

A manifest's version is independent of the numerical artifact it references.
Preparation generation and final artifact bindings still control sample and
reconstruction association; editing an artifact's version metadata does not
migrate its scientific contents.

Regeneration and migration
-------------------------

Generic version 1 contains retired baked model policies and is rejected.
Historical centered version 2 conflicts with ordinary raw version 2 and is also
rejected, including incomplete centered operands. Unknown versions and missing
or mismatched representations are rejected with regeneration guidance.

Regenerate unsupported preparations from their original observations,
transport/boundary inputs, selection and uncertainty policies using current
preparation APIs. For centered O₂, retain native accurate anomaly and mass
operands; rounded absolute boundary sensitivities cannot recover them. A workflow
may instead select a separately validated migration route for a known historical
artifact. Historical campaign conversion is not required to use the supported
regeneration route. Recreate preparation/sample manifests after regeneration;
use the original matching software when historical baked artifacts must be read.
Both NetCDF and Zarr carry these same scientific format contracts.
