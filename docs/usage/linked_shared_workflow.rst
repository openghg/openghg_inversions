Configured linked shared-baseline workflow
=========================================

The ``co2_o2_shared_baseline`` recipe runs the reviewed linked atmospheric
baseline model through the existing linked preparation, sample and postprocess
stages. The default ``co2_o2`` recipe and the separate CO2 recipes retain their
existing behavior. Both linked channels use one common concentration unit and
one shared positive flux state. The supplied O2 sensitivity already contains
its signed response; the model applies no additional oxidative ratio.

Preparing native inputs
-----------------------

Use the maintained acquisition and ``prepare_surface_site`` operations for each
channel. ``prepare_linked_surface_templates`` accepts the resulting two
``MergedData`` objects, an explicit positive retained prior, a complete
channel/source-to-state mapping, a labelled joint aggregation covariance and
provenance. It implements a declared direct-template model with zero affine
intercept. It does not perform a coherent reduction or infer source ratios.
For a coherent reduction, supply its native effective sensitivities, affine
term and complete joint covariance through ``prepare_co2_o2_inputs`` instead.

Centered execution requires explicitly configured ``PYTENSOR_FLAGS=floatX=float64``.
Native observations require measurement errors and centered boundary operands
``H_bc_anomaly``, ``H_bc_mass`` and an explicit stored scalar reference. CO2 and
O2 may retain different rows and sites. Missing endpoint data remains missing;
preparation must select usable native rows explicitly. The template adapter
records the acquisition windows, sites, inlets, selected sources and its model
assumption. Declared native boundary temporal-policy attributes are retained
per channel/site beside selected input identities and carried through saved
preparations, trace metadata and product provenance. Missing attributes are not
inferred. Caller provenance should identify additional overlays and catalog
lineage explicitly.

Call ``prepare_linked_shared_inputs`` with the linked preparation, separate
``BoundaryCorrectionConfig`` values for CO2/O2, original-state boolean ``outer``
and ``sampled_outer`` selectors, an outer mode, and a positive scalar
``delta_sd`` with physical concentration units for each tracer. A
``multiplicative_shrunk`` request also requires a resolved ``BiospherePrior``.
The supported modes are ``fixed``, ``multiplicative`` and
``multiplicative_shrunk``. Selected outer ocean states stay fixed; inner ocean
states stay sampled. Linked additive net-biosphere corrections are rejected.

Prepare on the complete retained support before selecting rows. The returned
``Co2O2PreparedInputs.shared_baseline`` freezes the native references, operators,
centering weights, complete-support means, selectors and calibration identity.
Saving uses numerical schema version 4; linked preparations without shared
baselines retain their existing schemas and remain readable.

Configured stages
-----------------

Save with ``co2_o2_handoffs.save_preparation`` or import that numerical artifact
with ``rhime prepare --model co2_o2``. Use this request for subsequent stages::

    recipe = "co2_o2_shared_baseline"

    [channels.co2]
    units = "ppm"

    [channels.o2]
    units = "ppm"

    [likelihood]
    kind = "fixed_independent"

    [sampling]
    nuts_sampler = "pymc"
    draws = 250
    tune = 250
    chains = 2
    random_seed = 246

    [outputs]
    output_format = "basic"
    output_path = "linked-products"
    save_concentrations = true

The configuration selects already frozen science. Boundary, offset and
biosphere overrides are rejected for this recipe. Omitted measurement-error
choices use the saved vector. A fixed OU likelihood accepts ``tau_hours`` and
either fixed species/site amplitudes or a positive ``site_amplitude_prior``;
the latter uses the existing matched cached sampler. OU sampling requires
PyMC. Ordinary and cached sampling reuse the maintained compact parameter
recording and deterministic reconstruction operation.

Run ``rhime prior-predictive``, ``rhime sample`` and ``rhime postprocess`` with
``--model co2_o2`` and the same recipe request. Preparation/sample manifests pin
numerical content as well as preparation generation. Replacing the preparation
or posterior invalidates a new checkpoint. Earlier manifests without content
digests retain their existing compatibility behavior.

Products and interpretation
---------------------------

Basic products retain native channel rows, physical source contributions,
physical baseline deltas, raw/centered BC and outer corrections, and each
removed mean. The sampled selected ``flux_scaling`` remains unchanged. A
separate full physical state fills removed outer states at their original
arithmetic prior means. Its dimension is ``physical_state`` to distinguish it
from the selected sampled state axis.

Atmospheric accounting obeys::

    modelled_concentration = physical_atmospheric_total - removed_mean_correction

Physical source contributions therefore have a different interpretation from
centered atmospheric adjustments. Saved products preserve both. Reopening
samples and calculating products constructs no PyMC graph and fits no new
calibration. Shared-baseline PARIS export is explicitly rejected pending a
reviewed mapping for the removed-mean accounting.

Offline controls
----------------

``examples/shared_baseline/linked_workflow.py`` constructs two tiny native
channels through ``prepare_surface_site``, with unequal time support and known
positive flux truth. It then freezes preparation, samples from a configuration,
and reopens the checkpoints for saved graph-free products. After freezing the
native transport/support, targets are generated by the reviewed pure shared
reconstruction at the declared flux truth. Raw physical source totals do not
define the shared atmospheric delta because outer column means are removed.
The example runs matched zero
and injected physical baseline controls, with injections +0.35 ppm CO2 and
-0.45 ppm O2. It writes truth, posterior means and 95% intervals, divergence counts, R-hat
and effective sample size to ``recovery.json``.
For the fixed mode, the declared fixed outer means equal the known outer truth;
the adjustable modes infer those departures from their declared prior.
The tiny example illustrates execution and recovery; it defines no campaign
acceptance or convergence criterion.

Run sampling on a compute node, for example::

    PYTENSOR_FLAGS=floatX=float64 python examples/shared_baseline/linked_workflow.py \
        --output-dir inversion_outputs/linked-controls \
        --mode multiplicative --draws 250 --tune 250

``--likelihood fixed_ou`` uses fixed site amplitudes; ``--likelihood inferred_ou``
exercises the matched cache. ``scripts/slurm_linked_controls.sh`` runs one array
covering the three outer modes plus both OU paths, each with zero/injected controls
and 1000 tuning plus 1000 retained draws per chain.

The example declares its template model and unresolved covariance explicitly.
It does not reproduce an archived delivery's temporal policy or certify a
real-data campaign. References and anomalies from different numerical
preparations must not be mixed.
