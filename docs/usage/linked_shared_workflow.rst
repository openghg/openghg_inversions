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

Supply the native effective sensitivities, affine term and complete joint covariance
through ``prepare_co2_o2_inputs``. Centered execution requires
``PYTENSOR_FLAGS=floatX=float64``. Native observations require measurement errors
and centered boundary operands ``H_bc_anomaly``, ``H_bc_mass`` and an explicit
stored scalar reference. CO2 and O2 may retain different rows and sites.

Alternatively, use the maintained acquisition and ``prepare_surface_site``
operations independently for each channel, then call
``prepare_linked_surface_templates`` with the resulting two ``MergedData``
objects. Supply an explicit positive retained prior, a complete channel/source-
to-state mapping, a labelled dense joint aggregation covariance and provenance.
This direct-template model has a declared zero affine intercept. It sums sources
mapped to the same coefficient and uses the already signed O2 responses once.
It does not perform a coherent reduction or infer source ratios. This Python
adapter uses four native NESW boundary curtains and has no boundary-basis
argument for twelve-region projection. Generic prepared-boundary models retain
their supported spatial bases.

Rows follow each channel's ``SiteOptions`` order, then each site's native time
order. The gathered covariance must match these exact CO2-then-O2 labels, with
``ppm`` unit coordinates on both axes. Native measurement errors, centered
endpoint operands and one agreed scalar boundary reference per channel are
required. Preparation converts concentration fields to physical ppm and keeps
borrowed payloads lazy except for a joint small scalar-reference boundary and
the covariance validation boundary. It records
acquisition windows, sites, inlets, selected source identities and the template
assumption. Explicitly declared boundary temporal-policy attributes are retained
per channel/site; absent attributes are not inferred. Caller provenance should
identify additional overlays and catalogue lineage explicitly.

Call ``prepare_linked_shared_inputs`` with the linked preparation, separate
``SharedBaselineChoice`` values in the ``choices`` mapping for CO2/O2, and
original-state boolean ``outer`` and ``sampled_outer`` selectors. Each choice
contains the current additive or anomaly-scaling boundary choice, an outer
mode, and a positive scalar ``delta_sd`` in native observation units. A
``multiplicative_shrunk`` request also requires a resolved ``BiospherePrior``.
The supported modes are ``fixed``, ``multiplicative`` and
``multiplicative_shrunk``. Selected outer ocean states stay fixed; inner ocean
states stay sampled. Linked additive net-biosphere corrections are rejected.

Prepare on the complete retained support before selecting rows. The returned
``Co2O2PreparedInputs.shared_baseline`` freezes the native references, operators,
centering weights, complete-support means, selectors and calibration identity.
Ordinary and shared linked preparations both write and accept numerical schema
version 6 only. Optional shared nodes must form a complete CO2/O2 pair. Prepare
earlier artifacts again. This optional workflow owns a current snapshot of the
bound ``SharedBaseline(choice, data)`` record; it saves factual operators and
explicit data-free choice fields. Loading reconstructs the record directly;
subsetting uses ``dataclasses.replace(bound, data=bound.data.isel(...))``.
Neither operation binds, rebuilds calendars, recomputes means or recalibrates.
The saved reference shift is row dependent: ``reference * H_bc_mass.sum("bc_region")``.

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
the latter uses the existing matched cached sampler. ``sampling.target_accept``
(default 0.95) tunes both its ordered amplitude and state NUTS steps. OU sampling requires
PyMC. Ordinary and cached sampling reuse the maintained compact parameter
recording and deterministic reconstruction operation when compact recording is
requested. The canonical graph coefficients are ``co2_bc_mean_shift`` and
``o2_bc_mean_shift``; ``baseline_delta`` is a reporting field only.

Run ``rhime prior-predictive``, ``rhime sample`` and ``rhime postprocess`` with
``--model co2_o2`` and the same recipe request. Preparation/sample manifests pin
numerical content as well as preparation generation. Replacing the preparation
or posterior invalidates the checkpoint. Current version-2 manifests require
recipe and content identities; older or incomplete records must be prepared and
sampled again.

Products and interpretation
---------------------------

Basic products retain native channel rows, physical source contributions,
unattributed atmospheric level deltas, raw/centered BC and outer corrections, and each
removed mean. The sampled selected ``flux_scaling`` remains unchanged. A
full physical state is reconstructed lazily for requested products, filling
removed outer states at their original arithmetic prior means. It uses the
original prior state axis and labels. Saved traces retain only selected sampled
states and the frozen original prior, without duplicate per-draw full states.

Atmospheric accounting obeys::

    modelled_concentration = physical_atmospheric_total - removed_mean_correction

Physical source contributions therefore have a different interpretation from
centered atmospheric adjustments. Saved products preserve both. Reopening
samples and calculating products constructs no PyMC graph and fits no new
calibration. Shared-baseline PARIS export is explicitly rejected pending a
reviewed mapping for the removed-mean accounting.


Offline Python example
----------------------

``examples/shared_baseline/linked_workflow.py`` constructs two native channels
through ``prepare_surface_site``, with unequal time support and known positive
flux truth. It supplies an explicit source/state mapping and zero unresolved
covariance to ``prepare_linked_surface_templates``. The zero covariance and
noiseless truth observations are declared synthetic assumptions.

The example then freezes shared preparation, samples from a configuration and
reopens checkpoints for graph-free products. After freezing native support,
concentration targets are generated by the shared atmospheric reconstruction
at the declared flux truth. The controls use either zero baseline shifts or
physical injections of +0.35 ppm CO2 and -0.45 ppm O2. For ``fixed`` outer mode,
fixed prior means equal the known outer truth; adjustable modes infer departures
from their declared prior.

Run sampling on a compute node, for example::

    PYTENSOR_FLAGS=floatX=float64 python examples/shared_baseline/linked_workflow.py \
        --output-dir inversion_outputs/linked-controls \
        --mode multiplicative --draws 250 --tune 250

``--likelihood fixed_ou`` uses fixed site amplitudes; ``--likelihood inferred_ou``
uses the matched cache. Each control writes truth, posterior means and intervals,
divergences, R-hat and effective sample size to ``recovery.json``. This tiny
example illustrates execution and recovery; it defines no campaign acceptance
or convergence criterion.

The example uses ``SharedBaselineChoice(None, 1.0, mode)`` for each channel and
passes those choices explicitly to the current snapshot workflow. Saved inputs
use format6. The builder returns ``built, cached, materialized_prepared``;
annotation uses that returned preparation. Pure reconstruction receives the
channel mapping of ``bound.data``, retaining physical source effects separately
from atmospheric removed means.
