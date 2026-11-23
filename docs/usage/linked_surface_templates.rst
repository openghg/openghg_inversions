Linked native surface templates
===============================

``prepare_linked_surface_templates`` builds a linked CO2/O2 preparation directly
from native surface responses, without a coherent reduction. Use the maintained
acquisition and ``prepare_surface_site`` operations independently for each
channel, then call ``prepare_linked_surface_templates`` with the resulting two
``MergedData`` objects. Supply an explicit positive retained prior, a complete
channel/source-to-state mapping, a labelled dense joint aggregation covariance
and provenance.

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
the covariance validation boundary. It records acquisition windows, sites,
inlets, selected source identities and the template assumption. Explicitly
declared boundary temporal-policy attributes are retained per channel/site;
absent attributes are not inferred. Caller provenance should identify
additional overlays and catalogue lineage explicitly.

The result is an ordinary ``Co2O2PreparedInputs``: save it with
``co2_o2_handoffs.save_preparation`` and run the configured linked stages as
usual. To hold outer states at their prior mean, label the retained prior's
states with ``basis_group`` and add the top-level ``[outer_flux]`` table
described in :doc:`atmospheric_baseline` and :doc:`configured_co2_priors`.

Offline Python example
----------------------

``examples/atmospheric_baseline/linked_workflow.py`` constructs two native
channels through ``prepare_surface_site``, with unequal time support and known
positive flux truth. It supplies an explicit source/state mapping and zero
unresolved covariance to ``prepare_linked_surface_templates``. The zero
covariance and noiseless truth observations are declared synthetic assumptions.

The prior labels GPP, TER and both ocean states as ``outer``. Each channel uses
a level-only additive boundary (``anomaly_scale = 0``) with a mean-shift prior,
and the ``[outer_flux]`` table holds the outer ocean states at their prior mean
while outer GPP and TER stay sampled. Concentration targets are the configured model's
prediction at the declared flux truth, made with ``pymc.do``. The controls use
either zero atmospheric levels or physical injections of +0.35 ppm CO2 and
-0.45 ppm O2. With ``--held outer`` every outer state is held, and the GPP and
TER prior means are set to their known truth; the default ``--held ocean``
infers outer GPP and TER from the declared prior.

Run sampling on a compute node, for example::

    PYTENSOR_FLAGS=floatX=float64 python examples/atmospheric_baseline/linked_workflow.py \
        --output-dir inversion_outputs/linked-controls \
        --held ocean --draws 250 --tune 250

``--likelihood fixed_ou`` uses fixed site amplitudes. Each control saves the
preparation, samples from the configuration, reopens the checkpoints for
products and writes truth, posterior means and intervals, divergences, R-hat
and effective sample size to ``recovery.json``. This tiny example illustrates
execution and recovery; it defines no campaign acceptance or convergence
criterion.
