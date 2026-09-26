Prior predictive checking tutorial
==================================

This tutorial prepares a standard RHIME inversion, samples simulated
concentrations without sampling a posterior, and uses the result to decide
whether the configured priors and observation model are scientifically
plausible. It then samples the posterior from the same prepared data.

A prior predictive check asks what observations the complete model could
generate before its unknown parameters are conditioned on the observed
concentrations. RHIME predictions remain conditional on the prepared sites,
times, footprints, flux inventories, boundary inputs, and measurement-error
estimates. The check considers the priors and likelihood together: a prior
that looks innocuous by itself can imply implausible concentrations after
transport, boundary conditions, and model-data mismatch are combined. The
check is a scientific judgement, not a binary test or proof that the priors
are correct.

If ``pollution_events_from_obs=True``, the configured mismatch scale also uses
the observed concentrations. The resulting simulations are a data-dependent
conditional check, not a wholly pre-data prior predictive distribution.

This workflow currently supports the staged ``standard`` and ``multisector``
recipes. This tutorial uses ``standard``. The CO₂ model family does not have a
staged CLI route; in particular, do not treat PyMC's generic prior-predictive
sampling as a substitute for a predictive implementation for models whose
likelihood is represented by a ``Potential``.

Prerequisites
-------------

Install the package and populate the companion tutorial store by following
the :doc:`standard tutorial prerequisites <rhime_standard_tutorial>`. The
commands below use its one-week CH₄ configuration. A different standard RHIME
INI can be substituted once that example works.

The companion-data instructions change into the data checkout while populating
the store. Return to the ``openghg_inversions`` checkout root before continuing;
all remaining shell commands use paths relative to that directory. Then choose
paths for the configuration and artifacts:

.. code-block:: bash

   CONFIG=openghg_inversions/rhime/config/standard_tutorial.ini
   RUN=outputs/prior-check

If the command is not installed in the active environment, prefix each
invocation with ``pixi run -e dev`` from a source checkout. The :doc:`CLI guide
<cli>` describes the supported environment alternatives.

Prepare the data once
---------------------

Run the preparation stage:

.. code-block:: bash

   openghg-inversions prepare \
     --config "$CONFIG" \
     --model standard \
     --output-dir "$RUN/prepare"

This writes ``prepared-inputs.nc`` and ``prepare-manifest.json`` beneath
``$RUN/prepare``. The prepared artifact fixes the observations, sites, times,
footprints, prior flux, boundary conditions, basis, and sensitivities used by
the subsequent stages. It does not contain posterior samples.

Generate prior predictions
--------------------------

Start with 500 simulated datasets to see the distribution's broad shape and
tails:

.. code-block:: bash

   openghg-inversions prior-predictive \
     --config "$CONFIG" \
     --model standard \
     --prepared-inputs "$RUN/prepare/prepared-inputs.nc" \
     --preparation-manifest "$RUN/prepare/prepare-manifest.json" \
     --draws 500 \
     --output-dir "$RUN/prior-predictive" \
     --strict

Five hundred draws are illustrative. Increase the count or repeat the check
when rare tails or high-dimensional extreme-series behaviour matters to the
scientific question.

The command builds the configured PyMC model and calls prior-predictive
sampling, but it does not call NUTS or sample a posterior. It writes:

* ``prior-predictive.nc``, an ArviZ ``InferenceData`` artifact; and
* ``prior-predictive-readiness.json``, a machine-readable check result.

``--strict`` exits nonzero if model construction fails or sampled values are
non-finite. A ``pass`` means only that the model produced finite values. It
does **not** mean that the simulated concentrations, variability, or extremes
are scientifically plausible.

Inspect the artifact
--------------------

Load the result and inspect its groups and variable names before plotting:

.. code-block:: python

   import arviz as az
   from openghg_inversions.serialization import load_inferencedata

   prior = load_inferencedata(
       "outputs/prior-check/prior-predictive/prior-predictive.nc"
   )
   print(prior.groups())
   print("prior variables:", sorted(prior.prior.data_vars))
   print("replicated observations:", sorted(prior.prior_predictive.data_vars))

The output should include at least ``prior``, ``prior_predictive``, and
``observed_data``. A missing group means the expected prior-check artifact was
not produced; inspect the readiness JSON and command output before continuing.

The important distinction is:

``prior``
   Samples of parameters and deterministic latent quantities. In the standard
   model, ``mu`` is the flux-derived contribution to the conditional mean,
   ``mu_bc`` is the boundary contribution when boundary conditions are
   enabled, and ``offset`` is present when an offset is enabled. Their sum is
   the latent conditional-mean concentration for each parameter draw.
   ``epsilon`` is the modelled observation-error scale.

``prior_predictive``
   Replicated observations. ``y`` includes the latent expected concentration
   and a new draw from the configured observation model. It is therefore the
   quantity to compare with plausible measurements.

``observed_data``
   The actual concentrations supplied to the likelihood. Overlaying them gives
   context, but prior predictions need not closely reproduce the observed
   series.

For example, compare the distribution of replicated and observed
concentrations:

.. code-block:: python

   import matplotlib.pyplot as plt

   az.plot_ppc(
       prior,
       group="prior",
       observed=True,
       var_names=["y"],
       kind="cumulative",
       num_pp_samples=100,
       random_seed=42,
   )
   plt.show()

``observed=True`` is explicit because ArviZ hides observed data by default for
a prior predictive plot. A single flattened comparison can conceal a failure
at one site or during one period, so also inspect the labelled structure. The
prepared artifact supplies the authoritative site/time measurement index:

.. code-block:: python

   import numpy as np
   from openghg_inversions.inversion_data import RhimePreparedInputs

   prepared = RhimePreparedInputs.load(
       "outputs/prior-check/prepare/prepared-inputs.nc"
   )
   observations = prepared.inv_inputs["mf"]
   concentration_units = observations.attrs.get("units", "mole fraction")
   print("concentration units:", concentration_units)
   measurement_index = prepared.inv_inputs.indexes["nmeasure"]
   sites = measurement_index.get_level_values("site")
   times = measurement_index.get_level_values("time")
   predictions = (
       prior.prior_predictive["y"]
       .stack(sample=("chain", "draw"))
       .transpose("sample", "nmeasure")
   )

   site_names = sites.unique().tolist()
   figure, axes = plt.subplots(len(site_names), 1, squeeze=False, sharex=False)
   for axis, site in zip(axes.flat, site_names):
       positions = np.flatnonzero(sites == site)
       intervals = predictions.isel(nmeasure=positions).quantile(
           [0.05, 0.5, 0.95], dim="sample"
       )
       axis.fill_between(
           times[positions],
           intervals.sel(quantile=0.05),
           intervals.sel(quantile=0.95),
           alpha=0.25,
           label="90% prior predictive interval",
       )
       axis.plot(times[positions], intervals.sel(quantile=0.5), label="median")
       trajectories = predictions.isel(sample=[0, 1, 2], nmeasure=positions)
       axis.plot(
           times[positions],
           trajectories.transpose("nmeasure", "sample"),
           color="0.5",
           alpha=0.5,
           linewidth=0.8,
       )
       axis.scatter(times[positions], observations.isel(nmeasure=positions), s=8,
                    label="observed")
       axis.set_title(site)
       axis.set_xlabel("Time")
       axis.set_ylabel(f"Concentration ({concentration_units})")
       axis.legend()
   figure.autofmt_xdate()
   plt.show()

The example produces one panel for MHD and one for TAC. Each panel should show
the observed CH₄ series, three complete simulated series, and the pointwise
90% prior predictive interval in the units printed above. Treat the panels as
a decision point. For example, if a displayed complete series or a substantial
part of the predictive interval has negative CH₄ concentrations, or has
excursions that are incompatible with the configured flux and boundary
scenario, stop and revise the responsible prior or mismatch model. If the
simulated ranges and structures are scientifically defensible, record that
judgement and its basis before continuing. Agreement with every observed point
is not the success criterion.

What to look for
----------------

Judge simulations in the physical units and scientific context of the run.
Useful questions include:

* Do simulated concentrations have plausible magnitudes, signs, variability,
  and extremes?
* Are site-to-site differences and temporal patterns compatible with what the
  transport and boundary inputs could produce?
* Does a meaningful fraction of draws put mass on impossible or scientifically
  absurd outcomes?
* Do individual full simulated series reveal failures hidden by marginal
  intervals or density plots?
* Can an implausible feature be traced to flux scaling, boundary scaling,
  offsets, or the model-data mismatch prior?

The aim is not to make every observation fall inside an extremely wide band.
A useful weakly informative model can generate extreme-but-plausible datasets
while assigning little mass to impossible ones. Choose summaries related to
the scientific question and known failure modes; no finite collection of plots
checks every implication of a model.

Revise and repeat
-----------------

If the simulations are implausible, change the relevant prior or model option
in the configuration and repeat both ``prepare`` and ``prior-predictive`` in a
new output directory. Prior settings are part of the preparation manifest's
configuration identity, so a manifest from the previous configuration is
intentionally rejected. Record why each change is scientifically justified;
repeatedly tuning a prior only to mimic the observed series turns the exercise
into data-dependent model fitting.

Sample the posterior from the checked data
------------------------------------------

When the prior predictive behaviour is credible, pass the same prepared input
and manifest to the posterior stage:

The packaged tutorial configuration uses only 50 tuning and 50 retained draws,
two chains, and disabled convergence checks. Those settings demonstrate the
workflow but are not adequate for scientific interpretation. Increase the
sampler settings and enable convergence checks for a scientific run.
Sampler-only changes do not invalidate the prepared artifact; changes to the
scientific preparation, model, or prior settings require a new preparation.

.. code-block:: bash

   openghg-inversions sample \
     --config "$CONFIG" \
     --model standard \
     --prepared-inputs "$RUN/prepare/prepared-inputs.nc" \
     --preparation-manifest "$RUN/prepare/prepare-manifest.json" \
     --output-dir "$RUN/sample"

This rebuilds the same configured model from the authenticated prepared data,
runs the configured sampler, and writes ``posterior.nc`` and
``sample-manifest.json``. It reuses the data artifact, not the in-memory PyMC
graph from the prior-predictive process.

Next run the staged convergence check:

.. code-block:: bash

   openghg-inversions diagnose \
     --posterior "$RUN/sample/posterior.nc" \
     --sample-manifest "$RUN/sample/sample-manifest.json" \
     --output-dir "$RUN/diagnose"

Inspect the resulting diagnostics, trace behaviour, effective sample sizes,
R-hat, and divergences before interpreting the posterior; see the
:ref:`staged convergence-check contract <staged-convergence-check>`. Then
perform a posterior predictive check:

.. code-block:: python

   posterior = load_inferencedata("outputs/prior-check/sample/posterior.nc")
   az.plot_ppc(
       posterior,
       group="posterior",
       observed=True,
       var_names=["y"],
       num_pp_samples=100,
       random_seed=42,
   )
   plt.show()

Prior and posterior predictive checks answer different questions. The first
tests whether the model's implications before parameter fitting are plausible,
conditional on the prepared inputs described above; the second assesses how
well the fitted model can reproduce relevant features of the observations.
Neither replaces sampler diagnostics, sensitivity analysis, or independent
scientific validation. A prior predictive check is also not simulation-based
calibration, which repeatedly simulates and refits data to test an inference
procedure.

Further reading
---------------

* PyMC, `Prior and Posterior Predictive Checks
  <https://www.pymc.io/projects/docs/en/v5.7.2/learn/core_notebooks/posterior_predictive.html>`_.
* ArviZ, `plot_ppc API
  <https://python.arviz.org/en/v0.23.4/api/generated/arviz.plot_ppc.html>`_.
* Stan User's Guide, `Posterior and Prior Predictive Checks
  <https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html>`_.
* Gabry, J. et al. (2019), “Visualization in Bayesian workflow”,
  `doi:10.1111/rssa.12378 <https://doi.org/10.1111/rssa.12378>`_.
* Gelman, A. et al. (2020), “Bayesian Workflow”,
  `arXiv:2011.01808 <https://arxiv.org/abs/2011.01808>`_.
