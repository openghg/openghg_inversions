# Process-specific biosphere prior prototype (OPE-197)

This isolated research recipe tests errors in GPP, autotrophic respiration (Ra),
and heterotrophic respiration (Rh) templates. Atmospheric signs belong to the
supplied sensitivities; all multipliers are positive. It changes scientific
priors and is not a neutral reparameterisation or a production default.

Each state has unique `flux_state`, `source` and `group` labels. Pairing follows
source/group identity, never array position. The full supplied state is the
complete uncertain flux model: no unresolved native directions or aggregation
error are implied. Exact-zero sensitivity columns are rejected, since deleting
their prior uncertainty would invalidate full-domain flux reconstruction.

For Gaussian latent correlation C = L L^T, use
`x_i = exp(-s^2/2 + s (L z)_i)`, with z independent standard normal. This keeps
conditional arithmetic means one. Three recipes are available:

- Independent: C = I and fixed s = sqrt(log(1 + 0.2^2)).
- Linked: fixed s, with log-correlation +0.6 between GPP and Ra in each group;
  all other off-diagonals zero. This is a sensitivity assumption, not a measured
  biological error correlation. Existing temporal templates are unchanged.
- Hierarchical: the linked C and one shared s ~ HalfNormal(a). By default
  `a^2 = [1 - exp(-2 s_fixed^2)]/2`, giving the same marginal multiplier SD 0.2.
  This pools dispersion around a fixed mean, not a learned population mean.

Fixed-width multiplier covariance is `exp(s^2 C_ij)-1`. Hierarchical covariance
is `(1-2 a^2 C_ij)^(-1/2)-1`. The latter requires a < 1/sqrt(2) for finite
marginal variance. A lognormal hyperprior on s would have infinite marginal
coefficient variance despite preserving its mean. Shared s induces dependence
between groups even where their marginal covariance is zero.

The model uses existing registered coordinates, sensitivity preparation,
correlated state, affine contribution and additive Gaussian likelihood
components. Observation SD is fixed (default 1 ppm); there are no likelihood
hyperparameters competing with the flux-prior width. The hierarchy is
non-centred but can still be harder to sample than fixed priors.

`run_process_biosphere_probe.py` reads a campaign-owned NetCDF containing H,
reference_flux, labelled truth_scaling/observations, fixed_prior_contribution,
and observation-support masks. It also offers a linked-net-matched control:
one fixed s is chosen before fitting to match the independent prior variance
of the supplied signed net functional. It does not match other prior moments.
The driver uses RhimeSampler/numpyro with four chains and 1000/1000 by default.

Example (float64 and writable compilation caches must be configured):

```bash
python prototypes/run_process_biosphere_probe.py --input probe_cases.nc \
  --output results --arm hierarchical --scenario base --support all_hours
```

The three named sources must really be available in the input product. TER
alone cannot recover Ra/Rh; introducing them changes the state space. Group
coefficients remain constant over the emission support represented by each
column. Time-varying or lagged coupling needs newly convolved templates.
Correlation and width hyperparameters cannot create information missing from
CO2. A few controlled cases screen geometry and recovery, not uncertainty
coverage or biological validity. Test coupling-violating truths and hyperprior
sensitivity before drawing wider conclusions.
