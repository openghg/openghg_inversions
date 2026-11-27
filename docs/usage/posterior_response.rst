Directional posterior response
==============================

``openghg_inversions.posterior_response`` provides optional analysis of saved
posterior draws, independently of model construction. Choose a small set of
physical flux functionals and concentration perturbations before examining
their response. The utility requires neither a dense gain matrix nor a new
inversion. Campaign-specific data loading, physical target construction and
replay belong to the caller.

Define the intervention
-----------------------

For a Gaussian likelihood with mean :math:`\mu` and covariance :math:`R`, use
the intervention :math:`\mu_\lambda=\mu-\lambda e`. For example, :math:`e` can
be the complete prior baseline minus a reference background. Include each
fixed affine, exterior-flux and boundary contribution once when defining that
baseline. With dimensionless :math:`\lambda`, :math:`e` has concentration units.
The likelihood score and local response of a fixed physical target :math:`q`
are

.. math::

   s_e=e^T R^{-1}(\mu-y), \qquad
   \left.\frac{d E_\lambda[q\mid y]}{d\lambda}\right|_0
   =\operatorname{Cov}(q,s_e\mid y).

This score includes each draw's inferred covariance. Bind the solver to that
draw's covariance, including inferred error amplitudes; substituting a posterior
mean covariance defines a different calculation. All score functions require
``covariance_depends_on_perturbation=False`` explicitly and reject ``True``:
covariance dependence on :math:`\lambda` requires additional score terms that
these functions do not implement. Dependence on sampled parameters is allowed.
The prior, direction and target must be independent of :math:`\lambda`, support
must remain stable, and differentiation must be permitted under the posterior
integral.

Pass ``mu_minus_y`` as a NumPy vector with shape ``(observation,)`` and
``directions`` as a NumPy matrix with shape ``(observation, direction)``.
``solve(rhs)`` returns :math:`R^{-1}\mathrm{rhs}` in the same shape and may use
an existing structured, sparse or matrix-free solver. Score evaluation is an
explicit eager numerical boundary. Resolve labels and materialize related lazy
inputs together before calling it. Keep the original chain and draw order when
collecting the returned score vectors.

.. code-block:: python

   import numpy as np
   from openghg_inversions.posterior_response import likelihood_score

   # One draw; production callers bind their structured solver to this draw's R.
   residual = np.array([0.2, -0.1])  # mu - y
   directions = np.array([[1.0], [0.5]])
   variance = np.array([2.0, 3.0])
   def solve(rhs):
       return rhs / (variance[:, None] if rhs.ndim == 2 else variance)

   score = likelihood_score(
       residual, directions, solve,
       covariance_depends_on_perturbation=False,
   )

Gaussian nuisance alternatives
-------------------------------

Suppose :math:`\mu=\mu_0+D\beta` and the nuisance prior is an independent,
untruncated :math:`N(m,C)` with positive-definite :math:`C`. The likelihood
covariance, nuisance design and perturbation directions must be independent of
:math:`\beta` conditional on the remaining sampled parameters. Only the nuisance
prior needs to be Gaussian; the flux
prior can remain non-Gaussian.

``gaussian_conditional_score`` accepts ``mu_without_nuisance_minus_y``,
``directions``, ``nuisance_design`` (:math:`D`), ``prior_mean`` (:math:`m`),
``prior_covariance`` (:math:`C`) and the draw-specific ``solve`` callable.
It analytically averages the score over the Gaussian conditional distribution
of :math:`\beta`. The target :math:`q` must not depend on the integrated
nuisance coefficients. This is appropriate for flux-only functionals; using
the averaged score with a target containing :math:`\beta` would omit a
conditional covariance term.

``gaussian_prior_shift_score`` accepts the joint ``mu_minus_y``, ``directions``,
sampled ``nuisance`` (:math:`\beta`), ``nuisance_design``, fixed
``translations`` (:math:`d`, shape ``(nuisance, direction)``), ``prior_mean``,
``prior_covariance`` and ``solve``. For an independently chosen approximation
:math:`Dd\approx e`, it evaluates

.. math::

   s_{\mathrm{alt}}=(e-Dd)^T R^{-1}(\mu-y)
                     -d^T C^{-1}(\beta-m).

The residual :math:`e-Dd` is retained exactly, even when the projection is poor.
Choose :math:`d` once before draw evaluation. The target must be invariant under
the nuisance translation, and other model terms must satisfy the stated linear,
independent Gaussian nuisance assumptions.

These alternatives are exact under their assumptions. Applying them to draws
from a positive truncated-normal prior, including historical
:math:`\operatorname{TN}(1,0.01^2)` boundary priors, is a **Gaussian
approximation**, and saved artifacts and reports must say so. A negligible
conditional tail probability on sampled draws is useful evidence about that
approximation, but does not establish a global bound on posterior moment error.
Conditional averaging need not reduce MCMC asymptotic variance; compare MCSE
empirically.

Estimate response and Monte Carlo error
---------------------------------------

Supply labelled xarray arrays ``q`` and ``score`` with matching ``chain`` and
``draw`` coordinates to ``response_summary``. Use separate nonsample dimensions,
such as ``functional`` for targets and ``direction`` or ``estimator`` for scores.
The function preserves lazy arrays and uses global centering with normalization
:math:`1/N`, where :math:`N` is the total number of retained draws across chains:

.. math::

   p_{ct}=(q_{ct}-\bar q)(s_{ct}-\bar s),\qquad
   \widehat g=\overline p,\qquad I_{ct}=p_{ct}-\widehat g.

The result includes ``gain``, ``centered_product``, ``influence`` and
``chain_contribution`` (the within-chain mean of the globally centred products).
The chain contribution is not a separately centred within-chain covariance.
Keep units and scientific target definitions with the caller's data; response
has units of target per unit perturbation.
Missing values propagate through the reductions instead of silently changing
the sample count.

``response_diagnostics(summary)`` is the explicit eager diagnostic boundary.
It densifies sparse payloads and computes related target, score and influence
arrays together, then applies ArviZ diagnostics with chain and
draw axes intact. Its ``metric`` axis includes mean MCSE, bulk and tail
effective sample sizes (ESS), and rank-normalized R-hat. Missing or undefined
diagnostics, including constant or insufficient series, require interpretation;
they do not certify convergence. MCSE measures numerical uncertainty in the
estimated derivative, rather than posterior uncertainty in flux.

For two estimators on the same draws, call
``paired_response_summary(q, score_a, score_b)`` and diagnose that result.
The difference is A minus B; its influence is the difference of their paired
influence series. Adding two marginal MCSEs in quadrature would discard their
same-draw dependence. Exact sample-coordinate matching is required; do not
flatten or reorder draws independently.

.. code-block:: python

   from openghg_inversions.posterior_response import (
       paired_response_summary, response_diagnostics, response_summary,
   )

   # q: (chain, draw, functional); score: (chain, draw, direction)
   summary = response_summary(q, score)
   diagnostics = response_diagnostics(summary)
   paired = paired_response_summary(q, score_conditional, score)
   paired_diagnostics = response_diagnostics(paired)
   summary.to_netcdf("local_response.nc")  # explicit serialization boundary

Report response, MCSE, ESS, R-hat and per-chain contributions together. A well
mixed but highly variable influence series can still leave the derivative's
sign unresolved. Splitting :math:`e` into its retained-row constant mean and
remaining structure gives additive scores and responses; diagnose each part
and their sum using the same draws.

Interpretation limits
----------------------

The result is local at :math:`\lambda=0`, averaged over the fitted joint
posterior. It does not predict a finite correction at :math:`\lambda=1`, remove
boundary or exterior degrees of freedom, or measure error against flux truth.
A noisy derivative must not be presented as a reliable flux correction.
Independent finite-intervention comparisons and truth-based assessments answer
different questions. Preserve the intervention, target units, solver/covariance
definition, prior assumptions and approximation labels with saved outputs.

The score-covariance identity is a standard Bayesian sensitivity result; see
`Giordano, Broderick and Jordan (2018), Theorem 1
<https://jmlr.org/papers/v19/17-670.html>`_.

.. seealso::

   :doc:`The API reference </reference/openghg_inversions.posterior_response>`
   describes the individual function arguments.
