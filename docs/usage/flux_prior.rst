Retained flux prior
===================

The CO₂ and linked CO₂/O₂ recipes sample one retained flux-scaling state
:math:`x`. Its prior is a
:class:`~openghg_inversions.correlated_state.CorrelatedPrior`: the retained
arithmetic mean :math:`m` and covariance :math:`B` from preparation, with a
marginal family per state. The ordinary and cached CO₂ builders and the linked
CO₂/O₂ builders take this one ``retained_prior`` value. The runners build it
from the prepared mean and covariance; the linked runners convert the prepared
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior` with
:meth:`~openghg_inversions.correlated_state.CorrelatedPrior.from_lognormal`.

.. _sampled-marginals:

Sampled marginals
-----------------

The model samples a
:class:`~openghg_inversions.correlated_state.CorrelatedPrior`: arithmetic
moments joined by a Gaussian copula, with a marginal family per state. A whitened
:math:`z\sim N(0,I)` is correlated as :math:`u=Lz` and each state is
:math:`x_i=F_i^{-1}(\Phi(u_i))`, where :math:`F_i` has the state's arithmetic
mean and SD. The latent correlations :math:`LL^T` are chosen pair by pair so
that the arithmetic covariance is reproduced exactly.

``"lognormal"`` is the default. With it on every state the prior is
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior` and the
model is built exactly as before:

.. math::

   S_{ij}&=\log(1+B_{ij}/(m_i m_j)),\\
   a_i&=\log(m_i)-S_{ii}/2,\\
   x&=\exp(a+Lz),\qquad LL^T=S,\quad z\sim N(0,I).

The logarithm arguments and positive definiteness of :math:`S` must be valid.
Positive-definite arithmetic covariance alone does not guarantee valid
multivariate lognormal moments. Incompatible requests fail clearly. Arithmetic
mean/covariance are preserved exactly, but preserving net mean and variance
does not preserve all its higher moments.

The other marginals have the same mean and SD with less right skew:

- ``"normal"``: :math:`x_i=m_i+s_i u_i`. Symmetric, so a posterior constrained
  only in a sum of states (net flux) is not pulled towards small gross terms,
  but scalings can be negative (about 6% probability at CV 0.63). Mainly a
  test and comparison family.
- ``"weibull"`` and ``"gamma"``: positive and defined for any CV. A mean-one
  lognormal at CV 0.63 has median 0.85; the Weibull 0.89 and gamma 0.87. In
  OPE-188 July 2021 fits, Weibull GPP/TER marginals removed about two thirds
  of the lognormal's gross shrinkage relative to a Gaussian in CO2-only fits
  and about half in linked fits; gamma removed a fifth to a quarter. Gamma
  states are held at their quantile for latent :math:`u=8` above it
  (probability 6e-16), because the JAX inverse incomplete gamma overflows
  near :math:`u=10`; both samplers use the same map.

A pair with a normal state, or two lognormal states, has a closed-form latent
correlation. Weibull and gamma pairs solve a 60-term Hermite series for it.
A pair covariance no latent correlation can reach (for example strong negative
correlation between very skewed states) fails clearly.

.. _latent-correlation-repair:

Latent-correlation repair
~~~~~~~~~~~~~~~~~~~~~~~~~

Each latent correlation is solved pair by pair, so the whole matrix need not be
positive definite even when the arithmetic covariance is. This happens with
wide, strongly correlated priors: for example, the OPE-188 July 2021 retained
prior with ORCHIDEE GPP/TER means (CV up to 2.3) and Weibull GPP/TER marginals
has a latent minimum eigenvalue of :math:`-5.3\times10^{-4}`. Such priors fail
by default. ``latent_correlation_floor`` (a
:class:`~openghg_inversions.correlated_state.CorrelatedPrior` argument, or
``[prepared_inputs.native_prior] latent_correlation_floor`` in a configuration)
opts in to a repair: eigenvalues below the floor are raised to it, and the
matrix is rescaled to a unit diagonal. Each marginal is unchanged, so every
state keeps its mean and SD exactly. The pair covariances move slightly off
their targets. How far they move depends on how negative the eigenvalues were,
not on the floor, so a repair that would move any implied state correlation by
more than 0.01 is refused: such pairwise targets do not belong to one copula.
For example, five Weibull states with CV 1.5 and equal arithmetic correlation
-0.24 have a positive-definite arithmetic covariance, but a latent minimum
eigenvalue of about -0.9, and repairing it would change state correlations by
about 0.1.

The repair is an eigenvalue floor rather than the nearest correlation matrix
(Higham's alternating projections). It is one deterministic step with no
convergence tolerance, and it always yields a positive-definite matrix, which
the Cholesky factor needs; Higham's result is only semidefinite and needs a
floor anyway. For a few small negative eigenvalues the two differ negligibly.
A floor of ``1e-6`` is a reasonable choice: lower values give the same
covariance but a worse-conditioned factor. On the ORCHIDEE prior above, a
floor of ``1e-6`` raises one eigenvalue, changes latent correlations by at
most :math:`3.0\times10^{-4}` and implied state correlations by at most
:math:`4.4\times10^{-4}`.

:attr:`~openghg_inversions.correlated_state.CorrelatedPrior.latent_correlation_repair`
records the floor, whether it was applied, the minimum eigenvalue before the
repair, the number of eigenvalues raised, and the largest changes in the latent
correlation and in the implied state covariance and correlation. The CO2
runners write it, the marginal family counts and the prior description into
the trace's ``rhime_model_metadata`` as ``prior_latent_correlation_repair``,
``prior_marginals`` and ``prior``. Partial activity uses the block of the
repaired correlation, so the active prior is still the marginal of the full
one. The floor does not apply when every marginal is lognormal: that prior's
latent covariance must be positive definite as before.

Activity
--------

Activity policies fix explicit inactive output coefficients. The model samples
the joint marginal of the active states, and a fully fixed state vector needs
no latent draw.
