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
correlation between very skewed states), or a latent correlation that is not
positive definite, fails clearly without repair.

Activity
--------

Activity policies fix explicit inactive output coefficients. The model samples
the joint marginal of the active states, and a fully fixed state vector needs
no latent draw.
