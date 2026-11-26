Retained flux prior
===================

The CO₂ and linked CO₂/O₂ recipes sample one retained flux-scaling state
:math:`x`. Its prior is a
:class:`~openghg_inversions.correlated_state.CorrelatedPrior`: the retained
arithmetic mean :math:`m` and covariance :math:`B` from preparation, with a
marginal family per state. The ordinary and cached CO₂ builders and the linked
CO₂/O₂ builders take one ``retained_prior`` value. They also accept a validated
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior` directly.
The CO₂ runners build the prior from the prepared mean and covariance. The
linked runners use the lognormal prior from their prepared inputs.

.. _sampled-marginals:

Sampled marginals
-----------------

The states are joined by a Gaussian copula that reproduces the arithmetic
moments. Lognormal is the only marginal available, built by
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior`:

.. math::

   S_{ij}&=\log(1+B_{ij}/(m_i m_j)),\\
   a_i&=\log(m_i)-S_{ii}/2,\\
   x&=\exp(a+Lz),\qquad LL^T=S,\quad z\sim N(0,I).

The logarithm arguments and positive definiteness of :math:`S` must be valid.
Positive-definite arithmetic covariance alone does not guarantee valid
multivariate lognormal moments; incompatible moments fail clearly. The
arithmetic mean and covariance are preserved exactly, but a linear functional
such as net flux keeps only its mean and variance, not all of its higher
moments.

Activity
--------

Activity policies fix explicit inactive output coefficients. The model samples
the joint marginal of the active states, and a fully fixed state vector needs
no latent draw.
