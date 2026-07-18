"""Experimental NumPy/Numba engine for spatial trans-dimensional MCMC.

The package separates fixed-capacity Voronoi problems and states, explicit
Metropolis-Hastings proposal accounting, deterministic seeded sampling, and
adaptation of filtered RHIME fine-grid inputs. The initial implementation is
limited to a single trans-dimensional sector with independent Gaussian
observation errors and lognormal region coefficients. Correlated errors,
boundary-condition inference, multi-sector models, and parallel tempering are
outside this initial slice.
"""

from openghg_inversions.tdmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    aggregate_design_numba,
    aggregate_design_numpy,
    assign_cells_numba,
    assign_cells_numpy,
    build_state,
    gaussian_log_likelihood_numba,
    gaussian_log_likelihood_numpy,
    lognormal_coefficient_log_prior_numba,
    lognormal_coefficient_log_prior_numpy,
    uniform_log_k_prior,
    uniform_nucleus_set_log_prior,
)
from openghg_inversions.tdmcmc.proposals import (
    TransitionTerms,
    accept_or_reject,
    propose_birth,
    propose_coefficient,
    propose_death,
    propose_global_move,
    propose_local_move,
)
from openghg_inversions.tdmcmc.postprocessing import (
    DEFAULT_QUANTILES,
    FineGridPosteriorSummary,
    posterior_mean_prediction,
    reconstruct_fine_grid_samples,
    summarize_fine_grid_posterior,
)
from openghg_inversions.tdmcmc.rhime_adapter import problem_from_rhime_inputs
from openghg_inversions.tdmcmc.sampling import (
    SamplerConfig,
    SamplingResult,
    SamplingTrace,
    sample,
)

__all__ = [
    "TransDimensionalProblem",
    "TransDimensionalState",
    "TransitionTerms",
    "DEFAULT_QUANTILES",
    "FineGridPosteriorSummary",
    "SamplerConfig",
    "SamplingResult",
    "SamplingTrace",
    "accept_or_reject",
    "aggregate_design_numba",
    "aggregate_design_numpy",
    "assign_cells_numba",
    "assign_cells_numpy",
    "build_state",
    "gaussian_log_likelihood_numba",
    "gaussian_log_likelihood_numpy",
    "lognormal_coefficient_log_prior_numba",
    "lognormal_coefficient_log_prior_numpy",
    "propose_birth",
    "propose_coefficient",
    "propose_death",
    "propose_global_move",
    "propose_local_move",
    "posterior_mean_prediction",
    "problem_from_rhime_inputs",
    "reconstruct_fine_grid_samples",
    "sample",
    "summarize_fine_grid_posterior",
    "uniform_log_k_prior",
    "uniform_nucleus_set_log_prior",
]
