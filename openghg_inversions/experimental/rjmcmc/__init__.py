"""Experimental NumPy/Numba engine for spatial reversible-jump MCMC.

The package separates fixed-capacity Voronoi problems and states, explicit
Metropolis-Hastings proposal accounting, deterministic seeded sampling, and
adaptation of filtered RHIME fine-grid inputs. The initial implementation is
limited to a single trans-dimensional sector with independent Gaussian
observation errors and lognormal region coefficients. Correlated errors,
boundary-condition inference, multi-sector models, and parallel tempering are
outside this initial slice.
"""

from openghg_inversions.experimental.rjmcmc.core import (
    FixedDesignBlock,
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
from openghg_inversions.experimental.rjmcmc.checkpoint_io import (
    CHECKPOINT_SCHEMA_ID,
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.proposals import (
    TransitionTerms,
    accept_or_reject,
    propose_birth,
    propose_coefficient,
    propose_death,
    propose_fixed_coefficient,
    propose_global_move,
    propose_local_move,
)
from openghg_inversions.experimental.rjmcmc.postprocessing import (
    DEFAULT_QUANTILES,
    FineGridPosteriorSummary,
    PosteriorPredictionSummary,
    posterior_mean_prediction,
    reconstruct_fine_grid_samples,
    summarize_fine_grid_posterior,
    summarize_posterior_prediction,
)
from openghg_inversions.experimental.rjmcmc.rhime_adapter import problem_from_rhime_inputs
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    FIXED_BLOCK_SCHEDULE_ID,
    SCHEDULE_ID,
    KernelSettings,
    PCG64State,
    SamplerCheckpoint,
    SamplerConfig,
    SamplingResult,
    SamplingTrace,
    continue_sample,
    sample,
)
from openghg_inversions.experimental.rjmcmc.xarray_output import sampling_trace_to_dataset

__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_QUANTILES",
    "FIXED_BLOCK_SCHEDULE_ID",
    "FineGridPosteriorSummary",
    "FixedDesignBlock",
    "KernelSettings",
    "PCG64State",
    "PosteriorPredictionSummary",
    "RetentionSettings",
    "SCHEDULE_ID",
    "SamplerCheckpoint",
    "SamplerConfig",
    "SamplingResult",
    "SamplingTrace",
    "TransDimensionalProblem",
    "TransDimensionalState",
    "TransitionTerms",
    "accept_or_reject",
    "aggregate_design_numba",
    "aggregate_design_numpy",
    "assign_cells_numba",
    "assign_cells_numpy",
    "build_state",
    "continue_sample",
    "gaussian_log_likelihood_numba",
    "gaussian_log_likelihood_numpy",
    "lognormal_coefficient_log_prior_numba",
    "lognormal_coefficient_log_prior_numpy",
    "load_checkpoint",
    "propose_birth",
    "propose_coefficient",
    "propose_death",
    "propose_fixed_coefficient",
    "propose_global_move",
    "propose_local_move",
    "posterior_mean_prediction",
    "problem_from_rhime_inputs",
    "reconstruct_fine_grid_samples",
    "sample",
    "sampling_trace_to_dataset",
    "save_checkpoint",
    "summarize_fine_grid_posterior",
    "summarize_posterior_prediction",
    "uniform_log_k_prior",
    "uniform_nucleus_set_log_prior",
]
