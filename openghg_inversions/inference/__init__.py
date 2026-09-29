"""Inference mechanics shared by explicit scientific recipes."""

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from .sampling import RhimeSampler

__all__ = ["RhimeSampler"]
