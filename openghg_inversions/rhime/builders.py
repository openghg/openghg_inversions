"""Scientific graph results and callable provenance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pymc as pm
from pytensor.tensor.variable import TensorVariable
from openghg_inversions.postprocessing.contracts import OutputContract


RhimeLikelihoodBuilder = Callable[..., TensorVariable]


@dataclass(frozen=True)
class RhimeModelBuildResult:
    """Concrete graph and its validated, serializable scientific output contract.

    Args:
        model: Concrete PyMC model for sampling.
        output_contract: Semantic variable roles, supported output formats and
            construction metadata. Custom models default to sampling-only
            capability when constructing an ordinary ``OutputContract``.
    """

    model: pm.Model
    output_contract: OutputContract


def callable_metadata(builder: Callable[..., Any]) -> dict[str, str]:
    """Return stable, serializable direct-Python callable identity metadata.

    Args:
        builder: Callable whose import module and qualified name identify the
            runtime customization.

    Returns:
        JSON-serializable module and qualified-name fields.
    """
    return {
        "module": getattr(builder, "__module__", type(builder).__module__),
        "qualname": getattr(builder, "__qualname__", type(builder).__qualname__),
    }
