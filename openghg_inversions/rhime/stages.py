"""Public file-backed RHIME stages and explicit family dispatch.

Select a concrete module once with ``select_stages`` before calling its named
operations. Family modules document preparation, sampling and replay contracts;
shared artifact and diagnostic owners are independent of this facade.
See :ref:`staged-rhime-python` for Python invocation and
:ref:`staged-rhime-lifecycle` for checkpoint ownership.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from collections.abc import Callable, Mapping

from ._stage_checks import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    diagnose_rhime_stage,
)
from .outputs import RhimeResult
from .params import StandardRecipeConfig

if TYPE_CHECKING:
    from .co2.configuration import Co2RecipeConfig

ModelKind = Literal["standard", "multisector", "co2"]


class StageOperations(Protocol):
    """Named file-backed operations implemented by a concrete recipe module.

    Select once before resolving configuration and invoking a stage. Numerical
    builders and partial recipes do not need this complete interface.
    """

    load_params: Callable[..., dict[str, Any]]
    resolve_config: Callable[..., Any]
    effective_configuration: Callable[..., dict[str, Any]]
    configuration_identity: Callable[..., str]
    prepare: Callable[..., dict[str, Any]]
    prior_predictive: Callable[..., dict[str, Any]]
    sample: Callable[..., dict[str, Any]]
    postprocess: Callable[..., RhimeResult]


def select_stages(model: ModelKind) -> StageOperations:
    """Select the concrete staged recipe, rejecting unsupported families."""
    if model == "standard":
        from . import _standard_stages

        return cast(StageOperations, _standard_stages)
    if model == "multisector":
        from . import _multisector_stages

        return cast(StageOperations, _multisector_stages)
    if model == "co2":
        from .co2 import stages

        return cast(StageOperations, stages)
    raise ValueError(f"Unsupported staged model {model!r}.")


def load_stage_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    model: ModelKind = "standard",
) -> dict[str, Any]:
    """Load existing RHIME parameters from one explicit source.

    ``CONFIG_FILE`` is intentionally not an environment default: an unrelated
    ambient variable must never select the scientific configuration.
    """

    return select_stages(model).load_params(
        config_file=config_file, params_file=params_file, overrides=overrides
    )


def resolve_stage_setup(params: Mapping[str, Any], *, model: ModelKind) -> StandardRecipeConfig | Co2RecipeConfig:
    """Resolve stage parameters through the canonical RHIME boundary."""

    return select_stages(model).resolve_config(params)


def effective_configuration(setup: StandardRecipeConfig | Co2RecipeConfig, *, model: ModelKind) -> dict[str, Any]:
    """Return the resolved scientific configuration used by every stage."""

    return select_stages(model).effective_configuration(setup)


def configuration_identity(setup: StandardRecipeConfig | Co2RecipeConfig, *, model: ModelKind) -> str:
    """Hash the resolved settings required for scientific replay.

    Standard and multisector identities cover preparation, period, model,
    and prior choices. CO2 identities cover recipe replay settings; separate
    artifact content hashes authenticate the prepared data.
    """

    return select_stages(model).configuration_identity(setup)


def prepare_rhime_stage(
    *,
    setup: StandardRecipeConfig | Co2RecipeConfig,
    model: ModelKind,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Dispatch preparation to the selected family's documented checkpoint contract."""

    return select_stages(model).prepare(setup=setup, output_dir=output_dir)


def prior_predictive_stage(
    *,
    setup: StandardRecipeConfig | Co2RecipeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Dispatch prior readiness using the selected family's error boundary.

    Standard/multisector catch construction/prediction KeyError or ValueError;
    CO2 execution errors propagate. All families propagate external loading,
    authentication and serialization failures. See the concrete operation
    docstrings for returned checks and artifact writes.
    """

    return select_stages(model).prior_predictive(
        setup=setup,
        prepared_inputs=prepared_inputs,
        output_dir=output_dir,
        check_output=check_output,
        preparation_manifest=preparation_manifest,
        draws=draws,
        stage=stage,
    )


def sample_rhime_stage(
    *,
    setup: StandardRecipeConfig | Co2RecipeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Sample prepared inputs without invoking preparation.

    Each family writes its declared manifest contract. Standard/multisector
    bind saved output roles; CO2 authenticates supplied affine artifacts separately.
    """

    return select_stages(model).sample(
        setup=setup,
        prepared_inputs=prepared_inputs,
        output_dir=output_dir,
        preparation_manifest=preparation_manifest,
    )


def postprocess_rhime_stage(
    *,
    setup: StandardRecipeConfig | Co2RecipeConfig,
    model: ModelKind,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
    sample_manifest: str | Path,
) -> RhimeResult:
    """Build products from authenticated saved artifacts.

    Supported family contracts replay without graph construction or resampling.
    Retired contracts and invalid bindings fail before posterior loading or writes.
    """

    return select_stages(model).postprocess(
        setup=setup,
        prepared_inputs=prepared_inputs,
        posterior=posterior,
        output_dir=output_dir,
        preparation_manifest=preparation_manifest,
        sample_manifest=sample_manifest,
    )


__all__ = [
    "CHECK_SCHEMA_VERSION",
    "CONVERGENCE_CHECK_NAME",
    "PREPARATION_CHECK_NAME",
    "ModelKind",
    "StageOperations",
    "select_stages",
    "diagnose_rhime_stage",
    "load_stage_params",
    "resolve_stage_setup",
    "effective_configuration",
    "configuration_identity",
    "prepare_rhime_stage",
    "prior_predictive_stage",
    "sample_rhime_stage",
    "postprocess_rhime_stage",
]
