"""Public file-backed RHIME stages and explicit family dispatch.

Concrete workflows own scientific execution and replay policy. Shared artifact
and check owners are independent of this facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

from . import _standard_stages
from ._stage_checks import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    diagnose_rhime_stage,
)
from .outputs import RhimeResult
from .params import RhimeRunnerSetup

if TYPE_CHECKING:
    from .co2.stages import Co2StageSetup

ModelKind = Literal["standard", "multisector", "co2"]


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

    if model == "co2":
        from .co2.stages import load_co2_stage_params

        return load_co2_stage_params(config_file=config_file, params_file=params_file, overrides=overrides)
    return _standard_stages.load_stage_params(
        config_file=config_file, params_file=params_file, overrides=overrides
    )


def resolve_stage_setup(params: Mapping[str, Any], *, model: ModelKind) -> RhimeRunnerSetup | Co2StageSetup:
    """Resolve stage parameters through the canonical RHIME boundary."""

    if model == "co2":
        from .co2.stages import resolve_co2_stage_setup

        return resolve_co2_stage_setup(params=params)
    if model not in ("standard", "multisector"):
        raise ValueError(f"Unsupported staged model {model!r}.")
    return _standard_stages.resolve_stage_setup(params=params, model=cast(_standard_stages.ModelKind, model))


def effective_configuration(setup: RhimeRunnerSetup | Co2StageSetup, *, model: ModelKind) -> dict[str, Any]:
    """Return the resolved scientific configuration used by every stage."""

    if model == "co2":
        from .co2.stages import effective_co2_configuration

        return effective_co2_configuration(setup=cast("Co2StageSetup", setup))
    return _standard_stages.effective_configuration(
        setup=cast(RhimeRunnerSetup, setup), model=cast(_standard_stages.ModelKind, model)
    )


def configuration_identity(setup: RhimeRunnerSetup | Co2StageSetup, *, model: ModelKind) -> str:
    """Hash the resolved settings required for scientific replay.

    Standard and multisector identities cover preparation, period, model,
    and prior choices. CO2 identities cover recipe replay settings; separate
    artifact content hashes authenticate the prepared data.
    """

    if model == "co2":
        from .co2.stages import co2_configuration_identity

        return co2_configuration_identity(setup=cast("Co2StageSetup", setup))
    return _standard_stages.configuration_identity(
        setup=cast(RhimeRunnerSetup, setup), model=cast(_standard_stages.ModelKind, model)
    )


def prepare_rhime_stage(
    *,
    setup: RhimeRunnerSetup | Co2StageSetup,
    model: ModelKind,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare and persist independently inspectable RHIME inputs."""

    if model == "co2":
        from .co2.stages import prepare_co2_stage

        return prepare_co2_stage(setup=cast("Co2StageSetup", setup), output_dir=output_dir)
    return _standard_stages.prepare_rhime_stage(
        setup=cast(RhimeRunnerSetup, setup),
        model=cast(_standard_stages.ModelKind, model),
        output_dir=output_dir,
    )


def prior_predictive_stage(
    *,
    setup: RhimeRunnerSetup | Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Build the configured graph and check finite prior-predictive draws."""

    if model == "co2":
        from .co2.stages import prior_predictive_co2_stage

        return prior_predictive_co2_stage(
            setup=cast("Co2StageSetup", setup),
            prepared_inputs=prepared_inputs,
            output_dir=output_dir,
            check_output=check_output,
            preparation_manifest=preparation_manifest,
            draws=draws,
            stage=stage,
        )
    return _standard_stages.prior_predictive_stage(
        setup=cast(RhimeRunnerSetup, setup),
        model=cast(_standard_stages.ModelKind, model),
        prepared_inputs=prepared_inputs,
        output_dir=output_dir,
        check_output=check_output,
        preparation_manifest=preparation_manifest,
        draws=draws,
        stage=stage,
    )


def sample_rhime_stage(
    *,
    setup: RhimeRunnerSetup | Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Sample prepared inputs without invoking preparation.

    Standard/multisector workflows persist version-2 sample manifests with
    output bindings. CO2 retains its version-1 manifest and authenticates any
    supplied affine artifact separately.
    """

    if model == "co2":
        from .co2.stages import sample_co2_stage

        return sample_co2_stage(
            setup=cast("Co2StageSetup", setup),
            prepared_inputs=prepared_inputs,
            output_dir=output_dir,
            preparation_manifest=preparation_manifest,
        )
    return _standard_stages.sample_rhime_stage(
        setup=cast(RhimeRunnerSetup, setup),
        model=cast(_standard_stages.ModelKind, model),
        prepared_inputs=prepared_inputs,
        output_dir=output_dir,
        preparation_manifest=preparation_manifest,
    )


def postprocess_rhime_stage(
    *,
    setup: RhimeRunnerSetup | Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
    sample_manifest: str | Path,
) -> RhimeResult:
    """Build products from authenticated saved artifacts.

    Standard/multisector version-2 sample manifests and CO2 version-1
    manifests replay without a graph. Historical standard/multisector
    version-1 manifests reconstruct their missing output roles. Invalid
    bindings never fall back.
    """

    if model == "co2":
        from .co2.stages import postprocess_co2_stage

        return postprocess_co2_stage(
            setup=cast("Co2StageSetup", setup),
            prepared_inputs=prepared_inputs,
            posterior=posterior,
            output_dir=output_dir,
            preparation_manifest=preparation_manifest,
            sample_manifest=sample_manifest,
        )
    return _standard_stages.postprocess_rhime_stage(
        setup=cast(RhimeRunnerSetup, setup),
        model=cast(_standard_stages.ModelKind, model),
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
