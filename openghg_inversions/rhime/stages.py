"""Public file-backed RHIME stages and explicit family dispatch.

Concrete workflows own scientific execution and replay policy. Shared artifact
and check owners are independent of this facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from collections.abc import Mapping

from openghg_inversions.rhime._runner_compat import translate_rhime_aliases
from . import _multisector_stages
from ._standard_stages import prepare, sample, postprocess, prior_predictive, load_preparation, save_preparation, save_samples
from ._stage_checks import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    diagnose_rhime_stage,
)
from .outputs import RhimeResult
from .params import RhimeConfig
from .requests import load_rhime_params

if TYPE_CHECKING:
    from .co2.stages import Co2StageSetup

ModelKind = Literal["multisector", "co2"]


def _reject_standard(model: str, function: str, replacement: str) -> None:
    """Fail clearly for untyped callers selecting the retired standard envelope."""
    if model == "standard":
        raise ValueError(
            f"`{function}` no longer supports model='standard'; use `stages.{replacement}` "
            "with a standard preparation or sample manifest."
        )


def load_stage_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    model: ModelKind | Literal["standard"] = "standard",
) -> dict[str, Any]:
    """Load existing RHIME parameters from one explicit source.

    ``CONFIG_FILE`` is intentionally not an environment default: an unrelated
    ambient variable must never select the scientific configuration. Relative
    file resource paths resolve beside the configuration file; explicit overrides
    resolve from the invocation working directory. Absolute paths retain their
    destinations.
    """

    if model == "co2":
        from .co2.stages import load_co2_stage_params

        return load_co2_stage_params(config_file=config_file, params_file=params_file, overrides=overrides)
    if (config_file is None) == (params_file is None):
        raise ValueError("Pass exactly one of `config_file` or `params_file`.")
    return load_rhime_params(config_file=config_file, params_file=params_file, overrides=overrides)


def resolve_stage_setup(params: Mapping[str, Any], *, model: ModelKind) -> RhimeConfig | Co2StageSetup:
    """Resolve effective stage options through the selected recipe's boundary.

    Args:
        params: File-derived or Python options after invocation overrides and
            staged path resolution.
        model: Recipe whose configuration and staged workflow will be used.

    Returns:
        Complete ``RhimeConfig`` for standard or multisector execution, or
        the CO2 workflow's setup. Resolution does not acquire scientific data
        or execute inference.

    Raises:
        ValueError: If the recipe is unsupported or its effective options are
            invalid.
    """

    if model == "co2":
        from .co2.stages import resolve_co2_stage_setup

        return resolve_co2_stage_setup(params=params)
    _reject_standard(model, "resolve_stage_setup", "prepare")
    if model != "multisector":
        raise ValueError(f"Unsupported staged model {model!r}.")
    return RhimeConfig.from_params(translate_rhime_aliases(params), multisector=True)


def effective_configuration(setup: RhimeConfig | Co2StageSetup, *, model: ModelKind) -> dict[str, Any]:
    """Project resolved choices into the selected workflow's manifest vocabulary.

    Args:
        setup: Resolved configuration for the selected recipe.
        model: Recipe that owns the manifest representation.

    Returns:
        Manifest settings for provenance and replay. Standard and multisector
        mappings contain ``model``, ``preparation``, ``model_spec``, ``output``
        and ``sampler``; preparation includes the complete requested site
        selectors. CO2 uses its own recipe, reconstruction and output fields.
        This is a staged artifact representation, not a general configuration
        serialization API or a retained execution run specification.
    """

    if model == "co2":
        from .co2.stages import effective_co2_configuration

        return effective_co2_configuration(setup=cast("Co2StageSetup", setup))
    _reject_standard(model, "effective_configuration", "load_preparation")
    return _multisector_stages.effective_configuration(setup=cast(RhimeConfig, setup), model="multisector")


def configuration_identity(setup: RhimeConfig | Co2StageSetup, *, model: ModelKind) -> str:
    """Hash the resolved settings required for scientific replay.

    Standard and multisector identities cover preparation, period, model,
    and prior choices. CO2 identities cover recipe replay settings. Numerical
    artifacts are loaded through their scientific owners without stage digests.
    """

    if model == "co2":
        from .co2.stages import co2_configuration_identity

        return co2_configuration_identity(setup=cast("Co2StageSetup", setup))
    _reject_standard(model, "configuration_identity", "load_preparation")
    return _multisector_stages.configuration_identity(setup=cast(RhimeConfig, setup), model="multisector")


def prepare_rhime_stage(
    *,
    setup: RhimeConfig | Co2StageSetup,
    model: ModelKind,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare and save inputs for the deferred multisector or CO2 staged workflow.

    Args:
        setup: Resolved configuration from :func:`resolve_stage_setup`.
        model: ``"multisector"`` or ``"co2"``; standard uses :func:`prepare`.
        output_dir: Stage directory receiving the prepared inputs and manifest.

    Returns:
        The written preparation manifest, including its ``manifest_path``.

    Raises:
        ValueError: If ``model="standard"`` or a requested site was not prepared.
    """

    _reject_standard(model, "prepare_rhime_stage", "prepare")

    if model == "co2":
        from .co2.stages import prepare_co2_stage

        return prepare_co2_stage(setup=cast("Co2StageSetup", setup), output_dir=output_dir)
    return _multisector_stages.prepare_rhime_stage(
        setup=cast(RhimeConfig, setup),
        model="multisector",
        output_dir=output_dir,
    )


def prior_predictive_stage(
    *,
    setup: RhimeConfig | Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Check finite prior-predictive draws for the deferred multisector or CO2 workflow.

    Args:
        setup: Resolved configuration matching the preparation.
        model: ``"multisector"`` or ``"co2"``; standard uses :func:`prior_predictive`.
        prepared_inputs: Saved prepared inputs.
        output_dir: Stage directory receiving the evidence and report.
        check_output: Optional report path beneath ``output_dir``.
        preparation_manifest: Manifest matching ``prepared_inputs``.
        draws: Positive number of prior-predictive draws.
        stage: Stage label recorded in the report.

    Returns:
        The readiness check written to the report.

    Raises:
        ValueError: If ``model="standard"``, ``draws`` or ``stage`` is invalid, or
            the preparation does not match ``setup``.
    """

    _reject_standard(model, "prior_predictive_stage", "prior_predictive")

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
    return _multisector_stages.prior_predictive_stage(
        setup=cast(RhimeConfig, setup),
        model="multisector",
        prepared_inputs=prepared_inputs,
        output_dir=output_dir,
        check_output=check_output,
        preparation_manifest=preparation_manifest,
        draws=draws,
        stage=stage,
    )


def sample_rhime_stage(
    *,
    setup: RhimeConfig | Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Sample saved inputs for the deferred multisector or CO2 workflow.

    Multisector writes version-2 sample manifests with output bindings; CO2
    keeps its version-1 manifest.

    Args:
        setup: Resolved configuration matching the preparation.
        model: ``"multisector"`` or ``"co2"``; standard uses :func:`sample`.
        prepared_inputs: Saved prepared inputs.
        output_dir: Stage directory receiving the posterior and manifest.
        preparation_manifest: Manifest matching ``prepared_inputs``.

    Returns:
        The written sample manifest, including its ``manifest_path``.

    Raises:
        ValueError: If ``model="standard"`` or the preparation does not match ``setup``.
    """

    _reject_standard(model, "sample_rhime_stage", "sample")

    if model == "co2":
        from .co2.stages import sample_co2_stage

        return sample_co2_stage(
            setup=cast("Co2StageSetup", setup),
            prepared_inputs=prepared_inputs,
            output_dir=output_dir,
            preparation_manifest=preparation_manifest,
        )
    return _multisector_stages.sample_rhime_stage(
        setup=cast(RhimeConfig, setup),
        model="multisector",
        prepared_inputs=prepared_inputs,
        output_dir=output_dir,
        preparation_manifest=preparation_manifest,
    )


def postprocess_rhime_stage(
    *,
    setup: RhimeConfig | Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
    sample_manifest: str | Path,
) -> RhimeResult:
    """Make products from saved samples for the deferred multisector or CO2 workflow.

    Version-2 multisector and version-1 CO2 manifests replay without a graph;
    historical multisector version-1 manifests rebuild their missing output
    roles. Invalid bindings never fall back.

    Args:
        setup: Resolved configuration matching the saved artifacts.
        model: ``"multisector"`` or ``"co2"``; standard uses :func:`postprocess`.
        prepared_inputs: Saved prepared inputs.
        posterior: Saved posterior trace.
        output_dir: Stage directory receiving the products and manifest.
        preparation_manifest: Manifest matching ``prepared_inputs``.
        sample_manifest: Manifest matching ``posterior``.

    Returns:
        The replayed result.

    Raises:
        ValueError: If ``model="standard"``, the artifacts do not match ``setup``,
            or the output binding is invalid.
    """

    _reject_standard(model, "postprocess_rhime_stage", "postprocess")

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
    return _multisector_stages.postprocess_rhime_stage(
        setup=cast(RhimeConfig, setup),
        model="multisector",
        prepared_inputs=prepared_inputs,
        posterior=posterior,
        output_dir=output_dir,
        preparation_manifest=preparation_manifest,
        sample_manifest=sample_manifest,
    )


__all__ = [
    "prepare", "sample", "postprocess", "prior_predictive",
    "save_preparation", "load_preparation", "save_samples",
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
