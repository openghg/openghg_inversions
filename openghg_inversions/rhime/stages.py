"""Standard numerical handoffs and retained cached CO2 staged entry points.

Multisector handoffs live in :mod:`.multisector_stages`. Each concrete recipe
owns its scientific execution; the old multisector envelope is retired.
Ordinary CO2 uses ``co2.handoffs`` and rejects these retained envelope wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from collections.abc import Mapping

from ._standard_stages import prepare, sample, postprocess, prior_predictive, load_preparation, load_samples, save_preparation, save_samples
from ._stage_checks import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    diagnose_rhime_stage,
)
from .requests import load_params

if TYPE_CHECKING:
    from .co2.co2_outputs import Co2Output
    from .co2.stages import Co2StageSetup

ModelKind = Literal["co2"]


def _reject_migrated(model: str, function: str, replacement: str) -> None:
    """Name the concrete replacement for retired family envelopes."""
    if model in {"standard", "multisector"}:
        module = "stages" if model == "standard" else "multisector_stages"
        raise ValueError(
            f"`{function}` no longer supports model={model!r}; use `{module}.{replacement}` "
            "with a preparation or sample manifest."
        )
    if model != "co2":
        raise ValueError(f"Unsupported staged model {model!r}.")


def load_stage_params(
    *,
    config_file: str | Path | None = None,
    params_file: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    model: Literal["standard", "multisector", "co2"] = "standard",
) -> dict[str, Any]:
    """Load existing RHIME parameters from one explicit source.

    ``CONFIG_FILE`` is intentionally not an environment default: an unrelated
    ambient variable must never select the scientific configuration. Standard
    and multisector file resource paths and explicit overrides resolve from the
    invocation working directory. The retained CO2 loader resolves file resource
    paths beside the configuration file and overrides from the invocation working
    directory. Absolute paths retain their destinations.

    Args:
        config_file: Explicit configuration file; pass exactly one source.
        params_file: Explicit JSON parameter file; pass exactly one source.
        overrides: Invocation options applied over file values.
        model: Family whose parameter loader is used; defaults to standard.

    Returns:
        Standard/multisector options grouped by data, model, sampler and output
        owner, or the retained CO2 family's parameter mapping.

    Raises:
        ValueError: If neither or both sources are supplied, or the source options
            are invalid for the selected loader.
    """

    if model == "co2":
        from .co2.stages import load_co2_stage_params

        return load_co2_stage_params(config_file=config_file, params_file=params_file, overrides=overrides)
    if (config_file is None) == (params_file is None):
        raise ValueError("Pass exactly one of `config_file` or `params_file`.")
    return load_params(config_file=config_file, params_file=params_file, overrides=overrides)


def resolve_stage_setup(params: Mapping[str, Any], *, model: ModelKind) -> Co2StageSetup:
    """Resolve effective stage options through the selected recipe's boundary.

    Args:
        params: File-derived or Python options after invocation overrides and
            staged path resolution.
        model: Recipe whose configuration and staged workflow will be used.

    Returns:
        The CO2 workflow's setup. Resolution does not acquire scientific data
        or execute inference.

    Raises:
        ValueError: If the recipe is unsupported or its effective options are
            invalid.
    """

    if model == "co2":
        from .co2.stages import resolve_co2_stage_setup

        return resolve_co2_stage_setup(params=params)
    _reject_migrated(model, "resolve_stage_setup", "prepare")
    raise ValueError(f"Unsupported staged model {model!r}.")


def effective_configuration(setup: Co2StageSetup, *, model: ModelKind) -> dict[str, Any]:
    """Project resolved choices into the selected workflow's manifest vocabulary.

    Args:
        setup: Resolved configuration for the selected recipe.
        model: Recipe that owns the manifest representation.

    Returns:
        CO2 recipe, reconstruction and output settings for provenance and replay.
        This is a staged artifact representation, not a general configuration
        serialization API or a retained execution run specification.
    """

    if model == "co2":
        from .co2.stages import effective_co2_configuration

        return effective_co2_configuration(setup=cast("Co2StageSetup", setup))
    _reject_migrated(model, "effective_configuration", "load_preparation")
    raise ValueError(f"Unsupported staged model {model!r}.")


def configuration_identity(setup: Co2StageSetup, *, model: ModelKind) -> str:
    """Hash the resolved settings required for scientific replay.

    CO2 identities cover recipe replay settings. Numerical
    artifacts are loaded through their scientific owners without stage digests.
    """

    if model == "co2":
        from .co2.stages import co2_configuration_identity

        return co2_configuration_identity(setup=cast("Co2StageSetup", setup))
    _reject_migrated(model, "configuration_identity", "load_preparation")
    raise ValueError(f"Unsupported staged model {model!r}.")


def prepare_rhime_stage(
    *,
    setup: Co2StageSetup,
    model: ModelKind,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare and save inputs for the retained CO2 staged workflow.

    Args:
        setup: Resolved configuration from :func:`resolve_stage_setup`.
        model: ``"co2"``; standard uses :func:`prepare`.
        output_dir: Stage directory receiving the prepared inputs and manifest.

    Returns:
        The written preparation manifest, including its ``manifest_path``.

    Raises:
        ValueError: If the model is standard or multisector or a requested site was not prepared.
    """

    _reject_migrated(model, "prepare_rhime_stage", "prepare")

    if model == "co2":
        from .co2.stages import prepare_co2_stage

        return prepare_co2_stage(setup=cast("Co2StageSetup", setup), output_dir=output_dir)
    raise ValueError(f"Unsupported staged model {model!r}.")


def prior_predictive_stage(
    *,
    setup: Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    check_output: str | Path | None = None,
    preparation_manifest: str | Path,
    draws: int = 100,
    stage: str = "prior-predictive",
) -> dict[str, Any]:
    """Check finite prior-predictive draws for the retained CO2 workflow.

    Args:
        setup: Resolved configuration matching the preparation.
        model: ``"co2"``; standard uses :func:`prior_predictive`.
        prepared_inputs: Saved prepared inputs.
        output_dir: Stage directory receiving the evidence and report.
        check_output: Optional report path beneath ``output_dir``.
        preparation_manifest: Manifest matching ``prepared_inputs``.
        draws: Positive number of prior-predictive draws.
        stage: Stage label recorded in the report.

    Returns:
        The readiness check written to the report.

    Raises:
        ValueError: If the model is standard or multisector, ``draws`` or ``stage`` is invalid, or
            the preparation does not match ``setup``.
    """

    _reject_migrated(model, "prior_predictive_stage", "prior_predictive")

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
    raise ValueError(f"Unsupported staged model {model!r}.")


def sample_rhime_stage(
    *,
    setup: Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
) -> dict[str, Any]:
    """Sample saved inputs for the retained CO2 workflow.

    CO2 retains its version-1 manifest.

    Args:
        setup: Resolved configuration matching the preparation.
        model: ``"co2"``; standard uses :func:`sample`.
        prepared_inputs: Saved prepared inputs.
        output_dir: Stage directory receiving the posterior and manifest.
        preparation_manifest: Manifest matching ``prepared_inputs``.

    Returns:
        The written sample manifest, including its ``manifest_path``.

    Raises:
        ValueError: If the model is standard or multisector or the preparation does not match ``setup``.
    """

    _reject_migrated(model, "sample_rhime_stage", "sample")

    if model == "co2":
        from .co2.stages import sample_co2_stage

        return sample_co2_stage(
            setup=cast("Co2StageSetup", setup),
            prepared_inputs=prepared_inputs,
            output_dir=output_dir,
            preparation_manifest=preparation_manifest,
        )
    raise ValueError(f"Unsupported staged model {model!r}.")


def postprocess_rhime_stage(
    *,
    setup: Co2StageSetup,
    model: ModelKind,
    prepared_inputs: str | Path,
    posterior: str | Path,
    output_dir: str | Path,
    preparation_manifest: str | Path,
    sample_manifest: str | Path,
) -> Co2Output:
    """Make products from saved samples for the retained CO2 workflow.

    The retained CO2 route replays its scientific output context.

    Args:
        setup: Resolved configuration matching the saved artifacts.
        model: ``"co2"``; standard uses :func:`postprocess`.
        prepared_inputs: Saved prepared inputs.
        posterior: Saved posterior trace.
        output_dir: Stage directory receiving the products and manifest.
        preparation_manifest: Manifest matching ``prepared_inputs``.
        sample_manifest: Manifest matching ``posterior``.

    Returns:
        The replayed result.

    Raises:
        ValueError: If the model is standard or multisector, the artifacts do not match ``setup``,
            or the output binding is invalid.
    """

    _reject_migrated(model, "postprocess_rhime_stage", "postprocess")

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
    raise ValueError(f"Unsupported staged model {model!r}.")


__all__ = [
    "prepare", "sample", "postprocess", "prior_predictive",
    "save_preparation", "load_preparation", "save_samples", "load_samples",
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
