from types import SimpleNamespace
from typing import cast

import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.models import registered_model
from openghg_inversions.rhime.builders import (
    RhimeModelBuildResult,
    RhimeModelBuilderContext,
    validate_model_build_result,
)


def _sampling_only_context() -> RhimeModelBuilderContext:
    return cast(
        RhimeModelBuilderContext,
        SimpleNamespace(
            run_spec=SimpleNamespace(output=SimpleNamespace(output_format="none")),
            prepared_inputs=SimpleNamespace(inv_inputs=xr.Dataset()),
            multisector=True,
        ),
    )


@pytest.mark.parametrize("registered", [False, True])
def test_complete_model_validation_requires_coord_registry(registered: bool) -> None:
    """Complete custom models share the coordinate-registration invariant."""
    model = registered_model() if registered else pm.Model()
    with model:
        pm.Normal("custom_y")
    result = RhimeModelBuildResult(
        model=model,
        variable_roles={"concentration": "custom_y"},
    )

    if registered:
        validate_model_build_result(result, context=_sampling_only_context())
    else:
        with pytest.raises(ValueError, match="registered_model"):
            validate_model_build_result(result, context=_sampling_only_context())


def test_sampling_only_model_rejects_invalid_declared_role() -> None:
    """Disabled immediate products do not hide invalid saved scientific roles."""
    with registered_model() as model:
        pm.Normal("custom_y")
    result = RhimeModelBuildResult(model=model, variable_roles={"concentration": "absent"})
    with pytest.raises(ValueError, match="concentration='absent'"):
        validate_model_build_result(result, context=_sampling_only_context())


def test_build_result_retains_one_validated_contract():
    with registered_model() as model:
        pm.Normal("y")
    roles = {"concentration": "y"}
    result = RhimeModelBuildResult(model=model, variable_roles=roles)
    roles["concentration"] = "changed"
    assert result.output_contract is result.output_contract
    assert result.output_contract.variable_roles == {"concentration": "y"}
