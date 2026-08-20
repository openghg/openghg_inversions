"""Tests for labelled active/fixed state-vector model construction."""

from __future__ import annotations

from typing import Any, cast

import dask.array as da
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.basis import project_basis_prior_stdev
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.models import (
    CoordRegistry,
    StateActivity,
    active_prior_args,
    attach_coord_registry,
    detect_zero_sensitivity,
    prepare_linear_design,
    registered_model,
    resolve_state_activity,
)
from openghg_inversions.models.components import add_linear_component, resolve_model_variable
from openghg_inversions.models.components import add_state_vector
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.multisector import (
    _prepare_multisector_flux_components,
    build_multisector_rhime_model as _build_multisector_model,
)
from openghg_inversions.rhime.specs import SectorSpec
from openghg_inversions.rhime.standard import build_standard_rhime_model as _build_standard_model
from openghg_inversions.sigma import SigmaAlignment


def _sensitivity() -> xr.DataArray:
    """Return labelled sensitivity with exact-zero and near-zero columns."""
    return xr.DataArray(
        [
            [1.0, 0.0, 2.0, 1.0e-12],
            [3.0, 0.0, 4.0, -1.0e-12],
        ],
        dims=("nmeasure", "region"),
        coords={
            "nmeasure": [0, 1],
            "region": ["inner-a", "zero", "outer-a", "inner-b"],
            "basis_group": ("region", ["inner", "inner", "outer", "inner"]),
        },
        name="H",
    )


def _model_inputs(h: xr.DataArray) -> xr.Dataset:
    """Return the minimal complete dataset needed by RHIME model builders."""
    nmeasure = h.sizes["nmeasure"]
    return xr.Dataset(
        {
            "H": h,
            "mf": ("nmeasure", np.full(nmeasure, 2.0)),
            "mf_error": ("nmeasure", np.full(nmeasure, 0.1)),
            "min_error": ("nmeasure", np.zeros(nmeasure)),
            "site_indicator": ("nmeasure", np.zeros(nmeasure, dtype=int)),
            "sigma_freq_index": ("nmeasure", np.zeros(nmeasure, dtype=int)),
        }
    )


def _sigma_alignment(inputs: xr.Dataset) -> SigmaAlignment:
    """Return prepared sigma alignment for the minimal builder inputs."""
    return SigmaAlignment.from_indices(inputs["site_indicator"], inputs["sigma_freq_index"])


def build_rhime_model(inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
    """Adapt test datasets to the standard builder's named-array contract."""
    return _build_standard_model(
        inputs["H"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        minimum_error=inputs["min_error"],
        aggregation_error=resolve_aggregation_error(inputs, "none"),
        boundary_sensitivity=inputs.get("H_bc"),
        site_indicator=inputs.get("site_indicator"),
        **kwargs,
    )


def build_rhime_multisector_model(inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
    """Adapt test datasets to the multisector builder's named-array contract."""
    return _build_multisector_model(
        inputs["H"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        minimum_error=inputs["min_error"],
        aggregation_error=resolve_aggregation_error(inputs, "none"),
        boundary_sensitivity=inputs.get("H_bc"),
        site_indicator=inputs.get("site_indicator"),
        **kwargs,
    )


def test_detect_zero_sensitivity_validates_and_retains_state_metadata() -> None:
    """Reduce the design to a labelled mask without dropping group metadata."""
    zero_sensitivity = detect_zero_sensitivity(_sensitivity())

    np.testing.assert_array_equal(zero_sensitivity, [False, True, False, False])
    assert zero_sensitivity.dims == ("region",)
    np.testing.assert_array_equal(zero_sensitivity["basis_group"], ["inner", "inner", "outer", "inner"])


def test_detect_zero_sensitivity_requires_a_two_dimensional_output_design() -> None:
    """Reject extra axes and designs lacking the declared output dimension."""
    with pytest.raises(ValueError, match="two-dimensional"):
        detect_zero_sensitivity(_sensitivity().expand_dims(extra=[0]))
    with pytest.raises(ValueError, match="output dimension 'nmeasure'"):
        detect_zero_sensitivity(_sensitivity().rename(nmeasure="observation"))


def test_detect_zero_sensitivity_accepts_a_named_output_dimension() -> None:
    """Detect zero columns when the caller declares a non-default output axis."""
    design = _sensitivity().rename(nmeasure="observation")

    zero_sensitivity = detect_zero_sensitivity(design, output_dim="observation")

    np.testing.assert_array_equal(zero_sensitivity, [False, True, False, False])
    assert zero_sensitivity.dims == ("region",)


def test_prepare_linear_design_removes_columns_and_retains_full_mapping() -> None:
    """Structural removal keeps a lossless labelled full-to-retained mapping."""
    prepared = prepare_linear_design(_sensitivity())

    assert prepared.design.dims == ("nmeasure", "region_retained")
    np.testing.assert_array_equal(prepared.design["region_retained"], ["inner-a", "outer-a", "inner-b"])
    np.testing.assert_array_equal(prepared.removed, [False, True, False, False])
    np.testing.assert_array_equal(prepared.removed["basis_group"], ["inner", "inner", "outer", "inner"])
    np.testing.assert_array_equal(prepared.retained_indices, [0, 2, 3])


def test_prepare_linear_design_keeps_retained_dask_payload_lazy() -> None:
    """Preparation computes the structural mask without densifying retained data."""
    prepared = prepare_linear_design(_sensitivity().chunk({"nmeasure": 1, "region": 2}))

    assert isinstance(prepared.design.data, da.Array)
    assert not isinstance(prepared.removed.data, da.Array)


def test_all_zero_linear_design_builds_zero_forward_and_full_fixed_state() -> None:
    """An all-zero design has no latent variables and reconstructs the full state."""
    h = xr.zeros_like(_sensitivity())
    prepared = prepare_linear_design(h)

    with registered_model() as model:
        result = add_linear_component(
            prepared,
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            var_name="x",
            output_name="mu",
        )

    assert prepared.design.sizes["region_retained"] == 0
    assert model.free_RVs == []
    np.testing.assert_allclose(model.named_vars["x"].eval(), np.ones(4))
    np.testing.assert_allclose(result.output.eval(), np.zeros(2))


def test_resolve_state_activity_combines_labels_groups_and_exact_zero() -> None:
    """Resolve masks by labels while retaining nonzero values of any magnitude."""
    h = _sensitivity()
    explicit = xr.DataArray(
        [True, False, True, True],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    resolved = resolve_state_activity(
        detect_zero_sensitivity(h),
        StateActivity(active=explicit, fixed_groups=("outer",)),
    )

    np.testing.assert_array_equal(resolved.zero_sensitivity, [False, True, False, False])
    np.testing.assert_array_equal(resolved.active, [True, False, False, True])
    assert resolved.n_active == 2
    np.testing.assert_array_equal(resolved.active_indices, [0, 3])


def test_active_prior_args_aligns_labelled_and_array_parameters() -> None:
    """Subset labelled and positional full-state prior arrays to active order."""
    h = _sensitivity()
    resolved = resolve_state_activity(
        detect_zero_sensitivity(h),
        StateActivity(fixed_groups=("outer",)),
    )
    labelled_mu = xr.DataArray(
        [40.0, 30.0, 20.0, 10.0],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    prior = active_prior_args(
        {
            "pdf": "normal",
            "mu": labelled_mu,
            "sigma": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        resolved,
    )

    np.testing.assert_array_equal(prior["mu"], [10.0, 40.0])
    np.testing.assert_array_equal(prior["sigma"], [1.0, 4.0])


def test_rhime_model_accepts_projected_labelled_prior_stdev() -> None:
    """The labelled basis projection passes directly through the model prior API."""
    basis = xr.DataArray(
        [[1, 1, 2]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0]},
    )
    flux = xr.DataArray(
        [[1.0, 1.0, 2.0]],
        dims=("lat", "lon"),
        coords=basis.coords,
    )
    basis_functions = BasisFunctions.from_flat_basis(
        basis,
        flux,
        operator_kwargs={"state_dim": "region"},
    )
    projected = project_basis_prior_stdev(
        basis_functions,
        area_grid=xr.ones_like(flux),
        grid_cell_prior_stdev=0.4,
    )
    sensitivity = xr.DataArray(
        [[1.0, 0.0], [2.0, 0.0]],
        dims=("nmeasure", "region"),
        coords={"nmeasure": [0, 1], "region": projected["region"]},
        name="H",
    )

    inputs = _model_inputs(sensitivity)
    model = build_rhime_model(
        inputs,
        sigma_alignment=_sigma_alignment(inputs),
        x_prior={"pdf": "normal", "mu": 1.0, "sigma": projected},
        use_bc=False,
        no_model_error=True,
    )

    assert model.named_vars["x_active"].eval().shape == (1,)
    assert model.named_vars["x_active"] in model.free_RVs


def test_state_activity_materializes_dask_backed_reductions_and_vectors() -> None:
    """Materialize lazy detection, policy resolution, and prior slicing."""
    h = _sensitivity().chunk({"nmeasure": 1, "region": 2})
    active = xr.DataArray(
        da.from_array([True, True, False, True], chunks="auto"),
        dims="region",
        coords={"region": h.region},
    )
    fixed = xr.DataArray(
        da.from_array([1.0, 2.0, 3.0, 4.0], chunks="auto"),
        dims="region",
        coords={"region": h.region},
    )
    resolved = resolve_state_activity(
        detect_zero_sensitivity(h),
        StateActivity(active=active, fixed_value=fixed),
    )
    lazy_mu = xr.DataArray(
        da.from_array([10.0, 20.0, 30.0, 40.0], chunks="auto"),
        dims="region",
        coords={"region": h.region},
    )

    prior = active_prior_args(
        {"pdf": "normal", "mu": lazy_mu, "sigma": xr.DataArray(0.5)},
        resolved,
    )

    assert resolved.n_active == 2
    np.testing.assert_array_equal(resolved.active_indices, [0, 3])
    np.testing.assert_array_equal(prior["mu"], [10.0, 40.0])
    assert prior["sigma"] == 0.5


@pytest.mark.parametrize("labels", [["inner-a", "zero", "outer-a", "outer-a"], None])
def test_detect_zero_sensitivity_rejects_invalid_canonical_labels(labels: list[str] | None) -> None:
    """Require a present and unique canonical state coordinate during detection."""
    h = _sensitivity()
    if labels is None:
        h = h.drop_indexes("region").drop_vars("region")
    else:
        h = h.assign_coords(region=labels)

    with pytest.raises(ValueError, match="labelled|unique"):
        detect_zero_sensitivity(h)


@pytest.mark.parametrize(
    "labels",
    [
        ["inner-a", "zero", "outer-a", "outer-a"],
        ["inner-a", "zero", "outer-a", "unexpected"],
    ],
)
def test_resolve_state_activity_rejects_duplicate_or_misaligned_policy_labels(
    labels: list[str],
) -> None:
    """Reject duplicate, missing, or extra labels on state-aligned policy values."""
    active = xr.DataArray([True] * 4, dims="region", coords={"region": labels})

    with pytest.raises(ValueError, match="unique|match the canonical"):
        resolve_state_activity(
            detect_zero_sensitivity(_sensitivity()),
            StateActivity(active=active),
        )


@pytest.mark.parametrize(
    "active",
    [1, 0.0, np.nan, "False", np.array([1, 0, 1, 0])],
)
def test_resolve_state_activity_rejects_non_boolean_active_values(
    active: object,
) -> None:
    """Do not silently coerce numbers, missing values, or strings to activity."""
    with pytest.raises(ValueError, match="active.*boolean"):
        resolve_state_activity(
            detect_zero_sensitivity(_sensitivity()),
            StateActivity(active=cast(Any, active)),
        )


def test_resolve_state_activity_requires_a_boolean_zero_mask() -> None:
    """Reject numeric inputs at the resolved-policy boundary."""
    zero_sensitivity = detect_zero_sensitivity(_sensitivity()).astype(int)

    with pytest.raises(ValueError, match="zero_sensitivity.*boolean"):
        resolve_state_activity(zero_sensitivity)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_resolve_state_activity_rejects_nonfinite_sensitivity_and_fixed_values(
    bad_value: float,
) -> None:
    """Reject every non-finite sensitivity or fixed-state value."""
    h = _sensitivity().copy()
    h[0, 0] = bad_value
    with pytest.raises(ValueError, match="Sensitivity.*finite"):
        detect_zero_sensitivity(h)

    fixed = np.ones(h.sizes["region"])
    fixed[1] = bad_value
    with pytest.raises(ValueError, match="fixed_value.*finite"):
        resolve_state_activity(
            detect_zero_sensitivity(_sensitivity()),
            StateActivity(fixed_value=fixed),
        )


@pytest.mark.parametrize("prior_kind", ["numpy", "xarray"])
def test_active_prior_args_rejects_nonfinite_full_state_arrays(prior_kind: str) -> None:
    """Reject non-finite positional and labelled full-state prior parameters."""
    values = np.array([1.0, np.nan, 3.0, 4.0])
    value: np.ndarray | xr.DataArray
    if prior_kind == "xarray":
        value = xr.DataArray(values, dims="region", coords={"region": _sensitivity().region})
    else:
        value = values
    resolved = resolve_state_activity(detect_zero_sensitivity(_sensitivity()))

    with pytest.raises(ValueError, match="Prior parameter 'mu'.*finite"):
        active_prior_args({"pdf": "normal", "mu": value}, resolved)


def test_state_linear_component_preserves_full_forward_identity() -> None:
    """Full H/state multiplication equals active plus fixed contributions."""
    h = _sensitivity()
    fixed_value = xr.DataArray(
        [4.0, 3.0, 2.0, 1.0],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    registry = CoordRegistry()
    with pm.Model() as model:
        attach_coord_registry(model, registry)
        result = add_linear_component(
            prepare_linear_design(h),
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 2.0, "sigma": 0.1},
            var_name="x",
            output_name="mu",
            state_activity=StateActivity(
                fixed_value=fixed_value,
                fixed_groups=("outer",),
            ),
        )

    activity = resolve_state_activity(detect_zero_sensitivity(h), StateActivity(
        fixed_value=fixed_value,
        fixed_groups=("outer",),
    ))
    x_full, mu, x_active = pm.draw(
        [result.state, result.output, model.named_vars["x_active"]],
        random_seed=42,
    )
    active = activity.active_indices
    fixed = activity.fixed_indices
    expected_split = (
        h.values[:, active] @ x_active + h.values[:, fixed] @ fixed_value.sel(region=h.region).values[fixed]
    )

    np.testing.assert_allclose(h.values @ x_full, mu)
    np.testing.assert_allclose(expected_split, mu)
    np.testing.assert_array_equal(model.named_vars["x_is_active"].eval(), [True, False, False, True])
    assert model.named_vars["x"].name == "x"
    assert model.named_vars["x_active"] in model.free_RVs
    assert list(registry.original_coords["region_x_active"]) == ["inner-a", "inner-b"]


def test_add_state_vector_registers_full_state_coord_in_a_fresh_model() -> None:
    """Construct a state graph from a resolved contract and register its coordinate."""
    activity = resolve_state_activity(
        xr.zeros_like(detect_zero_sensitivity(_sensitivity()), dtype=bool),
    )
    with registered_model() as model:
        result = add_state_vector(
            activity,
            prior_args={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
            var_name="x",
        )

    assert result.state in model.free_RVs
    assert model.coords["region"] == (0, 1, 2, 3)


def test_linear_component_preserves_plain_graph_when_all_states_are_retained() -> None:
    """Use the ordinary base prior graph when preparation removes nothing."""
    h = _sensitivity().drop_sel(region="zero")
    prior = {"pdf": "normal", "mu": 1.0, "sigma": 0.2}

    with pm.Model() as linear_model:
        attach_coord_registry(linear_model, CoordRegistry())
        result = add_linear_component(
            prepare_linear_design(h),
            data_name="hx",
            prior_args=prior,
            var_name="x",
            output_name="mu",
        )
    assert set(linear_model.named_vars) == {"hx", "x", "mu"}
    assert [rv.name for rv in linear_model.free_RVs] == ["x"]
    assert result.state is linear_model.named_vars["x"]
    assert result.output is linear_model.named_vars["mu"]


def test_state_linear_component_full_activity_preserves_reparameterised_names() -> None:
    """Keep the legacy base and latent names for a fully active lognormal state."""
    h = _sensitivity().drop_sel(region="zero")
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            prepare_linear_design(h),
            data_name="hx",
            prior_args={
                "pdf": "lognormal",
                "mean": 1.0,
                "stdev": 0.2,
                "reparameterise": True,
            },
            var_name="x",
            output_name="mu",
        )

    assert set(model.named_vars) == {"hx", "x_latent", "x", "mu"}
    assert [rv.name for rv in model.free_RVs] == ["x_latent"]
    assert result.output is model.named_vars["mu"]
    assert result.latent is model.named_vars["x_latent"]


def test_state_linear_component_restores_removed_states_in_canonical_order() -> None:
    """Restoring active and fixed partitions reproduces the full state and output."""
    h = _sensitivity()
    fixed = xr.DataArray(
        [11.0, 12.0, 13.0, 14.0],
        dims="region",
        coords={"region": h.region},
    )
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            prepare_linear_design(h),
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 2.0, "sigma": 0.1},
            var_name="x",
            output_name="mu",
            state_activity=StateActivity(
                active=np.array([True, False, True, False]),
                fixed_value=fixed,
            ),
        )

    full_state, active_state, output = pm.draw(
        [result.state, model.named_vars["x_active"], result.output],
        random_seed=42,
    )
    restored = fixed.to_numpy().copy()
    activity = resolve_state_activity(detect_zero_sensitivity(h), StateActivity(
        active=np.array([True, False, True, False]),
        fixed_value=fixed,
    ))
    restored[activity.active_indices] = active_state

    np.testing.assert_allclose(full_state, restored)
    np.testing.assert_allclose(output, h.to_numpy() @ restored)


def test_state_linear_component_supports_zero_active_states() -> None:
    """An all-fixed component creates no active prior and retains its full state."""
    h = _sensitivity()
    policy = StateActivity(active=False, fixed_value=2.5)

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            prepare_linear_design(h),
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            var_name="x",
            output_name="mu",
            state_activity=policy,
        )

    x_full, mu = pm.draw([result.state, result.output], random_seed=42)
    np.testing.assert_allclose(x_full, np.full(h.sizes["region"], 2.5))
    np.testing.assert_allclose(mu, h.values @ x_full)
    assert "x_active" not in model.named_vars
    assert not any(rv.name and rv.name.startswith("x") for rv in model.free_RVs)
    assert resolve_model_variable(model, "x") is model.named_vars["x"]

def test_multisector_rhime_can_freeze_a_sector_and_use_array_priors() -> None:
    """Per-sector policies can freeze all states while other sectors sample arrays."""
    h = _sensitivity()
    multi_h = xr.concat(
        [h.expand_dims(source=["ff-source"]), (2.0 * h).expand_dims(source=["ocean-source"])],
        dim="source",
    )
    inputs = _model_inputs(multi_h)
    ocean_mu = xr.DataArray(
        [1.4, 1.3, 1.2, 1.1],
        dims="region",
        coords={"region": ["inner-b", "outer-a", "zero", "inner-a"]},
    )

    model = build_rhime_multisector_model(
        inputs,
        sigma_alignment=_sigma_alignment(inputs),
        sectors=(
            SectorSpec(
                name="FF",
                flux_source="ff-source",
                x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                variable_suffix="ff",
                state_activity=StateActivity(active=False),
            ),
            SectorSpec(
                name="ocean",
                flux_source="ocean-source",
                x_prior={"pdf": "normal", "mu": ocean_mu, "sigma": 0.3},
                variable_suffix="ocean",
            ),
        ),
        use_bc=False,
        no_model_error=True,
    )

    assert "x_ff_active" not in model.named_vars
    assert "x_ocean_active" in model.named_vars
    np.testing.assert_allclose(model.named_vars["x_ff"].eval(), np.ones(4))
    assert model.named_vars["x_ocean"].eval().shape == (4,)
    assert model.named_vars["mu_ff"].eval().shape == (2,)
    assert model.named_vars["mu_ocean"].eval().shape == (2,)
    assert "mu" not in model.named_vars


def test_multisector_flux_preparation_applies_sector_override_and_shared_activity() -> None:
    """Flux preparation applies an override and the shared fallback directly."""
    h = _sensitivity()
    multi_h = xr.concat(
        [h.expand_dims(source=["ff-source"]), (2.0 * h).expand_dims(source=["ocean-source"])],
        dim="source",
    )
    sectors = (
        SectorSpec(
            name="FF",
            flux_source="ff-source",
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.2},
            variable_suffix="ff",
            state_activity=StateActivity(active=False, fixed_value=2.0),
        ),
        SectorSpec(
            name="ocean",
            flux_source="ocean-source",
            x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.3},
            variable_suffix="ocean",
        ),
    )

    components = _prepare_multisector_flux_components(
        multi_h,
        sectors,
        state_activity=StateActivity(active=False, fixed_value=3.0),
    )

    ff_sector, _, _, ff_activity = components[0]
    ocean_sector, _, _, ocean_activity = components[1]
    assert ff_sector is sectors[0]
    assert ocean_sector is sectors[1]
    assert not ff_activity.active.any()
    assert not ocean_activity.active.any()
    np.testing.assert_allclose(ff_activity.fixed_value, 2.0)
    np.testing.assert_allclose(ocean_activity.fixed_value, 3.0)
