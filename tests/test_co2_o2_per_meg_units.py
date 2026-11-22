"""CO2 ppm with declared per-meg delta(O2/N2) O2 on the configured, handoff and output paths."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from openghg_inversions.inversion_data.o2n2 import (
    O2_OBSERVABLE_PROVENANCE_KEY,
    DeltaO2N2Declaration,
    is_per_meg_unit,
    is_ratio_anomaly_unit,
)
from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs, prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_configuration import (
    Co2O2ModelConfig,
    validate_co2_o2_channel_units,
    validate_co2_o2_units,
)
from openghg_inversions.rhime.co2.co2_o2_outputs import Co2O2OutputOptions

from examples.co2_o2_handoffs import FACTS
from test_co2_o2_handoffs import constructed_joint_trace, no_inference  # noqa: F401
from test_rhime_co2_o2 import _independent_error, _inputs

DECLARATION = DeltaO2N2Declaration(
    reference_ratio=0.20946 / 0.78084,
    reference_ratio_source="CORSO TM5 O2 and N2 curtains",
    n2_policy="curtain",
)


def _declared(provenance: dict | None = None):
    """Linked inputs with ppm CO2 and per-meg O2 rows that declare delta(O2/N2)."""
    inputs = _inputs()
    assert inputs["o2_units"] == "per meg"
    for channel, count in (("co2", 2), ("o2", 3)):
        observed = inputs[f"{channel}_observations"]
        inputs[f"{channel}_observations"] = observed.assign_coords(
            site=(observed.dims[0], ["HFD"] * count),
            time=(
                observed.dims[0],
                np.datetime64("2021-07-01T00") + np.arange(count).astype("timedelta64[h]"),
            ),
        )
    if provenance is None:
        provenance = {**FACTS, O2_OBSERVABLE_PROVENANCE_KEY: DECLARATION.to_provenance()}
    inputs["provenance"] = provenance
    prepared = prepare_co2_o2_inputs(**inputs)
    return replace(prepared, independent_error_sd=_independent_error(prepared))


def _config(*, observable: str | None = "delta_o2_n2", o2_units: str = "per meg") -> dict:
    o2 = {"units": o2_units, "independent_error_sd": 5.0}
    if observable is not None:
        o2["observable"] = observable
    return {
        "format_version": 1,
        "recipe": "co2_o2",
        "channels": {"co2": {"units": "ppm", "independent_error_sd": 0.5}, "o2": o2},
    }


def test_declaration_round_trips_and_rejects_invalid_records():
    record = DECLARATION.to_provenance()
    assert record["kind"] == "delta_o2_n2"
    assert DeltaO2N2Declaration.from_provenance({O2_OBSERVABLE_PROVENANCE_KEY: record}) == DECLARATION
    assert DeltaO2N2Declaration.from_provenance({}) is None
    assert (
        DeltaO2N2Declaration.from_provenance({O2_OBSERVABLE_PROVENANCE_KEY: {"kind": "mole_fraction"}})
        is None
    )
    for bad in (
        {**record, "reference_ratio": -1.0},
        {**record, "reference_ratio": True},
        {**record, "reference_ratio_source": " "},
        {**record, "units": "ppm"},
        {**record, "kind": "apo"},
        {**record, "extra": 1},
        {"kind": "delta_o2_n2", "units": "per meg"},
    ):
        with pytest.raises(ValueError):
            DeltaO2N2Declaration.from_provenance({O2_OBSERVABLE_PROVENANCE_KEY: bad})
    assert all(is_per_meg_unit(unit) for unit in ("per meg", "per_meg", "permeg", "Per-Meg"))
    assert not is_per_meg_unit("ppm")
    assert all(is_ratio_anomaly_unit(unit) for unit in ("per meg", "per mil", "permille", "\u2030", "delta"))
    assert not any(is_ratio_anomaly_unit(unit) for unit in ("ppm", "ppb", "mol/mol", "micromol/mol"))


def test_declared_per_meg_channel_units_are_accepted():
    assert validate_co2_o2_channel_units(_declared()) == {"co2": "ppm", "o2": "per meg"}
    with pytest.raises(ValueError, match="identical CO2 and O2 mole-fraction"):
        validate_co2_o2_units(_declared())


def test_per_meg_without_declaration_or_matching_units_is_rejected():
    with pytest.raises(ValueError, match="declaration"):
        validate_co2_o2_channel_units(_declared(provenance=dict(FACTS)))
    undeclared = _declared(provenance=dict(FACTS))
    per_mil = undeclared.observations.assign_coords(
        observation_units=("observation", ["ppm"] * 2 + ["per mil"] * 3)
    )
    with pytest.raises(ValueError, match="declaration"):
        validate_co2_o2_channel_units(replace(undeclared, observations=per_mil))
    declared = _declared()
    relabelled = declared.observations.assign_coords(
        observation_units=("observation", ["ppm"] * 2 + ["permeg"] * 3)
    )
    with pytest.raises(ValueError, match="declared delta"):
        validate_co2_o2_channel_units(replace(declared, observations=relabelled))
    relabelled = declared.observations.assign_coords(
        observation_units=("observation", ["per meg"] * 2 + ["per meg"] * 3)
    )
    with pytest.raises(ValueError, match="CO2 observation_units must be a mole fraction"):
        validate_co2_o2_channel_units(replace(declared, observations=relabelled))


def test_configured_per_meg_binds_channel_errors_and_matches_dense_likelihood():
    from scipy.stats import multivariate_normal

    prepared = replace(_declared(), independent_error_sd=None)
    model_config = Co2O2ModelConfig.from_params(_config())
    assert model_config.o2_observable == "delta_o2_n2"
    arguments = model_config.bind(prepared)
    assert "o2_observable" not in arguments
    np.testing.assert_allclose(arguments["independent_error_sd"], [0.5, 0.5, 5.0, 5.0, 5.0])

    built = handoffs.build_rhime_co2_o2(prepared_inputs=prepared, **arguments)
    model = built.model
    point = model.initial_point()
    mean = model.compile_fn(model.replace_rvs_by_values([model["modelled_concentration"]]))(point)[0]
    covariance = prepared.aggregation_error.covariance.values + np.diag(
        arguments["independent_error_sd"].values ** 2
    )
    expected = multivariate_normal.logpdf(prepared.observations.values, mean=mean, cov=covariance)
    np.testing.assert_allclose(model.compile_logp(vars=[model["y"]])(point), expected, rtol=1e-5)


@pytest.mark.parametrize(
    "config,message",
    [
        (_config(observable=None), "mole fraction"),
        (_config(o2_units="ppm"), "per-meg"),
        (
            {
                **_config(),
                "channels": {
                    **_config()["channels"],
                    "co2": {"units": "ppm", "observable": "delta_o2_n2", "independent_error_sd": 0.5},
                },
            },
            "channels.co2.observable",
        ),
        (_config(observable="apo"), "observable must be one of"),
    ],
)
def test_configuration_rejects_undeclared_or_inconsistent_observables(config, message):
    with pytest.raises(ValueError, match=message):
        Co2O2ModelConfig.from_params(config)


def test_configured_observable_must_match_the_artifact():
    undeclared_mole_fraction = Co2O2ModelConfig.from_params(
        _config(observable="mole_fraction", o2_units="ppm")
    )
    with pytest.raises(ValueError, match="do not match prepared"):
        undeclared_mole_fraction.bind(_declared())


@pytest.mark.parametrize("extension", [".nc", ".zarr"])
def test_handoffs_import_save_and_replay_declared_per_meg(tmp_path, extension, no_inference):  # noqa: F811
    prepared = _declared()
    source = tmp_path / f"declared{extension}"
    prepared.save(source)
    imported = handoffs.prepare(prepared_path=source, output_dir=tmp_path / "imported")
    loaded = handoffs.load_preparation(imported).prepared
    assert DeltaO2N2Declaration.from_provenance(loaded.provenance) == DECLARATION

    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prepare")
    trace = constructed_joint_trace(prepared)
    sample = handoffs.save_samples(
        manifest,
        trace,
        actual_inference={"draws": 4, "chains": 1},
        preparation_generation=handoffs.load_preparation(manifest).generation,
        output_dir=tmp_path / "sample",
    )
    result = handoffs.postprocess(
        sample,
        output=Co2O2OutputOptions(
            output_format="basic",
            output_path=tmp_path / "products",
            output_name="per_meg",
        ),
    )
    for species, units, observable in (("co2", "ppm", "mole_fraction"), ("o2", "per meg", "delta_o2_n2")):
        for name in (f"{species}_concentration_components", f"{species}_basic"):
            product = result.outputs[name]
            assert product.attrs["units"] == units
            assert product.attrs["observable"] == observable
    o2 = result.outputs["o2_basic"]
    assert o2.attrs["o2_n2_reference_ratio"] == pytest.approx(0.20946 / 0.78084)
    assert json.loads(o2.attrs["o2_observable_declaration"]) == DECLARATION.to_provenance()
    assert "o2_n2_reference_ratio" not in result.outputs["co2_basic"].attrs
