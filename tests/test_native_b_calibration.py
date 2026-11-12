"""Tests for native prior covariance amplitude calibration."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.native_b_calibration import (
    calibrate_native_b_amplitudes,
    calibrated_source_covariance,
    evaluate_native_b_amplitudes,
)

SOURCES = ["co2_FF", "co2_GPP", "co2_TER", "co2_ocean"]
FUNCTIONALS = ["EUROPE_INNER", "AAA", "BBB"]


def _toy_case() -> dict[str, xr.DataArray]:
    """Return an analytic 6 x 7 grid with land, sea and outer classes."""
    lat = 50.0 + 0.5 * np.arange(6)
    lon = -2.0 + 0.5 * np.arange(7)
    i, j = np.meshgrid(np.arange(6), np.arange(7), indexing="ij")
    inner = (i >= 1) & (i <= 4) & (j >= 1) & (j <= 5)
    sea = inner & ((j == 5) | ((i == 4) & (j == 4)))
    land = inner & ~sea
    coords = {"lat": lat, "lon": lon}
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((6, 7))
    flux = np.stack(
        [
            1.0 + 0.3 * i + 0.2 * j,
            -(2.0 + np.sin(i + 0.5 * j)),
            1.5 + 0.5 * np.cos(0.7 * i - 0.3 * j),
            0.3 + 0.05 * (i - j),
        ]
    )
    masks = np.stack([inner, land & (j <= 2), land & (j >= 3)]).astype(float)
    return {
        "flux": xr.DataArray(flux, dims=("source", "lat", "lon"), coords={"source": SOURCES, **coords}),
        "weights": xr.DataArray(
            masks * area,
            dims=("functional", "lat", "lon"),
            coords={"functional": FUNCTIONALS, **coords},
        ),
        "classes": xr.DataArray(
            np.where(~inner, "outer", np.where(sea, "sea", "land")),
            dims=("lat", "lon"),
            coords=coords,
        ),
    }


def _dense_kernel(case: dict[str, xr.DataArray], length: float, *, blocked: bool) -> np.ndarray:
    """Return the dense unit kernel, optionally class blocked (test oracle only)."""
    lat = case["flux"]["lat"].values
    lon = case["flux"]["lon"].values
    k_lat = np.exp(-np.abs(lat[:, None] - lat[None, :]) / length)
    k_lon = np.exp(-np.abs(lon[:, None] - lon[None, :]) / length)
    kernel = np.kron(k_lat, k_lon)
    if blocked:
        labels = case["classes"].values.reshape(-1)
        kernel = kernel * (labels[:, None] == labels[None, :])
    return kernel


def _dense_unit_sd(case, length, source, functional, *, blocked=True):
    """Return ``sqrt(q.T K q)`` for one source and functional with dense ``K``."""
    q = (case["flux"].sel(source=source) * case["weights"].sel(functional=functional)).values.reshape(-1)
    return float(np.sqrt(q @ _dense_kernel(case, length, blocked=blocked) @ q)), float(q.sum())


def test_single_target_matches_dense_oracle_exactly():
    """One target per source: achieved SD equals the target and dense ``q.T B q``."""
    case = _toy_case()
    target = xr.DataArray(
        [0.10, 0.06, 0.06, 0.50],
        dims="source",
        coords={"source": SOURCES},
    ).expand_dims(functional=["EUROPE_INNER"])
    result = calibrate_native_b_amplitudes(
        case["flux"],
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=target,
        class_labels=case["classes"],
    )
    diagnostics = result.diagnostics.sel(functional="EUROPE_INNER")
    np.testing.assert_allclose(diagnostics["achieved_relative_sd"], [0.10, 0.06, 0.06, 0.50], rtol=1e-12)
    for source, relative in zip(SOURCES, [0.10, 0.06, 0.06, 0.50], strict=True):
        unit_sd, total = _dense_unit_sd(case, 1.5, source, "EUROPE_INNER")
        amplitude = float(result.amplitudes.sel(source=source))
        assert amplitude == pytest.approx(relative * abs(total) / unit_sd, rel=1e-12)
        for functional in FUNCTIONALS:
            unit_sd, _ = _dense_unit_sd(case, 1.5, source, functional)
            achieved = float(result.diagnostics["achieved_sd"].sel(source=source, functional=functional))
            assert achieved == pytest.approx(amplitude * unit_sd, rel=1e-12)
    assert set(result.diagnostics["status"].sel(functional=["AAA", "BBB"]).values.ravel()) == {"diagnostic"}
    assert result.diagnostics["used_for_calibration"].sum() == 4


def test_absolute_target_without_class_labels():
    """Absolute targets and the unblocked kernel follow the same closed form."""
    case = _toy_case()
    target_sd = xr.DataArray([2.0], dims="functional", coords={"functional": ["AAA"]})
    result = calibrate_native_b_amplitudes(
        case["flux"], case["weights"], correlation_length=2.0, target_sd=target_sd
    )
    for source in SOURCES:
        unit_sd, _ = _dense_unit_sd(case, 2.0, source, "AAA", blocked=False)
        assert float(result.amplitudes.sel(source=source)) == pytest.approx(2.0 / unit_sd, rel=1e-12)
    np.testing.assert_allclose(result.diagnostics["achieved_sd"].sel(functional="AAA"), 2.0, rtol=1e-12)


def test_several_regions_use_least_squares_in_log_sd():
    """Several targets give the geometric-mean ratio and zero mean log residual."""
    case = _toy_case()
    result = calibrate_native_b_amplitudes(
        case["flux"],
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=0.2,
        class_labels=case["classes"],
    )
    for source in SOURCES:
        ratios = []
        for functional in FUNCTIONALS:
            unit_sd, total = _dense_unit_sd(case, 1.5, source, functional)
            ratios.append(0.2 * abs(total) / unit_sd)
        expected = float(np.exp(np.mean(np.log(ratios))))
        assert float(result.amplitudes.sel(source=source)) == pytest.approx(expected, rel=1e-12)
    residual = result.diagnostics["log_sd_residual"].mean("functional")
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        # Frozen from verification-games 13ae5f0
        # native_b_calibration.calibrate_regional_land_b_amplitudes on this
        # toy case: direct-europe-sector, FF/GPP/TER 0.10, ocean 0.50,
        # class-blocked, gpp_ter_native_correlation 0.25 (diagnostic only).
        (1.5, [0.18255624712033772, 0.1726753955010142, 0.17762734407554182, 0.8366444881864152]),
        (3.0, [0.15648069500455236, 0.14811260992072164, 0.15205819557586892, 0.7142240090374251]),
    ],
)
def test_parity_with_verification_games_direct_europe_sector(length, expected):
    """Per-source EUROPE_INNER targets reproduce the VG production calibration."""
    case = _toy_case()
    target = xr.DataArray(
        [[0.10, 0.10, 0.10, 0.50]],
        dims=("functional", "source"),
        coords={"functional": ["EUROPE_INNER"], "source": SOURCES},
    )
    result = calibrate_native_b_amplitudes(
        case["flux"],
        case["weights"],
        correlation_length=length,
        target_relative_sd=target,
        class_labels=case["classes"],
    )
    np.testing.assert_allclose(result.amplitudes.sel(source=SOURCES), expected, rtol=1e-12)
    # The VG formula, sigma = target * |total| / sqrt(q.T K_blocked q), reimplemented.
    for source, relative, frozen in zip(SOURCES, [0.10, 0.10, 0.10, 0.50], expected, strict=True):
        unit_sd, total = _dense_unit_sd(case, length, source, "EUROPE_INNER")
        assert relative * abs(total) / unit_sd == pytest.approx(frozen, rel=1e-12)


def test_correlation_length_changes_amplitude_as_dense_kernel_predicts():
    """Longer correlation raises the unit SD of a same-signed total, lowering ``a``."""
    case = _toy_case()
    lengths = (0.25, 1.5, 3.0, 6.0)
    amplitudes = [
        calibrate_native_b_amplitudes(
            case["flux"].sel(source=["co2_FF"]),
            case["weights"],
            correlation_length=length,
            target_relative_sd=0.1,
            class_labels=case["classes"],
        )
        .amplitudes.squeeze()
        .item()
        for length in lengths
    ]
    assert np.all(np.diff(amplitudes) < 0.0)
    # With a fixed target the amplitude scales as the inverse unit SD, so
    # a(L1) / a(L2) is the geometric mean over regions of SD_L2 / SD_L1.
    for first, second in ((0, 1), (1, 2), (2, 3)):
        expected = np.exp(
            np.mean(
                [
                    np.log(
                        _dense_unit_sd(case, lengths[second], "co2_FF", functional)[0]
                        / _dense_unit_sd(case, lengths[first], "co2_FF", functional)[0]
                    )
                    for functional in FUNCTIONALS
                ]
            )
        )
        assert amplitudes[first] / amplitudes[second] == pytest.approx(expected, rel=1e-12)


def test_per_month_calibration_is_independent_and_frozen_policy_is_expressible():
    """Each month is calibrated alone; a frozen month x factor is evaluated on all."""
    case = _toy_case()
    january = case["flux"]
    july = case["flux"] * (1.0 + 0.5 * case["flux"]["lon"])
    time = xr.DataArray(np.array(["2021-01-01", "2021-07-01"], dtype="datetime64[ns]"), dims="time")
    flux = xr.concat([january, july], dim=time)
    target = xr.DataArray([0.1], dims="functional", coords={"functional": ["EUROPE_INNER"]})
    kwargs = {"correlation_length": 1.5, "class_labels": case["classes"]}
    both = calibrate_native_b_amplitudes(flux, case["weights"], target_relative_sd=target, **kwargs)
    assert both.amplitudes.dims == ("source", "time")
    for month, single in enumerate((january, july)):
        alone = calibrate_native_b_amplitudes(single, case["weights"], target_relative_sd=target, **kwargs)
        np.testing.assert_allclose(both.amplitudes.isel(time=month), alone.amplitudes, rtol=1e-12)
    assert not np.allclose(both.amplitudes.isel(time=0), both.amplitudes.isel(time=1))

    frozen = both.amplitudes.isel(time=0, drop=True) * 1.7
    diagnostics = evaluate_native_b_amplitudes(
        flux, case["weights"], frozen, target_relative_sd=target, **kwargs
    )
    achieved = diagnostics["achieved_relative_sd"].sel(functional="EUROPE_INNER")
    np.testing.assert_allclose(achieved.isel(time=0), 0.17, rtol=1e-12)
    assert diagnostics["status"].sel(functional="EUROPE_INNER").isin(["target"]).all()
    assert not np.allclose(achieved.isel(time=1), 0.17)


def test_two_slice_dimensions_keep_their_labels():
    """Slice dimensions beyond time are reassembled with the right labels."""
    case = _toy_case()
    flux = case["flux"].expand_dims(member=["a", "b"], time=[1, 2, 3]).copy()
    flux.loc[{"member": "b", "time": 3}] = flux.sel(member="b", time=3) * (2.0 + flux["lat"] - 50.0)
    result = calibrate_native_b_amplitudes(
        flux,
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=0.1,
        class_labels=case["classes"],
    )
    alone = calibrate_native_b_amplitudes(
        flux.sel(member="b", time=3),
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=0.1,
        class_labels=case["classes"],
    )
    np.testing.assert_allclose(result.amplitudes.sel(member="b", time=3), alone.amplitudes, rtol=1e-12)
    np.testing.assert_allclose(
        result.diagnostics["achieved_sd"].sel(member="b", time=3),
        alone.diagnostics["achieved_sd"],
        rtol=1e-12,
    )


def test_per_class_amplitudes_match_class_targets():
    """Targets that separate land and sea identify one amplitude per class."""
    case = _toy_case()
    inner = case["weights"].sel(functional="EUROPE_INNER", drop=True)
    area = xr.ones_like(inner) * np.cos(np.deg2rad(inner["lat"]))
    land = inner.where(case["classes"] == "land", 0.0)
    sea = inner.where(case["classes"] == "sea", 0.0)
    outer = area.where(case["classes"] == "outer", 0.0)
    weights = xr.concat(
        [land, sea, outer, inner],
        dim=xr.DataArray(["LAND", "SEA", "OUTER", "INNER"], dims="functional"),
    )
    flux = case["flux"].sel(source=["co2_FF"])
    target = xr.DataArray(
        [0.10, 0.40, 0.50],
        dims="functional",
        coords={"functional": ["LAND", "SEA", "OUTER"]},
    )
    result = calibrate_native_b_amplitudes(
        flux,
        weights,
        correlation_length=1.5,
        target_relative_sd=target,
        class_labels=case["classes"],
        per_class=True,
    )
    assert result.amplitudes.dims == ("source", "class")
    np.testing.assert_allclose(
        result.diagnostics["achieved_relative_sd"].sel(functional=["LAND", "SEA", "OUTER"]).squeeze(),
        [0.10, 0.40, 0.50],
        rtol=1e-10,
    )
    # Dense oracle for the diagnostic INNER total, which mixes both classes.
    kernel = _dense_kernel(case, 1.5, blocked=True)
    a = result.amplitudes.squeeze("source")
    labels = case["classes"].values.reshape(-1)
    scale = np.array([float(a.sel({"class": label})) for label in labels])
    q = (flux.isel(source=0) * inner).values.reshape(-1)
    dense = float(np.sqrt((scale * q) @ kernel @ (scale * q)))
    achieved = float(result.diagnostics["achieved_sd"].sel(functional="INNER").squeeze())
    assert achieved == pytest.approx(dense, rel=1e-10)

    with pytest.raises(ValueError, match="not identified"):
        calibrate_native_b_amplitudes(
            flux,
            weights,
            correlation_length=1.5,
            target_relative_sd=target.sel(functional=["LAND", "SEA"]),
            class_labels=case["classes"],
            per_class=True,
        )


def test_calibrated_source_covariance_reproduces_achieved_sd():
    """The configuration hook builds a covariance whose ``q.T B q`` matches."""
    case = _toy_case()
    result = calibrate_native_b_amplitudes(
        case["flux"],
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=0.1,
        class_labels=case["classes"],
    )
    covariance = calibrated_source_covariance(
        result.amplitudes,
        latitude=case["flux"]["lat"],
        longitude=case["flux"]["lon"],
        correlation_length=1.5,
        class_labels=case["classes"],
    )
    assert covariance.source_labels == tuple(SOURCES)
    q = case["flux"] * case["weights"]
    variance = (q * covariance.apply(q)).sum(("lat", "lon"))
    np.testing.assert_allclose(np.sqrt(variance), result.diagnostics["achieved_sd"], rtol=1e-12)


def test_invalid_inputs_are_rejected_and_near_zero_totals_reported():
    """Ambiguous or misspelt targets fail; cancelling totals are flagged."""
    case = _toy_case()
    kwargs = {"correlation_length": 1.5}
    with pytest.raises(ValueError, match="exactly one"):
        calibrate_native_b_amplitudes(case["flux"], case["weights"], **kwargs)
    with pytest.raises(ValueError, match="unknown 'source' labels"):
        calibrate_native_b_amplitudes(
            case["flux"],
            case["weights"],
            target_relative_sd=xr.DataArray([0.1], dims="source", coords={"source": ["co2_XX"]}),
            **kwargs,
        )
    with pytest.raises(ValueError, match="requires class_labels"):
        calibrate_native_b_amplitudes(
            case["flux"], case["weights"], target_relative_sd=0.1, per_class=True, **kwargs
        )

    lon = case["flux"]["lon"]
    signed = case["flux"].copy()
    signed.loc[{"source": "co2_ocean"}] = (lon - lon.mean()).broadcast_like(signed.sel(source="co2_ocean"))
    weights = xr.ones_like(case["weights"].isel(functional=[0]))
    target = xr.DataArray([0.1, 0.1, 0.1, 0.5], dims="source", coords={"source": SOURCES})
    with pytest.raises(ValueError, match="No usable target"):
        calibrate_native_b_amplitudes(signed, weights, target_relative_sd=target, **kwargs)
    diagnostics = evaluate_native_b_amplitudes(
        signed,
        weights,
        xr.ones_like(target),
        target_relative_sd=target,
        **kwargs,
    )
    assert diagnostics["status"].sel(source="co2_ocean").item() == "near-zero-total"
    assert bool(diagnostics["unit_relative_sd"].sel(source="co2_ocean").isnull())
