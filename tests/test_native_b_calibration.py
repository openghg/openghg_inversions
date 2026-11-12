"""Tests for native prior covariance amplitude calibration."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.native_b_calibration import (
    calibrate_native_b_amplitudes,
    calibrated_source_covariance,
    calibrated_source_covariances,
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
    assert set(diagnostics["status"].values) == {"met"}
    assert set(diagnostics["total_status"].values) == {"ok"}
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

    assert both.calibration_slices == {"time": tuple(time.values)}

    # The frozen January x 1.7 policy needs an explicit opt-in.
    frozen = both.amplitudes.isel(time=0) * 1.7
    with pytest.raises(ValueError, match="allow_broadcast"):
        evaluate_native_b_amplitudes(flux, case["weights"], frozen, target_relative_sd=target, **kwargs)
    with pytest.raises(ValueError, match="allow_broadcast"):
        evaluate_native_b_amplitudes(
            flux.isel(time=[1]), case["weights"], frozen, target_relative_sd=target, **kwargs
        )
    with pytest.raises(ValueError, match="unrecorded slice"):
        evaluate_native_b_amplitudes(
            flux, case["weights"], frozen.drop_vars("time"), target_relative_sd=target, **kwargs
        )
    same_month = evaluate_native_b_amplitudes(
        flux.isel(time=[0]), case["weights"], frozen, target_relative_sd=target, **kwargs
    )
    np.testing.assert_allclose(same_month["achieved_relative_sd"].sel(functional="EUROPE_INNER"), 0.17)
    diagnostics = evaluate_native_b_amplitudes(
        flux, case["weights"], frozen, target_relative_sd=target, allow_broadcast=True, **kwargs
    )
    achieved = diagnostics["achieved_relative_sd"].sel(functional="EUROPE_INNER")
    np.testing.assert_allclose(achieved.isel(time=0), 0.17, rtol=1e-12)
    assert set(diagnostics["status"].sel(functional="EUROPE_INNER").values.ravel()) == {"unmet"}
    assert not np.allclose(achieved.isel(time=1), 0.17)

    # Per-month amplitudes are directly usable: one covariance per month.
    covariances = calibrated_source_covariances(
        both.amplitudes,
        slice_dim="time",
        latitude=flux["lat"],
        longitude=flux["lon"],
        **kwargs,
    )
    assert list(covariances) == list(time.values)
    for month, covariance in enumerate(covariances.values()):
        sigmas = [covariance.source_covariances[source].sigma for source in SOURCES]
        np.testing.assert_allclose(sigmas, both.amplitudes.isel(time=month))


def test_length_one_slice_dimension_is_supported():
    """A single selected month keeps its slice dimension and label."""
    case = _toy_case()
    time = xr.DataArray(np.array(["2021-01-01"], dtype="datetime64[ns]"), dims="time")
    flux = case["flux"].expand_dims(time=time)
    result = calibrate_native_b_amplitudes(
        flux.sel(time=["2021-01-01"]),
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=0.1,
        class_labels=case["classes"],
    )
    alone = calibrate_native_b_amplitudes(
        case["flux"],
        case["weights"],
        correlation_length=1.5,
        target_relative_sd=0.1,
        class_labels=case["classes"],
    )
    assert result.amplitudes.sizes == {"source": 4, "time": 1}
    np.testing.assert_array_equal(result.amplitudes["time"], time)
    np.testing.assert_allclose(result.amplitudes.isel(time=0), alone.amplitudes, rtol=1e-12)
    assert result.diagnostics["achieved_sd"].sizes["time"] == 1


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
    assert not result.class_at_bound.any()
    assert set(result.diagnostics["status"].sel(functional=["LAND", "SEA", "OUTER"]).values.ravel()) == {
        "met"
    }
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


def test_invalid_inputs_are_rejected():
    """Ambiguous, misspelt or unsupported requests fail clearly."""
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
    with pytest.raises(ValueError, match="only statistic='geometric-mean'"):
        calibrate_native_b_amplitudes(
            case["flux"],
            case["weights"],
            target_relative_sd=0.1,
            per_class=True,
            statistic="median",
            class_labels=case["classes"],
            **kwargs,
        )
    zero = case["weights"].isel(functional=[0]) * 0.0
    with pytest.raises(ValueError, match="No usable target"):
        calibrate_native_b_amplitudes(case["flux"], zero, target_relative_sd=0.1, **kwargs)


def test_cancelling_total_uses_stabilized_scale():
    """A mixed-sign total is not given a near-zero SD target."""
    case = _toy_case()
    lon = case["flux"]["lon"]
    flux = case["flux"].sel(source=["co2_ocean"]).copy()
    # Net 0.125 against an absolute total of about 12 over the 20 inner cells.
    flux[:] = (lon - lon.mean()).broadcast_like(flux) + 0.125 / 20.0
    weights = (case["weights"].sel(functional=["EUROPE_INNER"]) > 0).astype(float)
    result = calibrate_native_b_amplitudes(
        flux,
        weights,
        correlation_length=1.5,
        target_relative_sd=0.5,
        class_labels=case["classes"],
    )
    row = result.diagnostics.sel(source="co2_ocean", functional="EUROPE_INNER")
    q = (flux.squeeze("source") * weights.squeeze("functional")).values
    absolute = float(np.abs(q).sum())
    assert float(row["native_signed_total"]) == pytest.approx(0.125)
    assert float(row["native_absolute_total"]) == pytest.approx(absolute) == pytest.approx(12.025)
    assert row["total_status"].item() == "cancellation"
    assert float(row["target_scale"]) == pytest.approx(0.1 * absolute)
    unit_sd, _ = _dense_unit_sd({**case, "flux": flux, "weights": weights}, 1.5, "co2_ocean", "EUROPE_INNER")
    assert float(result.amplitudes.squeeze()) == pytest.approx(0.5 * 0.1 * absolute / unit_sd, rel=1e-12)
    assert float(row["achieved_sd"]) == pytest.approx(0.05 * absolute, rel=1e-12)
    assert float(row["achieved_cv"]) == pytest.approx(0.05 * absolute / 0.125, rel=1e-12)
    assert row["status"].item() == "met"

    plain = calibrate_native_b_amplitudes(
        flux,
        weights,
        correlation_length=1.5,
        target_relative_sd=0.5,
        class_labels=case["classes"],
        cancellation_floor=0.0,
    )
    assert plain.diagnostics["total_status"].item() == "ok"
    ratio = 0.125 / (0.1 * absolute)
    assert float(plain.amplitudes.squeeze()) == pytest.approx(float(result.amplitudes.squeeze()) * ratio)


def test_unmet_and_infeasible_targets_are_reported():
    """Inconsistent targets are ``unmet``; a collapsed class is ``at_bound``."""
    case = _toy_case()
    flux = case["flux"].sel(source=["co2_FF"])
    inconsistent = xr.DataArray([0.05, 0.30], dims="functional", coords={"functional": ["AAA", "BBB"]})
    kwargs = {"correlation_length": 1.5, "class_labels": case["classes"]}
    result = calibrate_native_b_amplitudes(flux, case["weights"], target_relative_sd=inconsistent, **kwargs)
    status = result.diagnostics["status"].squeeze()
    assert status.sel(functional=["AAA", "BBB"]).values.tolist() == ["unmet", "unmet"]
    ratio = result.diagnostics["achieved_to_target_ratio"].squeeze()
    assert float(ratio.sel(functional="AAA") * ratio.sel(functional="BBB")) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="2 calibration targets were not met"):
        calibrate_native_b_amplitudes(
            flux, case["weights"], target_relative_sd=inconsistent, strict=True, **kwargs
        )

    inner = case["weights"].sel(functional="EUROPE_INNER", drop=True)
    land = inner.where(case["classes"] == "land", 0.0)
    outer = (case["classes"] == "outer") * np.cos(np.deg2rad(inner["lat"]))
    weights = xr.concat([land, inner, outer], dim=xr.DataArray(["LAND", "ALL", "OUTER"], dims="functional"))
    # LAND at 10% but land plus sea at 2% would need a negative sea variance.
    target = xr.DataArray(
        [0.10, 0.02, 0.5], dims="functional", coords={"functional": ["LAND", "ALL", "OUTER"]}
    )
    result = calibrate_native_b_amplitudes(flux, weights, target_relative_sd=target, per_class=True, **kwargs)
    bound = result.class_at_bound.squeeze()
    assert bound.sel({"class": "sea"}) and not bound.sel({"class": ["land", "outer"]}).any()
    sea = float(result.amplitudes.squeeze().sel({"class": "sea"}))
    land_amplitude = float(result.amplitudes.squeeze().sel({"class": "land"}))
    assert sea < 1.0e-4 * land_amplitude
    status = result.diagnostics["status"].squeeze()
    assert status.sel(functional="ALL").item() == "at_bound"
    assert status.sel(functional="LAND").item() == "unmet"
    assert status.sel(functional="OUTER").item() == "met"
    with pytest.raises(ValueError, match="not met"):
        calibrate_native_b_amplitudes(
            flux, weights, target_relative_sd=target, per_class=True, strict=True, **kwargs
        )


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        # Frozen from verification-games 13ae5f0
        # native_b_calibration.calibrate_native_b_amplitudes on this toy case,
        # per-source-country mode, countries AAA and BBB, target 0.5, one land
        # source per call, ocean over the inner domain.
        (1.5, {"co2_FF": 0.6418873512613098, "co2_GPP": 0.634377730134052, "co2_TER": 0.6384230267137582}),
        (3.0, {"co2_FF": 0.5707911901089585, "co2_GPP": 0.5674285662049466, "co2_TER": 0.5690698109848451}),
    ],
)
def test_median_statistic_matches_verification_games_over_two_countries(length, expected):
    """The median rule reproduces VG's ``target / median(unit relative SD)``."""
    case = _toy_case()
    target = xr.DataArray([0.5, 0.5], dims="functional", coords={"functional": ["AAA", "BBB"]})
    result = calibrate_native_b_amplitudes(
        case["flux"].sel(source=list(expected)),
        case["weights"],
        correlation_length=length,
        target_relative_sd=target,
        class_labels=case["classes"],
        statistic="median",
    )
    np.testing.assert_allclose(
        result.amplitudes.sel(source=list(expected)), list(expected.values()), rtol=1e-12
    )
    # With two regions the median is the midpoint of the unit relative SDs.
    for source, frozen in expected.items():
        relative = [
            _dense_unit_sd(case, length, source, country)[0]
            / abs(_dense_unit_sd(case, length, source, country)[1])
            for country in ("AAA", "BBB")
        ]
        assert 0.5 / np.mean(relative) == pytest.approx(frozen, rel=1e-12)
    ratio = result.diagnostics["achieved_to_target_ratio"].sel(functional=["AAA", "BBB"])
    np.testing.assert_allclose(ratio.median("functional"), 1.0, rtol=1e-12)
