"""Shared conformance tests for declared xarray contracts."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import dask
import pytest
import xarray as xr

import openghg_inversions
from openghg_inversions.xr_contracts import registered_contracts, xr_contract


PACKAGE_ROOT = Path(openghg_inversions.__file__).parent
TESTS_ROOT = Path(__file__).parent


def _import_contract_modules() -> None:
    """Import every ``contracts.py`` module in the package, found by filename."""
    for path in PACKAGE_ROOT.rglob("contracts.py"):
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        importlib.import_module(".".join(parts))


def _contract_examples() -> dict[type, object]:
    """Collect ``CONTRACT_EXAMPLES`` from contract test modules, keyed by contract type."""
    examples: dict[type, object] = {}
    for path in TESTS_ROOT.rglob("test_*contracts.py"):
        if path == Path(__file__):
            continue
        spec = importlib.util.spec_from_file_location(f"_contract_examples_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        examples.update(getattr(module, "CONTRACT_EXAMPLES", {}))
    return examples


_import_contract_modules()
CONTRACTS = registered_contracts()
EXAMPLES = _contract_examples()


def _forbid_compute(*_args, **_kwargs):
    """Dask scheduler that fails if validation computes anything."""
    raise AssertionError("Contract validation computed a Dask payload.")


def test_contract_modules_are_discovered():
    """Discovery imports at least the boundary contracts without an import list."""
    assert "openghg_inversions.models.boundary.contracts.RawCenteredSensitivity" in CONTRACTS


def test_every_declared_contract_supplies_examples():
    """A declared contract without examples fails the suite."""
    missing = sorted(name for name, contract in CONTRACTS.items() if contract not in EXAMPLES)
    assert not missing


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_validation_returns_the_identical_object_without_computing(name, lazy):
    """Validation preserves identity, class and lazy payloads."""
    contract = CONTRACTS[name]
    data, kwargs = EXAMPLES[contract](lazy)
    assert isinstance(data, xr.DataArray | xr.Dataset | xr.DataTree)
    original_class = type(data)
    with dask.config.set(scheduler=_forbid_compute):
        result = contract.validate(data, **kwargs)
    assert result is data
    assert type(result) is original_class


def test_declaration_rejects_unsupported_types():
    """Only xarray containers with their own class-level validator are declared."""
    with pytest.raises(TypeError, match="must subclass"):
        xr_contract(int)  # type: ignore[type-var]

    class NoValidator(xr.Dataset):
        __slots__ = ()

    NoValidator.__module__ = "openghg_inversions.example.contracts"
    with pytest.raises(TypeError, match="own staticmethod or classmethod validate"):
        xr_contract(NoValidator)

    class InstanceValidator(xr.Dataset):
        __slots__ = ()

        def validate(self) -> InstanceValidator:
            return self

    InstanceValidator.__module__ = "openghg_inversions.example.contracts"
    with pytest.raises(TypeError, match="own staticmethod or classmethod validate"):
        xr_contract(InstanceValidator)


def test_declaration_rejects_modules_outside_contracts():
    """Declarations must live in a discoverable ``contracts`` module."""

    class Misplaced(xr.DataArray):
        __slots__ = ()

        @staticmethod
        def validate(data: xr.DataArray) -> Misplaced:
            return data  # type: ignore[return-value]

    with pytest.raises(TypeError, match="'contracts' module"):
        xr_contract(Misplaced)
