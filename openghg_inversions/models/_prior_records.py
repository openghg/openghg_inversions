"""Constructed prior descriptions: finite scalars and explicit symbolic references."""

from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

from pydantic import ConfigDict, Field, field_validator

from openghg_inversions._record_schema import JsonFloat, Record

if TYPE_CHECKING:
    import xarray as xr

Name = Annotated[str, Field(pattern=r"\S")]


class PriorDataReference(Record):
    data: Name


class PriorVariableReference(Record):
    variable: Name


class PriorSymbolicReference(Record):
    symbolic: Literal[True]

    @field_validator("symbolic", mode="before")
    @classmethod
    def require_boolean(cls, value: object) -> object:
        if value is not True:
            raise ValueError("symbolic must be the Boolean true")
        return value


PriorParameter = str | bool | int | JsonFloat | PriorDataReference | PriorVariableReference | PriorSymbolicReference | None


class PriorDescription(Record):
    """Describe a used prior without turning symbolic parameters into a recipe.

    Distribution-specific parameters stay open for scientific extensions. Each
    value must be a finite JSON scalar or a declared reference; array payloads
    belong to the trace's registered ``constant_data`` group.
    """

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, PriorParameter] = Field(init=False)
    pdf: Name

    def validate_references(self, trace: xr.DataTree) -> None:
        """Check registered names without reading or computing their payloads."""
        for value in (self.__pydantic_extra__ or {}).values():
            if isinstance(value, PriorDataReference):
                if "constant_data" not in trace.children or value.data not in trace["constant_data"].data_vars:
                    raise ValueError(f"Missing registered prior data {value.data!r}.")
