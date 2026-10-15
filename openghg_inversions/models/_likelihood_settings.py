"""Validated component settings; numerical prior leaves remain borrowed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
from functools import cache
from typing import Annotated, Any, ClassVar, Union

import numpy as np
from pydantic import (
    BeforeValidator, ConfigDict, Field, StrictBool, StrictStr, TypeAdapter, ValidationError, ValidationInfo, field_validator, model_validator,
)
from pydantic.dataclasses import dataclass

from openghg_inversions._record_schema import Record
from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.models.priors import PriorArgs


SETTINGS_CONFIG = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class LikelihoodSettings:
    """Small component values selected by a Pydantic discriminated union."""

    @staticmethod
    def own_containers(value: Any) -> Any:
        """Copy ordinary option containers while borrowing opaque numerical leaves."""
        if isinstance(value, dict):
            return {key: LikelihoodSettings.own_containers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [LikelihoodSettings.own_containers(item) for item in value]
        if isinstance(value, tuple):
            return tuple(LikelihoodSettings.own_containers(item) for item in value)
        return value

    @classmethod
    def variants(cls) -> tuple[type, ...]:
        from .additive_sigma import AdditiveSigmaSettings
        from .fixed_error import FixedErrorSettings
        from .pollution_event import PollutionEventSettings

        return PollutionEventSettings, AdditiveSigmaSettings, FixedErrorSettings

    @classmethod
    @cache
    def adapter(cls) -> TypeAdapter:
        return TypeAdapter(Annotated[Union[cls.variants()], Field(discriminator="kind")])

    @classmethod
    def option_names(cls) -> frozenset[str]:
        return frozenset(field.name for variant in cls.variants() for field in fields(variant) if field.name != "kind")

    @classmethod
    def from_params(
        cls, remaining: dict[str, Any], *, mismatch_model: str | None, start_date: str,
    ) -> LikelihoodSettings | None:
        """Consume owned external options; Pydantic selects and validates their owner."""
        names = cls.option_names() & remaining.keys()
        if mismatch_model is None:
            if names:
                raise ValueError(
                    "Built-in likelihood option(s) cannot be used with `mismatch_model=None`: "
                    f"{sorted(names)!r}. Pass custom options through `likelihood_kwargs`."
                )
            return None
        try:
            return cls.adapter().validate_python(
                {"kind": mismatch_model, **{name: remaining.pop(name) for name in names}},
                context={"start_date": start_date},
            )
        except ValidationError as error:
            title = f"Invalid `mismatch_model={mismatch_model!r}` options"
            raise ValidationError.from_exception_data(title, error.errors(include_url=False)) from error

    def to_record(self, *, priors: Mapping[str, PriorArgs]) -> Record:
        """Persist construction-owned priors through the concrete component."""
        raise NotImplementedError


BorrowedPrior = Annotated[PriorArgs, Field(strict=True), BeforeValidator(LikelihoodSettings.own_containers)]


@dataclass(frozen=True, config=SETTINGS_CONFIG)
class SigmaSettings(LikelihoodSettings):
    """Shared period choices, with explicit temporal projection and borrowed priors."""

    DEFAULT_SIGMA_PRIOR: ClassVar[PriorArgs]

    sigma_prior: BorrowedPrior | None = None
    sigma_freq: StrictStr | None = None
    sigma_per_site: StrictBool = True
    sigma_freq_anchor: StrictStr | None = None

    @field_validator("sigma_freq_anchor", mode="before")
    @classmethod
    def project_anchor(cls, value: Any) -> str | None:
        if value is not None and not isinstance(value, str | datetime | np.datetime64):
            raise ValueError("sigma_freq_anchor must be a date-time string or temporal scalar")
        return _json_value(value)

    @model_validator(mode="before")
    @classmethod
    def resolve_requested_defaults(cls, value: Any, info: ValidationInfo) -> Any:
        if info.context is None:
            return value
        options = dict(value)
        if options.get("sigma_prior") is None:
            options["sigma_prior"] = dict(cls.DEFAULT_SIGMA_PRIOR)
        options.setdefault("sigma_freq_anchor", info.context["start_date"])
        return options

    def recorded_options(self) -> dict[str, Any]:
        """Exclude every caller prior before serializing ordinary option values."""
        return TypeAdapter(type(self)).dump_python(self, exclude={"kind", "sigma_prior", "power"})
