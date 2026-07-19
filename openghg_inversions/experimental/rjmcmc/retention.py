"""Collection-time retention settings for experimental RJMCMC traces."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _non_negative_integer(value: int, *, name: str) -> int:
    """Return ``value`` as a non-negative built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class RetentionSettings:
    """Select saved states by their global completed-transition number.

    A state after global transition count ``t`` is retained when
    ``t >= warmup_transitions`` and
    ``(t - warmup_transitions) % thin == 0``. The initial state has transition
    count zero. Continuation segments preserve this global phase.

    Args:
        warmup_transitions: Number of attempted transitions before the first
            retained state.
        thin: Positive transition interval between retained states.

    Raises:
        ValueError: If either setting is malformed.
    """

    warmup_transitions: int = 0
    thin: int = 1

    def __post_init__(self) -> None:
        """Validate non-negative warmup and positive thinning."""
        warmup = _non_negative_integer(self.warmup_transitions, name="warmup_transitions")
        thin = _non_negative_integer(self.thin, name="thin")
        if thin < 1:
            raise ValueError("thin must be a positive integer.")
        object.__setattr__(self, "warmup_transitions", warmup)
        object.__setattr__(self, "thin", thin)

    def retains(self, transitions_completed: int) -> bool:
        """Return whether a state at a global transition count is retained."""
        transition = _non_negative_integer(
            transitions_completed,
            name="transitions_completed",
        )
        return (
            transition >= self.warmup_transitions and (transition - self.warmup_transitions) % self.thin == 0
        )


__all__ = ["RetentionSettings"]
