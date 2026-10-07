"""Described parameters and shared proposal validation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: Literal["float", "int", "categorical"]
    description: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    log: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise ValueError("Parameters require a name and description")
        if self.kind == "categorical":
            if not self.choices or self.log or self.low is not None or self.high is not None:
                raise ValueError("Categorical parameters require choices and no numeric bounds")
            if any(not isinstance(v, str | bool | int | float | type(None)) for v in self.choices):
                raise ValueError("Categorical choices must be JSON scalars")
            if any(isinstance(v, float) and not math.isfinite(v) for v in self.choices):
                raise ValueError("Categorical choices must be finite")
            if len({(type(v), v) for v in self.choices}) != len(self.choices):
                raise ValueError("Categorical choices must be distinct")
        elif self.kind in ("float", "int"):
            if self.choices or any(
                isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
                for v in (self.low, self.high)
            ):
                raise ValueError("Numeric parameters require finite numeric bounds and no choices")
            if self.low > self.high or (self.log and self.low <= 0):
                raise ValueError("Invalid numeric bounds")
            if self.kind == "int" and any(type(v) is not int for v in (self.low, self.high)):
                raise ValueError("Integer bounds must be integers")
        else:
            raise ValueError(f"Unknown parameter kind: {self.kind}")

    def validate(self, value: Any) -> None:
        if self.kind == "categorical":
            valid = any(type(value) is type(v) and value == v for v in self.choices)
        else:
            valid = (
                not isinstance(value, bool)
                and isinstance(value, int | float)
                and math.isfinite(value)
                and self.low <= value <= self.high
                and (self.kind != "int" or type(value) is int)
            )
        if not valid:
            raise ValueError(f"Invalid value for {self.name}: {value!r}")


def mixed_distance(space: SearchSpace) -> Callable[[Mapping[str, Any], Mapping[str, Any]], float]:
    """Mean per-parameter distance in [0, 1]: unit-scaled for numeric, 0/1 mismatch for categorical."""
    live = tuple(p for p in space.parameters if p.kind == "categorical" or p.low != p.high)

    def unit(parameter: Parameter, value: float) -> float:
        low, high = parameter.low, parameter.high
        if parameter.log:
            value, low, high = math.log10(value), math.log10(low), math.log10(high)
        return min(1.0, max(0.0, (value - low) / (high - low)))

    def distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        # Zero-width parameters are gated constants: counting them would shrink every distance.
        if not live:
            return 0.0
        total = 0.0
        for parameter in live:
            x, y = a[parameter.name], b[parameter.name]
            if parameter.kind == "categorical":
                total += 0.0 if type(x) is type(y) and x == y else 1.0
            else:
                total += abs(unit(parameter, x) - unit(parameter, y))
        return total / len(live)

    return distance


class SearchSpace:
    def __init__(
        self,
        parameters: tuple[Parameter, ...],
        *,
        constraints: Callable[[Mapping[str, Any]], None] | None = None,
        distance: Callable[[Mapping[str, Any], Mapping[str, Any]], float] | None = None,
        constraint_description: str = "",
    ) -> None:
        if not parameters or len({p.name for p in parameters}) != len(parameters):
            raise ValueError("Search space requires uniquely named parameters")
        if constraints is not None and not constraint_description:
            raise ValueError("Describe coupled constraints for the controller")
        self.parameters = parameters
        self.constraints = constraints
        self.distance = distance if distance is not None else mixed_distance(self)
        self.constraint_description = constraint_description

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        expected = {p.name for p in self.parameters}
        if set(values) != expected:
            raise ValueError(f"Parameter keys differ: missing={expected - set(values)}, extra={set(values) - expected}")
        for parameter in self.parameters:
            parameter.validate(values[parameter.name])
        if self.constraints is not None:
            self.constraints(values)
        return dict(values)

    def describe(self) -> dict[str, Any]:
        return {"parameters": [asdict(p) for p in self.parameters], "constraints": self.constraint_description}
