"""Interfaces for experiments and search controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class TrialContext:
    """Experiments must train with `seed` (run seed plus trial index) so the recorded seed is the one used."""

    trial_id: int
    seed: int
    trials_remaining: int
    output_dir: str
    task_description: str
    direction: Literal["minimize", "maximize"] = "minimize"
    score_name: str = "score"
    note: str = ""


@dataclass(frozen=True)
class Proposal:
    parameters: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    score: float
    artefact: str
    metrics: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    seed: int
    parameters: dict[str, Any]
    status: Literal["complete", "failed"]
    score: float | None
    artefact: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Controller(Protocol):
    def ask(self, context: TrialContext, history: tuple[TrialResult, ...]) -> Proposal: ...

    def tell(self, result: TrialResult) -> None: ...


class Experiment(Protocol):
    def run_trial(self, parameters: dict[str, Any], context: TrialContext) -> Observation: ...

    def evaluate_selected(self, artefact: str) -> dict[str, Any]: ...
