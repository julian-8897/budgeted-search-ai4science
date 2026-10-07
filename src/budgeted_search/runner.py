"""A serial runner with explicit selection and failure semantics."""

from __future__ import annotations

import math
import numbers
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from budgeted_search.contracts import Controller, Experiment, TrialContext, TrialResult
from budgeted_search.records import EventLog, build_manifest, json_value, write_json
from budgeted_search.space import SearchSpace


@dataclass(frozen=True)
class RunConfig:
    trials: int
    seed: int
    task_description: str
    direction: Literal["minimize", "maximize"] = "minimize"
    on_trial_error: Literal["abort", "continue"] = "abort"
    score_name: str = "score"

    def __post_init__(self) -> None:
        if type(self.trials) is not int or self.trials <= 0 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Trials must be positive and seed must be a non-negative integer")
        if not self.task_description.strip():
            raise ValueError("Describe the task")
        if not self.score_name.strip():
            raise ValueError("Name the score")
        if self.direction not in ("minimize", "maximize") or self.on_trial_error not in ("abort", "continue"):
            raise ValueError("Invalid selection or failure policy")


def prefer_finite(candidate: TrialResult, incumbent: TrialResult | None, direction: str) -> bool:
    if candidate.status != "complete" or candidate.score is None or not math.isfinite(candidate.score):
        return False
    if incumbent is None:
        return True
    return candidate.score < incumbent.score if direction == "minimize" else candidate.score > incumbent.score


class Runner:
    def __init__(
        self,
        config: RunConfig,
        space: SearchSpace,
        *,
        prefer: Callable[[TrialResult, TrialResult | None, str], bool] = prefer_finite,
    ) -> None:
        self.config = config
        self.space = space
        self.prefer = prefer

    def run(
        self,
        experiment: Experiment,
        controller: Controller,
        output_dir: Path,
        *,
        project_root: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        events = EventLog(output_dir / "events.jsonl")
        history: list[TrialResult] = []
        selected = None
        write_json(
            output_dir / "config.json", {"run": self.config, "space": self.space.describe(), "metadata": metadata}
        )
        write_json(output_dir / "manifest.json", build_manifest(project_root, metadata))
        events.append("run_started")
        try:
            for trial_id in range(self.config.trials):
                trial_dir = output_dir / f"trial_{trial_id:04d}"
                trial_dir.mkdir()
                context = TrialContext(
                    trial_id=trial_id,
                    seed=self.config.seed + trial_id,
                    trials_remaining=self.config.trials - trial_id,
                    output_dir=str(trial_dir),
                    task_description=self.config.task_description,
                    direction=self.config.direction,
                    score_name=self.config.score_name,
                )
                try:
                    proposal = controller.ask(context, tuple(history))
                except Exception:
                    events.append("proposal_failed", trial_id=trial_id, audit=getattr(controller, "last_audit", {}))
                    raise
                events.append("proposal", trial_id=trial_id, proposal=proposal)
                parameters = proposal.parameters
                started = time.monotonic()
                error = None
                try:
                    parameters = self.space.validate(parameters)
                    observation = experiment.run_trial(parameters, context)
                    score = observation.score
                    if isinstance(score, bool) or not isinstance(score, numbers.Real):
                        raise TypeError("Selection score must be a number")
                    if not isinstance(observation.artefact, str) or not observation.artefact:
                        raise ValueError("Completed trials require an artefact reference")
                    result = TrialResult(
                        trial_id,
                        context.seed,
                        parameters,
                        "complete",
                        float(score),
                        observation.artefact,
                        json_value(observation.metrics),
                        json_value(observation.feedback),
                        json_value({**observation.costs, "wall_seconds": time.monotonic() - started}),
                    )
                except Exception as exc:
                    error = exc
                    result = TrialResult(
                        trial_id,
                        context.seed,
                        json_value(parameters),
                        "failed",
                        None,
                        None,
                        costs={"wall_seconds": time.monotonic() - started},
                        error=f"{type(exc).__name__}: {exc}",
                    )
                events.append("trial_finished", result=result)
                history.append(result)
                write_json(trial_dir / "result.json", result)
                controller.tell(result)
                if error is not None and self.config.on_trial_error == "abort":
                    raise error
                if self.prefer(result, selected, self.config.direction):
                    selected = result
            if selected is None:
                raise RuntimeError("No selectable trial; held-out evaluation was not run")
            events.append("selection", trial_id=selected.trial_id, artefact=selected.artefact)
            test_metrics = experiment.evaluate_selected(selected.artefact)
            summary = {"status": "complete", "selected": selected, "trials": history, "test_metrics": test_metrics}
            write_json(output_dir / "summary.json", summary)
            events.append("run_finished", selected_trial=selected.trial_id)
            return summary
        except BaseException as exc:
            events.append("run_failed", error=f"{type(exc).__name__}: {exc}")
            write_json(output_dir / "failure.json", {"error": f"{type(exc).__name__}: {exc}", "trials": history})
            raise
