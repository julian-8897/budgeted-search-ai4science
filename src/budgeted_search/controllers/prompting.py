"""Prompt templates and message construction for the language controller."""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from budgeted_search.contracts import TrialContext, TrialResult
from budgeted_search.records import json_value
from budgeted_search.space import Parameter, SearchSpace

TEMPLATE_NAMES = ("system", "initial", "iteration")


@dataclass(frozen=True)
class PromptTemplates:
    system: str
    initial: str
    iteration: str

    @classmethod
    def default(cls) -> PromptTemplates:
        root = resources.files("budgeted_search").joinpath("prompts")
        return cls(*(root.joinpath(f"{name}.md").read_text(encoding="utf-8") for name in TEMPLATE_NAMES))

    @classmethod
    def from_dir(cls, path: Path) -> PromptTemplates:
        return cls(*((Path(path) / f"{name}.md").read_text(encoding="utf-8") for name in TEMPLATE_NAMES))

    def write_to(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        for name in TEMPLATE_NAMES:
            (path / f"{name}.md").write_text(getattr(self, name), encoding="utf-8")


def render(template: str, values: Mapping[str, str]) -> str:
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None and field not in values:
            raise ValueError(f"Template references unknown placeholder {{{field}}}; available: {sorted(values)}")
    return template.format_map(values)


def format_value(value: Any, max_list_items: int = 10) -> str:
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else repr(value)
    if isinstance(value, str | int):
        return str(value)
    if isinstance(value, list | tuple):
        items = list(value)
        if len(items) > max_list_items:
            last = len(items) - 1
            items = [items[round(i * last / (max_list_items - 1))] for i in range(max_list_items)]
        return "[" + ", ".join(format_value(item, max_list_items) for item in items) + "]"
    if isinstance(value, dict) and set(value) == {"nonfinite"}:
        return str(value["nonfinite"])
    return json.dumps(json_value(value), allow_nan=False)


def improvement(direction: str, earlier_scores: Sequence[float], current: float) -> float:
    """Signed gain over the best earlier score; positive means the current trial is a new best."""
    if not earlier_scores:
        return 0.0
    if direction == "minimize":
        return min(earlier_scores) - current
    return current - max(earlier_scores)


def _finite_score(result: TrialResult) -> float | None:
    if result.status == "complete" and result.score is not None and math.isfinite(result.score):
        return result.score
    return None


def _config_json(parameters: Mapping[str, Any], strategy: str | None = None) -> str:
    config = {"strategy": strategy, **parameters} if strategy is not None else dict(parameters)
    return json.dumps(json_value(config), allow_nan=False)


def _describe_parameter(parameter: Parameter) -> tuple[str, str, str]:
    if parameter.kind == "categorical":
        return "categorical", json.dumps(list(parameter.choices)), "n/a"
    scale = "log" if parameter.log else "linear"
    if parameter.kind == "int":
        return "int", f"{parameter.low} to {parameter.high}", scale
    return "float", f"{parameter.low:.6g} to {parameter.high:.6g}", scale


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def search_space_table(space: SearchSpace) -> str:
    rows = ["| Name | Type | Range or choices | Scale | Description |", "| --- | --- | --- | --- | --- |"]
    for parameter in space.parameters:
        kind, domain, scale = _describe_parameter(parameter)
        rows.append(f"| `{parameter.name}` | {kind} | {_cell(domain)} | {scale} | {_cell(parameter.description)} |")
    return "\n".join(rows)


def _label(records: Mapping[int, Mapping[str, str]], trial_id: int) -> str:
    return records[trial_id]["strategy"] if trial_id in records else "other"


def _history_lines(
    history: Sequence[TrialResult], records: Mapping[int, Mapping[str, str]], max_list_items: int
) -> str:
    lines = []
    for result in history:
        score = format_value(result.score, max_list_items) if result.score is not None else "n/a"
        line = (
            f"Trial {result.trial_id} [{_label(records, result.trial_id)}]: "
            f"config={_config_json(result.parameters)} score={score} status={result.status}"
        )
        if result.error:
            line += f" error={result.error.splitlines()[0][:120]}"
        lines.append(line)
    labels = [_label(records, result.trial_id) for result in history]
    explore, exploit = labels.count("explore"), labels.count("exploit")
    lines.append(
        f"Strategy distribution (all {len(history)} prior trials): "
        f"{explore} explore, {exploit} exploit, {len(labels) - explore - exploit} other"
    )
    return "\n".join(lines)


def _feedback_block(previous: TrialResult, score_name: str, max_list_items: int) -> str:
    score = format_value(previous.score, max_list_items) if previous.score is not None else "n/a"
    lines = [f"- Status: {previous.status}", f"- {score_name}: {score}"]
    lines += [f"- {key}: {format_value(value, max_list_items)}" for key, value in previous.feedback.items()]
    if previous.error:
        lines.append(f"- Error: {previous.error}")
    return "\n".join(lines)


def build_messages(
    templates: PromptTemplates,
    space: SearchSpace,
    context: TrialContext,
    history: Sequence[TrialResult],
    records: Mapping[int, Mapping[str, str]],
    *,
    max_list_items: int = 10,
) -> list[dict[str, str]]:
    """Render the system and user messages; `records` holds the strategy and reasoning this controller proposed."""
    history = sorted(history, key=lambda result: result.trial_id)
    system = render(
        templates.system,
        {
            "task_description": context.task_description,
            "score_name": context.score_name,
            "direction_phrase": "lower is better" if context.direction == "minimize" else "higher is better",
            "trial_budget": str(context.trial_id + context.trials_remaining),
            "search_space": search_space_table(space),
            "constraints": space.constraint_description or "None",
            "parameter_keys": ", ".join(f'"{p.name}"' for p in space.parameters),
        },
    )
    if not history:
        user = render(
            templates.initial,
            {
                "trials_remaining": str(context.trials_remaining),
                "score_name": context.score_name,
                "note": context.note,
            },
        )
    else:
        previous = history[-1]
        earlier = [score for r in history[:-1] if (score := _finite_score(r)) is not None]
        current = _finite_score(previous)
        scored = [(score, r.trial_id) for r in history if (score := _finite_score(r)) is not None]
        if scored:
            best_score, best_trial = (min if context.direction == "minimize" else max)(scored, key=lambda s: s[0])
            best_summary = f"{format_value(best_score)} (trial {best_trial})"
        else:
            best_summary = "none yet (no completed trial with a finite score)"
        record = records.get(previous.trial_id)
        user = render(
            templates.iteration,
            {
                "previous_trial": str(previous.trial_id),
                "note": context.note,
                "previous_config": _config_json(previous.parameters, record["strategy"] if record else None),
                "previous_reasoning": (record["reasoning"] or "(empty)")
                if record
                else "Not proposed by this controller",
                "observed_feedback": _feedback_block(previous, context.score_name, max_list_items),
                "score_name": context.score_name,
                "best_summary": best_summary,
                "improvement": f"{improvement(context.direction, earlier, current):+.6g}"
                if current is not None
                else "n/a",
                "trials_remaining": str(context.trials_remaining),
                "history_length": str(len(history)),
                "history_summary": _history_lines(history, records, max_list_items),
            },
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
