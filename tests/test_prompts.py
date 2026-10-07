from __future__ import annotations

import pytest

from budgeted_search.contracts import TrialContext, TrialResult
from budgeted_search.controllers.prompting import PromptTemplates, build_messages, format_value, improvement, render
from budgeted_search.space import Parameter, SearchSpace


def space():
    return SearchSpace(
        (
            Parameter("x", "float", "Coordinate | pipe", low=-1.0, high=1.0),
            Parameter("scale", "float", "Log scale", low=1e-3, high=1.0, log=True),
            Parameter("depth", "int", "Layers", low=1, high=8),
            Parameter("mode", "categorical", "Variant", choices=("plain", "shifted")),
        )
    )


def context(trial_id=0, remaining=5, **options):
    return TrialContext(trial_id, 42 + trial_id, remaining, "/tmp", "Tune the synthetic task", **options)


def done(index, score, **options):
    parameters = {"x": 0.1 * index, "scale": 0.1, "depth": 2, "mode": "plain"}
    return TrialResult(index, 42 + index, parameters, "complete", score, f"a{index}", **options)


def failed(index, error="RuntimeError: out of memory\ntraceback line"):
    return TrialResult(
        index, 42 + index, {"x": 0.0, "scale": 0.1, "depth": 2, "mode": "plain"}, "failed", None, None, error=error
    )


def user(messages):
    return messages[1]["content"]


def test_system_prompt_lists_the_search_space_and_keys():
    system = build_messages(PromptTemplates.default(), space(), context(score_name="validation error"), (), {})[0]
    text = system["content"]
    assert system["role"] == "system"
    assert "| `x` | float | -1 to 1 | linear | Coordinate \\| pipe |" in text
    assert "| `scale` | float | 0.001 to 1 | log | Log scale |" in text
    assert "| `depth` | int | 1 to 8 | linear | Layers |" in text
    assert '| `mode` | categorical | ["plain", "shifted"] | n/a | Variant |' in text
    assert '"x", "scale", "depth", "mode"' in text
    assert "validation error" in text and "lower is better" in text
    assert "Tune the synthetic task" in text and "Constraints: None" in text
    assert "has 5 trials in total" in text


def test_system_prompt_states_constraints_and_direction():
    constrained = SearchSpace(space().parameters, constraints=lambda _: None, constraint_description="x < 0.5")
    text = build_messages(PromptTemplates.default(), constrained, context(direction="maximize"), (), {})[0]["content"]
    assert "Constraints: x < 0.5" in text and "higher is better" in text


def test_initial_prompt_has_no_history_and_carries_the_note():
    text = user(build_messages(PromptTemplates.default(), space(), context(note="Handoff text."), (), {}))
    assert "trial 0" in text and "5 trials remain" in text and "Handoff text." in text
    assert "History" not in text


def test_iteration_prompt_condenses_every_prior_trial():
    history = (done(0, 0.5), done(1, 0.4), failed(2))
    records = {0: {"reasoning": "first", "strategy": "explore"}, 1: {"reasoning": "second", "strategy": "exploit"}}
    text = user(build_messages(PromptTemplates.default(), space(), context(3, 2, note="Note."), history, records))
    assert "Trial 2 results:" in text and "Note." in text
    assert "Trial 0 [explore]: config=" in text and "score=0.5 status=complete" in text
    assert "Trial 1 [exploit]: config=" in text
    assert "Trial 2 [other]: config=" in text and "status=failed error=RuntimeError: out of memory" in text
    assert "traceback line" not in text.split("**History**")[1]
    assert "Strategy distribution (all 3 prior trials): 1 explore, 1 exploit, 1 other" in text
    assert "Not proposed by this controller" in text
    assert "- Error: RuntimeError: out of memory\ntraceback line" in text
    assert "Improvement over the earlier best: n/a" in text
    assert "Best `score` so far: 0.4 (trial 1)" in text
    assert "Trials remaining, including the one you are about to propose: 2" in text


def test_iteration_prompt_shows_previous_reasoning_strategy_and_feedback():
    history = (done(0, 0.5, feedback={"loss": [float(i) for i in range(30)], "note": "ok", "nan": float("nan")}),)
    records = {0: {"reasoning": "Try a wide scale.", "strategy": "explore"}}
    text = user(build_messages(PromptTemplates.default(), space(), context(1, 4), history, records))
    assert "Try a wide scale." in text
    assert '"strategy": "explore"' in text
    assert "- score: 0.5" in text and "- note: ok" in text and "- nan: nan" in text
    loss_line = next(line for line in text.splitlines() if line.startswith("- loss:"))
    assert loss_line.startswith("- loss: [0, 3, ") and loss_line.endswith(", 29]") and loss_line.count(",") == 9


@pytest.mark.parametrize(
    ("direction", "scores", "expected"),
    [
        ("minimize", (5.0, 3.0), "+2"),
        ("minimize", (3.0, 5.0), "-2"),
        ("minimize", (3.0, 3.0), "+0"),
        ("minimize", (4.0,), "+0"),
        ("maximize", (3.0, 5.0), "+2"),
        ("maximize", (5.0, 3.0), "-2"),
        ("maximize", (4.0,), "+0"),
    ],
)
def test_improvement_sign_follows_direction(direction, scores, expected):
    history = tuple(done(i, score) for i, score in enumerate(scores))
    text = user(
        build_messages(PromptTemplates.default(), space(), context(len(scores), 3, direction=direction), history, {})
    )
    assert f"Improvement over the earlier best: {expected} (positive = new best" in text


def test_improvement_uses_only_trials_before_the_previous_one():
    assert improvement("minimize", [5.0, 2.0], 3.0) == -1.0
    assert improvement("maximize", [5.0, 2.0], 7.0) == 2.0
    assert improvement("minimize", [], 3.0) == 0.0


def test_history_is_ordered_by_trial_id():
    text = user(build_messages(PromptTemplates.default(), space(), context(2, 3), (done(1, 0.4), done(0, 0.5)), {}))
    assert text.index("Trial 0 [") < text.index("Trial 1 [")
    assert "Trial 1 results:" in text


def test_unknown_placeholder_in_a_custom_template_raises():
    templates = PromptTemplates("{task_description} {missing}", "initial", "iteration")
    with pytest.raises(ValueError, match="unknown placeholder {missing}"):
        build_messages(templates, space(), context(), (), {})
    with pytest.raises(ValueError, match="unknown placeholder"):
        render("{}", {"a": "b"})


def test_escaped_braces_render_literally():
    assert render('{{"a": "{name}"}}', {"name": "v"}) == '{"a": "v"}'


def test_templates_round_trip_through_a_directory(tmp_path):
    default = PromptTemplates.default()
    default.write_to(tmp_path / "prompts")
    assert PromptTemplates.from_dir(tmp_path / "prompts") == default


def test_format_value():
    assert format_value(0.123456789) == "0.123457"
    assert format_value(float("nan")) == "nan" and format_value(float("-inf")) == "-inf"
    assert format_value({"nonfinite": "inf"}) == "inf"
    assert format_value({"a": {"b": 1}}) == '{"a": {"b": 1}}'
    assert format_value([1.0, 2.0]) == "[1, 2]"
    assert format_value(True) == "true" and format_value(None) == "null"
    assert format_value(list(range(100)), 5) == "[0, 25, 50, 74, 99]"
