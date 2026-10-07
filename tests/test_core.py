from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from budgeted_search.contracts import Observation
from budgeted_search.controllers.replay import ReplayController
from budgeted_search.records import json_value
from budgeted_search.runner import RunConfig, Runner
from budgeted_search.space import Parameter, SearchSpace, mixed_distance


def space():
    return SearchSpace((Parameter("x", "float", "Coordinate", low=-1.0, high=1.0),))


class Experiment:
    def __init__(self, fail=None, scores=None):
        self.fail = fail
        self.scores = scores
        self.evaluations = []

    def run_trial(self, parameters, context):
        if context.trial_id == self.fail:
            raise ValueError("Synthetic trial failure")
        score = self.scores[context.trial_id] if self.scores is not None else parameters["x"] ** 2
        return Observation(score, f"trial-{context.trial_id}", feedback={"seed": context.seed})

    def evaluate_selected(self, artefact):
        self.evaluations.append(artefact)
        return {"held_out": 999}


def run(tmp_path, experiment, proposals, **options):
    return Runner(RunConfig(len(proposals), 42, "Synthetic task", **options), space()).run(
        experiment,
        ReplayController(proposals),
        tmp_path / "run",
        project_root=tmp_path,
        metadata={},
    )


def test_selects_validation_artefact_and_tests_once(tmp_path):
    experiment = Experiment()
    summary = run(tmp_path, experiment, [{"x": 0.8}, {"x": 0.2}, {"x": 0.2}])
    assert summary["selected"].trial_id == 1
    assert experiment.evaluations == ["trial-1"]
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert [e["result"]["seed"] for e in events if e["event"] == "trial_finished"] == [42, 43, 44]
    assert not any("held_out" in json.dumps(e) for e in events if e["event"] == "proposal")


def test_history_contains_no_test_feedback(tmp_path):
    class RecordingController(ReplayController):
        def ask(self, context, history):
            assert all("held_out" not in result.metrics and "held_out" not in result.feedback for result in history)
            return super().ask(context, history)

    experiment = Experiment()
    Runner(RunConfig(2, 42, "Task"), space()).run(
        experiment,
        RecordingController([{"x": 0.5}, {"x": 0.0}]),
        tmp_path / "run",
        project_root=tmp_path,
        metadata={},
    )
    assert len(experiment.evaluations) == 1


def test_failure_is_persisted_and_aborts_before_test(tmp_path):
    experiment = Experiment(fail=0)
    with pytest.raises(ValueError, match="Synthetic"):
        run(tmp_path, experiment, [{"x": 0.2}])
    assert experiment.evaluations == []
    failure = json.loads((tmp_path / "run/failure.json").read_text())
    assert failure["trials"][0]["status"] == "failed"


def test_continue_keeps_failed_trials(tmp_path):
    summary = run(tmp_path, Experiment(fail=0), [{"x": 0.2}, {"x": 0.8}], on_trial_error="continue")
    assert len(summary["trials"]) == 2
    assert summary["selected"].trial_id == 1


def test_nonfinite_trials_never_reach_test(tmp_path):
    experiment = Experiment(scores=[float("nan"), float("inf")])
    with pytest.raises(RuntimeError, match="No selectable"):
        run(tmp_path, experiment, [{"x": 0.0}, {"x": 0.5}])
    assert experiment.evaluations == []
    record = json.loads((tmp_path / "run/trial_0000/result.json").read_text())
    assert record["score"] == {"nonfinite": "nan"}


def test_maximisation_and_first_tie_selection(tmp_path):
    summary = run(tmp_path, Experiment(), [{"x": 0.1}, {"x": 0.9}, {"x": -0.9}], direction="maximize")
    assert summary["selected"].trial_id == 1


def test_existing_directory_is_not_overwritten(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "previous.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run(tmp_path, Experiment(), [{"x": 0.1}])
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -2, "0.1"])
def test_invalid_values_are_rejected(value):
    with pytest.raises(ValueError):
        space().validate({"x": value})


def test_space_rejects_missing_and_extra_keys():
    for params in ({}, {"x": 0.0, "unknown": 1}):
        with pytest.raises(ValueError):
            space().validate(params)


def test_integer_and_categorical_types_are_explicit():
    p = Parameter("steps", "int", "Steps", low=1, high=8)
    for value in (True, 1.5, 2.0):
        with pytest.raises(ValueError):
            p.validate(value)
    categorical = Parameter("mode", "categorical", "Mode", choices=(1, "one"))
    with pytest.raises(ValueError):
        categorical.validate(True)


def test_bad_space_definition():
    with pytest.raises(ValueError):
        Parameter("x", "float", "Coordinate", low=0, high=1, log=True)
    with pytest.raises(ValueError):
        SearchSpace((Parameter("x", "int", "Count", low=1, high=2),), constraints=lambda _: None)


def test_replay_uses_generated_events(tmp_path):
    run(tmp_path, Experiment(), [{"x": 0.2}])
    replay = ReplayController.from_events(tmp_path / "run/events.jsonl")
    assert replay.proposals == [{"x": 0.2}]


def test_imports_do_not_require_training_or_provider_packages():
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(source)!r}); "
        "import budgeted_search.runner, budgeted_search.space, budgeted_search.controllers.llm; "
        "import budgeted_search.controllers.hybrid, budgeted_search.controllers.optuna; "
        "assert not {'optuna', 'openai', 'numpy'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


class Array:
    def tolist(self):
        return [1.0, float("nan")]


class CountingReplay(ReplayController):
    tells = 0

    def tell(self, result):
        type(self).tells += 1
        super().tell(result)


def test_array_like_feedback_is_serialised(tmp_path):
    class ArrayExperiment(Experiment):
        def run_trial(self, parameters, context):
            return Observation(1.0, "artefact", metrics={"m": Array()}, feedback={"f": Array()})

    run(tmp_path, ArrayExperiment(), [{"x": 0.2}])
    record = json.loads((tmp_path / "run/trial_0000/result.json").read_text())
    assert record["feedback"] == {"f": [1.0, {"nonfinite": "nan"}]}
    assert record["metrics"] == {"m": [1.0, {"nonfinite": "nan"}]}


class UnserialisableFeedback(Experiment):
    def run_trial(self, parameters, context):
        if context.trial_id == 0:
            return Observation(1.0, "artefact", feedback={"bad": object()})
        return super().run_trial(parameters, context)


def test_unserialisable_feedback_is_a_failed_trial_under_continue(tmp_path):
    summary = run(tmp_path, UnserialisableFeedback(), [{"x": 0.2}, {"x": 0.3}], on_trial_error="continue")
    record = json.loads((tmp_path / "run/trial_0000/result.json").read_text())
    assert record["status"] == "failed"
    assert "Unsupported record value: object" in record["error"]
    assert summary["selected"].trial_id == 1


def test_unserialisable_feedback_aborts_with_result_written(tmp_path):
    with pytest.raises(TypeError, match="Unsupported record value"):
        run(tmp_path, UnserialisableFeedback(), [{"x": 0.2}])
    assert json.loads((tmp_path / "run/trial_0000/result.json").read_text())["status"] == "failed"
    assert (tmp_path / "run/failure.json").exists()


def test_invalid_proposal_is_a_failed_trial_and_controller_is_told(tmp_path):
    CountingReplay.tells = 0
    experiment = Experiment()
    summary = Runner(RunConfig(2, 42, "Task", on_trial_error="continue"), space()).run(
        experiment,
        CountingReplay([{"x": 5.0}, {"x": 0.5}]),
        tmp_path / "run",
        project_root=tmp_path,
        metadata={},
    )
    assert CountingReplay.tells == 2
    first = summary["trials"][0]
    assert (first.status, first.parameters) == ("failed", {"x": 5.0})
    assert "Invalid value for x" in first.error
    assert summary["selected"].trial_id == 1


def test_invalid_proposal_aborts_under_abort_policy(tmp_path):
    with pytest.raises(ValueError, match="Invalid value for x"):
        run(tmp_path, Experiment(), [{"x": 5.0}])
    assert json.loads((tmp_path / "run/trial_0000/result.json").read_text())["parameters"] == {"x": 5.0}


class FloatSubclass(float):
    pass


@pytest.mark.parametrize("score", [Fraction(1, 4), FloatSubclass(0.25), 1])
def test_any_real_score_is_accepted(tmp_path, score):
    summary = run(tmp_path, Experiment(scores=[score]), [{"x": 0.2}])
    assert summary["selected"].score == float(score)
    assert type(summary["selected"].score) is float


def test_boolean_score_is_a_failed_trial(tmp_path):
    with pytest.raises(TypeError, match="must be a number"):
        run(tmp_path, Experiment(scores=[True]), [{"x": 0.2}])


def test_context_carries_score_name_and_seed(tmp_path):
    seen = []

    class Recording(Experiment):
        def run_trial(self, parameters, context):
            seen.append((context.score_name, context.seed, context.note))
            return super().run_trial(parameters, context)

    run(tmp_path, Recording(), [{"x": 0.2}, {"x": 0.3}], score_name="validation error")
    assert seen == [("validation error", 42, ""), ("validation error", 43, "")]


def test_blank_score_name_is_rejected():
    with pytest.raises(ValueError, match="Name the score"):
        RunConfig(1, 0, "Task", score_name=" ")


def test_json_value_duck_types_array_likes():
    assert json_value({"a": Array(), "b": (1, 2)}) == {"a": [1.0, {"nonfinite": "nan"}], "b": [1, 2]}
    with pytest.raises(TypeError):
        json_value(object())


def mixed_space(*extra):
    return SearchSpace(
        (
            Parameter("x", "float", "Linear", low=0.0, high=1.0),
            Parameter("s", "float", "Log", low=1e-3, high=1.0, log=True),
            Parameter("m", "categorical", "Mode", choices=("a", "b")),
            *extra,
        )
    )


def test_mixed_distance_scales_log_and_linear_parameters():
    log_only = SearchSpace((Parameter("s", "float", "Log", low=1e-3, high=1.0, log=True),))
    assert log_only.distance({"s": 0.01}, {"s": 0.1}) == pytest.approx(1 / 3)
    linear_only = SearchSpace((Parameter("s", "float", "Linear", low=1e-3, high=1.0),))
    assert linear_only.distance({"s": 0.01}, {"s": 0.1}) == pytest.approx(0.09 / 0.999)


def test_mixed_distance_averages_numeric_and_categorical_terms():
    distance = mixed_space().distance
    a = {"x": 0.0, "s": 1e-3, "m": "a"}
    assert distance(a, {"x": 1.0, "s": 1.0, "m": "b"}) == pytest.approx(1.0)
    assert distance(a, {"x": 0.3, "s": 1e-3, "m": "a"}) == pytest.approx(0.1)
    assert distance(a, {"x": 0.0, "s": 1e-3, "m": "b"}) == pytest.approx(1 / 3)
    assert distance(a, a) == 0.0


def test_mixed_distance_clips_to_the_unit_interval():
    distance = SearchSpace((Parameter("x", "float", "Linear", low=0.0, high=1.0),)).distance
    assert distance({"x": -4.0}, {"x": 7.0}) == 1.0


def test_mixed_distance_excludes_zero_width_parameters():
    gated = Parameter("w", "float", "Gated constant", low=0.5, high=0.5)
    space = mixed_space(gated)
    a, b = {"x": 0.0, "s": 1e-3, "m": "a", "w": 0.5}, {"x": 1.0, "s": 1.0, "m": "b", "w": 0.5}
    assert space.distance(a, b) == pytest.approx(1.0)


def test_mixed_distance_is_zero_when_every_parameter_is_excluded():
    space = SearchSpace((Parameter("w", "float", "Gated constant", low=0.5, high=0.5),))
    assert space.distance({"w": 0.5}, {"w": 0.5}) == 0.0
    assert mixed_distance(space)({"w": 0.5}, {"w": 0.5}) == 0.0


def test_explicit_distance_overrides_the_default():
    explicit = SearchSpace((Parameter("x", "float", "Linear", low=0.0, high=1.0),), distance=lambda a, b: 7.0)
    assert explicit.distance({"x": 0.0}, {"x": 0.0}) == 7.0
