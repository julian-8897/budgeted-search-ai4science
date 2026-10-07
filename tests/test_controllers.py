from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from budgeted_search.contracts import Observation, TrialContext, TrialResult
from budgeted_search.controllers.llm import (
    IncompleteEnsembleError,
    LLMController,
    OpenAITransport,
    Response,
    ServedModelChangedError,
    parse_proposal,
)
from budgeted_search.controllers.replay import ReplayController
from budgeted_search.records import json_value
from budgeted_search.runner import RunConfig, Runner
from budgeted_search.space import Parameter, SearchSpace

ROOT = Path(__file__).resolve().parents[1]


def context(index=0, **options):
    return TrialContext(index, 42 + index, 5 - index, "/tmp", "Synthetic task", **options)


def result(index, x, score):
    return TrialResult(index, 42 + index, {"x": x}, "complete", score, f"trial-{index}")


def space():
    return SearchSpace((Parameter("x", "float", "Coordinate", low=0.0, high=1.0),))


def full_space():
    return SearchSpace(
        (
            Parameter("x", "float", "Coordinate", low=0.0, high=1.0),
            Parameter("depth", "int", "Layers", low=1, high=8),
            Parameter("mode", "categorical", "Variant", choices=("plain", "shifted")),
        )
    )


def reply(x=0.5, **extra):
    return json.dumps({"x": x, **extra})


def sequence(*contents, model="m", fingerprint="fp"):
    """A thread-safe transport returning scripted contents (or raising scripted exceptions) in call order."""
    lock, items, calls = threading.Lock(), iter(contents), []

    def transport(messages):
        with lock:
            item = next(items)
            calls.append(messages)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, Response) else Response(item, model, fingerprint)

    transport.calls = calls
    return transport


def controller(transport, **options):
    options.setdefault("sleep", lambda seconds: None)
    return LLMController(options.pop("space", space()), transport, **options)


@pytest.mark.parametrize(
    "content",
    [
        '{"x": 0.5, "reasoning": "r", "strategy": "exploit"}',
        '```json\n{"x": 0.5, "reasoning": "r", "strategy": "exploit"}\n```',
        'Here is my proposal:\n{"reasoning": "r", "strategy": "exploit", "x": 0.5}\nHope it helps.',
        'Sure ```json\n{"reasoning": "a } brace in a string", "x": 0.5}``` thanks',
    ],
)
def test_parse_accepts_fenced_and_wrapped_json(content):
    parameters, reasoning, strategy = parse_proposal(content, space())
    assert parameters == {"x": 0.5}
    assert strategy in ("exploit", "explore")
    assert reasoning


def test_parse_defaults_reasoning_and_strategy():
    assert parse_proposal(reply(), space()) == ({"x": 0.5}, "", "explore")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (reply(extra=1), "extra"),
        ("{}", "missing"),
        (reply(strategy="wander"), "strategy"),
        (reply(reasoning=3), "reasoning"),
        (reply(x=1.5), "Invalid value for x"),
        ("[0.5]", "JSON object"),
        ("no json here", "no JSON object"),
        ('{"x": 0.5', "unterminated"),
        ("", "no JSON object"),
    ],
)
def test_parse_rejects_malformed_proposals(content, message):
    with pytest.raises(ValueError, match=message):
        parse_proposal(content, space())


def test_parse_converts_integral_floats_for_integer_parameters():
    parameters, _, _ = parse_proposal('{"x": 0.5, "depth": 3.0, "mode": "plain"}', full_space())
    assert parameters["depth"] == 3 and type(parameters["depth"]) is int
    for depth in ("3.5", "9", "9.0", "true"):
        with pytest.raises(ValueError, match="Invalid value for depth"):
            parse_proposal(f'{{"x": 0.5, "depth": {depth}, "mode": "plain"}}', full_space())


def test_reserved_parameter_names_are_rejected():
    reserved = SearchSpace((Parameter("strategy", "categorical", "Clash", choices=("a",)),))
    with pytest.raises(ValueError, match="reserved"):
        LLMController(reserved, sequence())


def test_first_valid_response_with_audit_and_no_credentials():
    transport = sequence(Response(reply(reasoning="why", strategy="exploit"), "model", "fp", {"tokens": 3}, "thinking"))
    proposal = controller(transport).ask(context(), ())
    assert proposal.parameters == {"x": 0.5}
    audit = json_value(proposal.audit)
    json.dumps(audit)
    assert audit["controller"] == "llm" and audit["messages"][0]["role"] == "system"
    attempt = audit["members"][0][0]
    assert (attempt["outcome"], attempt["reasoning"], attempt["usage"]) == ("valid", "thinking", {"tokens": 3})
    assert (audit["reasoning"], audit["strategy"]) == ("why", "exploit")
    assert audit["selection"]["survivors"] == 1 and audit["identity"] == {"model": "model", "fingerprint": "fp"}


def test_thinking_traces_are_never_reinjected():
    transport = sequence(Response(reply(reasoning="digest"), "m", "fp", {}, "SECRET THINKING"), reply(0.6))
    llm = controller(transport)
    first = llm.ask(context(), ())
    llm.tell(result(0, 0.5, 1.0))
    llm.ask(context(1), (result(0, 0.5, 1.0),))
    second_prompt = transport.calls[1][1]["content"]
    assert "digest" in second_prompt and "Trial 0 [explore]" in second_prompt
    assert "SECRET THINKING" not in json.dumps(transport.calls[1]) and first.parameters == {"x": 0.5}


def test_transport_exception_then_success_waits_once():
    waits = []
    llm = controller(sequence(ConnectionError("down"), reply()), sleep=waits.append)
    llm.ask(context(), ())
    assert waits == [1.0]
    assert [a["outcome"] for a in llm.last_audit["members"][0]] == ["error", "valid"]
    assert llm.last_audit["members"][0][0]["error"] == "ConnectionError: down"


@pytest.mark.parametrize("bad", ["not json", "", reply(x=2.0), reply(extra=1)])
def test_invalid_content_is_retried(bad):
    llm = controller(sequence(bad, reply(0.3)))
    assert llm.ask(context(), ()).parameters == {"x": 0.3}


def test_backoff_is_exponential_and_capped():
    waits = []
    llm = controller(sequence(*["bad"] * 4), max_attempts=4, retry_wait=(1.0, 3.0), sleep=waits.append)
    with pytest.raises(RuntimeError, match="No ensemble member"):
        llm.ask(context(), ())
    assert waits == [1.0, 2.0, 3.0]
    assert len(llm.last_audit["members"][0]) == 4


def test_exhausted_member_is_dropped_and_recorded():
    transport = sequence(reply(0.4), *["bad"] * 3, reply(0.5))
    llm = controller(transport, ensemble_k=3, parallel=False)
    proposal = llm.ask(context(), ())
    selection = proposal.audit["selection"]
    assert (selection["requested"], selection["survivors"], selection["members"]) == (3, 2, [0, 2])
    assert [len(attempts) for attempts in proposal.audit["members"]] == [1, 3, 1]
    assert proposal.audit["members"][1][-1]["outcome"] == "error"


def test_parallel_ensemble_survives_a_failing_member():
    transport = sequence(reply(0.4), reply(0.5), *["bad"] * 3)
    llm = controller(transport, ensemble_k=3, parallel=True)
    proposal = llm.ask(context(), ())
    assert proposal.audit["selection"]["survivors"] == 2


def test_require_complete_raises_when_a_member_is_dropped():
    llm = controller(sequence(reply(), *["bad"] * 3), ensemble_k=2, require_complete=True, parallel=False)
    with pytest.raises(IncompleteEnsembleError):
        llm.ask(context(), ())
    assert len(llm.last_audit["members"][1]) == 3


def test_all_members_failing_raises_runtime_error():
    llm = controller(sequence(*["bad"] * 6), ensemble_k=2, parallel=False)
    with pytest.raises(RuntimeError, match="No ensemble member"):
        llm.ask(context(), ())
    assert not llm.pending


def test_model_change_aborts_without_retry():
    transport = sequence(Response(reply(), "a", "fp"), Response("not json", "b", "fp"), reply())
    llm = controller(transport, ensemble_k=2, parallel=False)
    with pytest.raises(ServedModelChangedError, match="model changed"):
        llm.ask(context(), ())
    assert len(transport.calls) == 2
    assert [a["model"] for member in llm.last_audit["members"] for a in member] == ["a", "b"]


def test_fingerprint_change_across_asks_aborts_without_retry():
    transport = sequence(Response(reply(), "a", "one"), Response(reply(), "a", "two"), reply())
    llm = controller(transport)
    llm.ask(context(), ())
    llm.tell(result(0, 0.5, 1.0))
    with pytest.raises(ServedModelChangedError):
        llm.ask(context(1), (result(0, 0.5, 1.0),))
    assert len(transport.calls) == 2


def test_missing_identity_does_not_lock():
    transport = sequence(Response(reply(), None, None), Response(reply(), "a", None), Response(reply(), "b", None))
    llm = controller(transport)
    llm.ask(context(), ())
    assert llm.identity == (None, None)
    llm.tell(result(0, 0.5, 1.0))
    llm.ask(context(1), ())
    assert llm.identity == ("a", None)
    llm.tell(result(1, 0.5, 1.0))
    with pytest.raises(ServedModelChangedError):
        llm.ask(context(2), ())


@pytest.mark.parametrize("parallel", [False, True])
def test_medoid_selects_the_central_candidate(parallel):
    llm = controller(sequence(reply(0.1), reply(0.2), reply(0.9)), ensemble_k=3, parallel=parallel)
    proposal = llm.ask(context(), ())
    assert proposal.parameters == {"x": 0.2}
    selection = proposal.audit["selection"]
    assert sorted(a["parameters"]["x"] for a in proposal.audit["candidates"]) == [0.1, 0.2, 0.9]
    assert selection["max_pairwise_distance"] == pytest.approx(0.8)
    assert selection["mean_pairwise_distance"] == pytest.approx((0.1 + 0.7 + 0.8) / 3)
    assert min(selection["distance_sums"]) == pytest.approx(0.1 + 0.7)


@pytest.mark.parametrize("parallel", [False, True])
def test_medoid_ties_pick_the_lowest_member_index(parallel):
    llm = controller(sequence(reply(0.2), reply(0.8)), ensemble_k=2, parallel=parallel)
    proposal = llm.ask(context(), ())
    candidates = {c["member"]: c["parameters"] for c in proposal.audit["candidates"]}
    assert proposal.audit["selection"]["selected_member"] == 0
    assert proposal.parameters == candidates[0]


def test_medoid_uses_an_explicit_distance_and_rejects_invalid_ones():
    explicit = SearchSpace(space().parameters, distance=lambda a, b: abs(a["x"] - b["x"]) ** 0.5)
    assert controller(sequence(reply(0.1), reply(0.5), reply(0.6)), space=explicit, ensemble_k=3, parallel=False).ask(
        context(), ()
    ).parameters == {"x": 0.5}
    broken = SearchSpace(space().parameters, distance=lambda a, b: float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        controller(sequence(reply(0.1), reply(0.5)), space=broken, ensemble_k=2, parallel=False).ask(context(), ())


def test_ask_tell_protocol_is_enforced():
    llm = controller(sequence(reply(), reply()))
    with pytest.raises(RuntimeError, match="No pending"):
        llm.tell(result(0, 0.5, 1.0))
    llm.ask(context(), ())
    with pytest.raises(RuntimeError, match="not been observed"):
        llm.ask(context(), ())


@pytest.mark.parametrize(
    "options",
    [{"ensemble_k": 0}, {"max_attempts": 0}, {"retry_wait": (5.0, 1.0)}, {"max_list_items": 1}],
)
def test_invalid_controller_options_are_rejected(options):
    with pytest.raises(ValueError):
        LLMController(space(), sequence(), **options)


def hybrid_parts(tpe_options=None):
    from budgeted_search.controllers.hybrid import HybridController
    from budgeted_search.controllers.optuna import OptunaController

    tpe = OptunaController(
        space(),
        sampler="tpe",
        seed=42,
        direction="minimize",
        sampler_options={"n_startup_trials": 0, **(tpe_options or {})},
    )
    return HybridController, tpe


def test_hybrid_seeds_tpe_with_every_warmstart_trial():
    pytest.importorskip("optuna")
    hybrid_class, tpe = hybrid_parts()
    hybrid = hybrid_class(ReplayController([{"x": 0.1}, {"x": 0.9}]), tpe, warmstart_trials=2)
    history = []
    for index in range(3):
        proposal = hybrid.ask(context(index), tuple(history))
        observation = result(index, proposal.parameters["x"], float(index + 1))
        hybrid.tell(observation)
        history.append(observation)
    assert len(tpe.study.trials) == 3
    assert [t.params for t in tpe.study.trials[:2]] == [{"x": 0.1}, {"x": 0.9}]
    assert [t.value for t in tpe.study.trials] == [1.0, 2.0, 3.0]


def test_hybrid_passes_a_handoff_note_during_warmstart_only():
    pytest.importorskip("optuna")
    hybrid_class, tpe = hybrid_parts()
    notes = {"warmstart": [], "tpe": []}

    class Spy(ReplayController):
        def ask(self, context, history):
            notes["warmstart"].append(context.note)
            return super().ask(context, history)

    original = tpe.ask

    def tpe_ask(ctx, history):
        notes["tpe"].append(ctx.note)
        return original(ctx, history)

    tpe.ask = tpe_ask
    hybrid = hybrid_class(Spy([{"x": 0.1}, {"x": 0.9}]), tpe, warmstart_trials=2)
    for index in range(3):
        hybrid.tell(result(index, hybrid.ask(context(index), ()).parameters["x"], float(index)))
    prefix = "This run uses an LLM warm-start for the first 2 proposals. "
    assert notes["warmstart"] == [
        prefix + "2 LLM-controlled proposals remain before TPE takes over.",
        prefix + "1 LLM-controlled proposal remains before TPE takes over.",
    ]
    assert notes["tpe"] == [""]


def test_hybrid_note_reaches_the_llm_prompt():
    pytest.importorskip("optuna")
    hybrid_class, tpe = hybrid_parts()
    transport = sequence(reply(0.3), reply(0.6))
    hybrid = hybrid_class(controller(transport), tpe, warmstart_trials=2)
    proposal = hybrid.ask(context(0), ())
    assert "LLM warm-start for the first 2 proposals" in proposal.audit["messages"][1]["content"]
    assert proposal.audit["phase"] == "warmstart"
    hybrid.tell(result(0, 0.3, 1.0))
    second = hybrid.ask(context(1), (result(0, 0.3, 1.0),))
    assert "1 LLM-controlled proposal remains" in second.audit["messages"][1]["content"]


def test_optuna_failure_remains_a_failure():
    optuna = pytest.importorskip("optuna")
    from budgeted_search.controllers.optuna import OptunaController

    optuna_controller = OptunaController(space(), sampler="random", seed=1, direction="minimize")
    proposal = optuna_controller.ask(context(), ())
    optuna_controller.tell(TrialResult(0, 42, proposal.parameters, "failed", None, None, error="synthetic failure"))
    assert optuna_controller.study.trials[0].state == optuna.trial.TrialState.FAIL


def test_optuna_rejects_mismatched_selection_direction():
    pytest.importorskip("optuna")
    from budgeted_search.controllers.optuna import OptunaController

    optuna_controller = OptunaController(space(), sampler="random", seed=1, direction="maximize")
    with pytest.raises(ValueError, match="directions differ"):
        optuna_controller.ask(context(), ())


def fake_openai_client(reasoning=None):
    message = SimpleNamespace(content=reply())
    if reasoning is not None:
        message.reasoning_content = reasoning
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        model="served-model",
        system_fingerprint="fp-1",
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 7}),
    )
    client = mock.MagicMock()
    client.chat.completions.create.return_value = completion
    return client


def test_openai_transport_captures_reasoning_and_hides_the_key():
    pytest.importorskip("openai")
    client = fake_openai_client(reasoning="thinking trace")
    with mock.patch("openai.OpenAI", return_value=client) as factory:
        transport = OpenAITransport(model="m", api_key="sk-secret-key")
    assert factory.call_args.kwargs["max_retries"] == 0
    response = transport([{"role": "user", "content": "hi"}])
    assert response == Response(reply(), "served-model", "fp-1", {"total_tokens": 7}, "thinking trace")
    assert "sk-secret-key" not in json.dumps(json_value(response))
    assert "sk-secret-key" not in json.dumps(client.chat.completions.create.call_args.kwargs)
    with mock.patch("openai.OpenAI", return_value=fake_openai_client()):
        assert OpenAITransport(model="m", api_key="k")([]).reasoning is None
    with pytest.raises(ValueError, match="reserved"):
        OpenAITransport(model="m", api_key="k", request_options={"messages": []})


def test_run_directory_never_contains_the_api_key(tmp_path):
    pytest.importorskip("openai")
    with mock.patch("openai.OpenAI", return_value=fake_openai_client(reasoning="trace")):
        transport = OpenAITransport(model="m", api_key="sk-secret-key")

    class Quadratic:
        def run_trial(self, parameters, context):
            return Observation(parameters["x"], "artefact", feedback={"loss": parameters["x"]})

        def evaluate_selected(self, artefact):
            return {"held_out": 0.0}

    Runner(RunConfig(2, 42, "Task"), space()).run(
        Quadratic(), controller(transport, ensemble_k=2), tmp_path / "run", project_root=tmp_path, metadata={}
    )
    files = [path for path in (tmp_path / "run").rglob("*") if path.is_file()]
    assert files and not any("sk-secret-key" in path.read_text() for path in files)
    assert any("trace" in path.read_text() for path in files if path.name == "events.jsonl")


def run_example(tmp_path, *args, check=True, env=None):
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment.pop("LLM_API_KEY", None)
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "examples/minimal/run.py"), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        check=check,
    )


def test_example_replays_offline(tmp_path):
    completed = run_example(tmp_path, "--output", str(tmp_path / "replay"))
    assert "Selected trial 3; score=0.0" in completed.stdout
    summary = json.loads((tmp_path / "replay/summary.json").read_text())
    assert summary["test_metrics"] == {"objective": 0.0}


@pytest.mark.parametrize("sampler", ["random", "tpe"])
def test_example_runs_optuna_controllers(tmp_path, sampler):
    pytest.importorskip("optuna")
    run_example(tmp_path, "--controller", sampler, "--trials", "4", "--output", str(tmp_path / sampler))
    assert (tmp_path / sampler / "summary.json").exists()


@pytest.mark.parametrize("name", ["llm", "hybrid"])
def test_example_requires_an_api_key_for_llm_controllers(tmp_path, name):
    pytest.importorskip("optuna")
    completed = run_example(
        tmp_path, "--controller", name, "--model", "m", "--output", str(tmp_path / "x"), check=False
    )
    assert completed.returncode != 0 and "LLM_API_KEY" in completed.stderr
