"""An example of the public experiment interface on a synthetic objective."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from budgeted_search.contracts import Controller, Observation, TrialContext
from budgeted_search.controllers.replay import ReplayController
from budgeted_search.records import write_json
from budgeted_search.runner import RunConfig, Runner
from budgeted_search.space import Parameter, SearchSpace

REPLAY = (
    {"x": -1.0, "scale": 1.0, "mode": "plain"},
    {"x": -0.5, "scale": 0.01, "mode": "shifted"},
    {"x": 0.0, "scale": 0.3, "mode": "shifted"},
    {"x": 0.25, "scale": 0.1, "mode": "shifted"},
    {"x": 1.0, "scale": 0.001, "mode": "plain"},
)


def objective(parameters: dict) -> float:
    shift = 0.0 if parameters["mode"] == "shifted" else 0.5
    return (parameters["x"] - 0.25) ** 2 + (math.log10(parameters["scale"]) + 1) ** 2 + shift


class SyntheticExperiment:
    """Deterministic, so it ignores context.seed; a stochastic experiment must train with it."""

    def run_trial(self, parameters: dict, context: TrialContext) -> Observation:
        artefact = Path(context.output_dir) / "candidate.json"
        write_json(artefact, parameters)
        score = objective(parameters)
        return Observation(score, str(artefact), metrics={"objective": score}, feedback={"objective": score})

    def evaluate_selected(self, artefact: str) -> dict:
        return {"objective": objective(json.loads(Path(artefact).read_text()))}


def build_llm(args: argparse.Namespace, space: SearchSpace, parser: argparse.ArgumentParser):
    from budgeted_search.controllers.llm import LLMController, OpenAITransport

    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error(f"Set the {args.api_key_env} environment variable to your API key")
    if not args.model:
        parser.error("--model is required for LLM controllers")
    transport = OpenAITransport(model=args.model, api_key=key, base_url=args.base_url)
    return LLMController(space, transport, ensemble_k=args.ensemble_k)


def build_controller(args: argparse.Namespace, space: SearchSpace, parser: argparse.ArgumentParser) -> Controller:
    if args.replay_log is not None:
        if args.controller != "replay":
            parser.error("--replay-log requires --controller replay")
        return ReplayController.from_events(args.replay_log)
    if args.controller == "replay":
        if args.trials > len(REPLAY):
            parser.error(f"The built-in offline sequence has {len(REPLAY)} proposals")
        return ReplayController(list(REPLAY))
    if args.controller == "llm":
        return build_llm(args, space, parser)
    from budgeted_search.controllers.optuna import OptunaController

    if args.controller == "hybrid":
        from budgeted_search.controllers.hybrid import HybridController

        tpe = OptunaController(
            space,
            sampler="tpe",
            seed=args.seed,
            direction="minimize",
            sampler_options={"n_startup_trials": 0},
        )
        return HybridController(build_llm(args, space, parser), tpe, warmstart_trials=args.warmstart_trials)
    return OptunaController(space, sampler=args.controller, seed=args.seed, direction="minimize")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller", choices=("replay", "random", "tpe", "llm", "hybrid"), default="replay")
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--ensemble-k", type=int, default=1)
    parser.add_argument("--warmstart-trials", type=int, default=2)
    args = parser.parse_args()
    space = SearchSpace(
        (
            Parameter("x", "float", "Offset of the quadratic minimum", low=-1.0, high=1.0),
            Parameter("scale", "float", "Positive scale, best near 0.1", low=1e-3, high=1.0, log=True),
            Parameter(
                "mode", "categorical", "Variant; one value removes a constant penalty", choices=("plain", "shifted")
            ),
        )
    )
    controller = build_controller(args, space, parser)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("runs") / f"minimal-{timestamp}"
    config = RunConfig(
        args.trials,
        args.seed,
        "Minimise a synthetic objective over a coordinate, a log-scale parameter and a categorical mode",
        score_name="objective",
    )
    result = Runner(config, space).run(
        SyntheticExperiment(),
        controller,
        output,
        project_root=Path.cwd(),
        metadata={"experiment": "synthetic_objective"},
    )
    print(f"Selected trial {result['selected'].trial_id}; score={result['selected'].score}; output={output}")


if __name__ == "__main__":
    main()
