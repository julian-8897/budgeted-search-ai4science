"""Random and TPE search sharing a declared parameter space."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

from budgeted_search.contracts import Proposal, TrialContext, TrialResult
from budgeted_search.space import SearchSpace


class OptunaController:
    def __init__(
        self,
        space: SearchSpace,
        *,
        sampler: Literal["random", "tpe"],
        seed: int,
        direction: Literal["minimize", "maximize"],
        sampler_options: dict[str, Any] | None = None,
        objective: Callable[[TrialResult], float | None] | None = None,
    ) -> None:
        import optuna

        if sampler not in ("random", "tpe") or direction not in ("minimize", "maximize"):
            raise ValueError("Invalid sampler or direction")
        options = dict(sampler_options or {})
        if "seed" in options:
            raise ValueError("Pass seed separately")
        sampler_class = optuna.samplers.RandomSampler if sampler == "random" else optuna.samplers.TPESampler
        self.study = optuna.create_study(direction=direction, sampler=sampler_class(seed=seed, **options))
        self.space = space
        self.sampler = sampler
        self.objective = objective or self._finite_objective
        self.pending = None
        self.distributions = {}
        for parameter in space.parameters:
            if parameter.kind == "categorical":
                distribution = optuna.distributions.CategoricalDistribution(parameter.choices)
            elif parameter.kind == "int":
                distribution = optuna.distributions.IntDistribution(parameter.low, parameter.high, log=parameter.log)
            else:
                distribution = optuna.distributions.FloatDistribution(parameter.low, parameter.high, log=parameter.log)
            self.distributions[parameter.name] = distribution

    @staticmethod
    def _finite_objective(result: TrialResult) -> float | None:
        if result.status == "complete" and result.score is not None and math.isfinite(result.score):
            return result.score
        return None

    def ask(self, context: TrialContext, history: tuple[TrialResult, ...]) -> Proposal:
        if self.pending is not None:
            raise RuntimeError("Previous Optuna proposal has not been observed")
        if self.study.direction.name.lower() != context.direction:
            raise ValueError("Runner and Optuna selection directions differ")
        self.pending = self.study.ask(fixed_distributions=self.distributions)
        return Proposal(dict(self.pending.params), {"controller": self.sampler, "optuna_trial": self.pending.number})

    def tell(self, result: TrialResult) -> None:
        import optuna

        if self.pending is None:
            raise RuntimeError("No pending Optuna proposal")
        value = self.objective(result)
        if value is not None and not math.isfinite(value):
            raise ValueError("Sampler objective must be finite or None")
        self.pending.set_user_attr("trial_status", result.status)
        self.pending.set_user_attr("error", result.error)
        if value is None:
            self.study.tell(self.pending, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(self.pending, value)
        self.pending = None

    def observe_external(self, result: TrialResult) -> None:
        """Seed a study with a completed proposal from another controller."""
        import optuna

        if self.pending is not None:
            raise RuntimeError("Cannot add an external trial while a proposal is pending")
        parameters = self.space.validate(result.parameters)
        value = self.objective(result)
        if value is not None and not math.isfinite(value):
            raise ValueError("Sampler objective must be finite or None")
        trial = optuna.trial.create_trial(
            params=parameters,
            distributions=self.distributions,
            value=value,
            state=optuna.trial.TrialState.COMPLETE if value is not None else optuna.trial.TrialState.FAIL,
            user_attrs={"phase": "warmstart", "trial_status": result.status},
        )
        self.study.add_trial(trial)
