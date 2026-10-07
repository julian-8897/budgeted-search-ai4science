"""Controller warm-start followed by TPE with shared trial observations."""

from __future__ import annotations

from dataclasses import replace

from budgeted_search.contracts import Controller, Proposal, TrialContext, TrialResult
from budgeted_search.controllers.optuna import OptunaController


class HybridController:
    def __init__(self, warmstart: Controller, tpe: OptunaController, *, warmstart_trials: int) -> None:
        if type(warmstart_trials) is not int or warmstart_trials <= 0:
            raise ValueError("Warm-start trial count must be positive")
        if tpe.sampler != "tpe":
            raise ValueError("Hybrid handoff requires TPE")
        self.warmstart = warmstart
        self.tpe = tpe
        self.warmstart_trials = warmstart_trials
        self.completed = 0
        self.pending: Controller | None = None

    def ask(self, context: TrialContext, history: tuple[TrialResult, ...]) -> Proposal:
        if self.pending is not None:
            raise RuntimeError("Previous hybrid proposal has not been observed")
        if context.trial_id == 0 and self.warmstart_trials >= context.trials_remaining:
            raise ValueError("Hybrid budget must include at least one TPE trial")
        controller = self.warmstart if self.completed < self.warmstart_trials else self.tpe
        self.pending = controller
        if controller is self.warmstart:
            context = replace(context, note=self._handoff_note())
        proposal = controller.ask(context, history)
        return Proposal(
            proposal.parameters,
            {**proposal.audit, "phase": "warmstart" if controller is self.warmstart else "tpe"},
        )

    def _handoff_note(self) -> str:
        remaining = self.warmstart_trials - self.completed
        return (
            f"This run uses an LLM warm-start for the first {self.warmstart_trials} proposals. "
            f"{remaining} LLM-controlled {'proposal remains' if remaining == 1 else 'proposals remain'} "
            "before TPE takes over."
        )

    def tell(self, result: TrialResult) -> None:
        if self.pending is None:
            raise RuntimeError("No pending hybrid proposal")
        self.pending.tell(result)
        if self.pending is self.warmstart:
            self.tpe.observe_external(result)
        self.completed += 1
        self.pending = None

    @property
    def last_audit(self) -> dict:
        return getattr(self.pending, "last_audit", {})
