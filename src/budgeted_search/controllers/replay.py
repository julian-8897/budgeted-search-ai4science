"""Offline replay of user-supplied proposals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from budgeted_search.contracts import Proposal, TrialContext, TrialResult


class ReplayController:
    def __init__(self, proposals: list[dict[str, Any]]) -> None:
        self.proposals = proposals
        self.index = 0
        self.pending = False

    @classmethod
    def from_events(cls, path: Path) -> ReplayController:
        proposals = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("event") == "proposal":
                    proposals.append(event["proposal"]["parameters"])
        if not proposals:
            raise ValueError("No proposals found in the supplied event log")
        return cls(proposals)

    def ask(self, context: TrialContext, history: tuple[TrialResult, ...]) -> Proposal:
        if self.pending:
            raise RuntimeError("Previous replay proposal has not been observed")
        if self.index >= len(self.proposals):
            raise ValueError("Replay proposals exhausted")
        self.pending = True
        return Proposal(dict(self.proposals[self.index]), {"controller": "replay", "index": self.index})

    def tell(self, result: TrialResult) -> None:
        if not self.pending:
            raise RuntimeError("No pending replay proposal")
        self.pending = False
        self.index += 1
