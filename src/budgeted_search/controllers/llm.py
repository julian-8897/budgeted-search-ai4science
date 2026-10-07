"""Language-model controller: full-history prompts, bounded retries, ensemble medoid and identity lock."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from budgeted_search.contracts import Proposal, TrialContext, TrialResult
from budgeted_search.controllers.prompting import PromptTemplates, build_messages
from budgeted_search.space import SearchSpace

STRATEGIES = ("explore", "exploit")
RESERVED_KEYS = ("reasoning", "strategy")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class ServedModelChangedError(RuntimeError):
    """The provider served a different model or fingerprint than earlier responses."""


class IncompleteEnsembleError(RuntimeError):
    """At least one ensemble member exhausted its attempts while `require_complete` was set."""


@dataclass(frozen=True)
class Response:
    content: str
    model: str | None
    fingerprint: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None


def _first_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("Response contains no JSON object")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Response contains an unterminated JSON object")


def parse_proposal(content: str, space: SearchSpace) -> tuple[dict[str, Any], str, str]:
    """Return validated parameters, reasoning and strategy; out-of-range values are rejected, never clamped."""
    text = content.strip()
    fenced = _FENCE.match(text)
    text = fenced.group(1) if fenced else text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_first_object(text))
    if not isinstance(payload, dict):
        raise ValueError("Response must be a JSON object")
    reasoning = payload.pop("reasoning", "")
    strategy = payload.pop("strategy", "explore")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    for parameter in space.parameters:
        value = payload.get(parameter.name)
        if parameter.kind == "int" and isinstance(value, float) and value.is_integer():
            payload[parameter.name] = int(value)
    return space.validate(payload), reasoning, strategy


class LLMController:
    """Propose configurations from a transport, which must be thread-safe when members are sampled in parallel."""

    def __init__(
        self,
        space: SearchSpace,
        transport: Callable[[list[dict[str, str]]], Response],
        *,
        ensemble_k: int = 1,
        templates: PromptTemplates | None = None,
        max_attempts: int = 3,
        retry_wait: tuple[float, float] = (1.0, 60.0),
        sleep: Callable[[float], None] = time.sleep,
        require_complete: bool = False,
        max_list_items: int = 10,
        parallel: bool = True,
    ) -> None:
        if type(ensemble_k) is not int or ensemble_k <= 0:
            raise ValueError("Ensemble size must be positive")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("Attempts per member must be positive")
        if len(retry_wait) != 2 or not 0 <= retry_wait[0] <= retry_wait[1]:
            raise ValueError("Retry wait must be (min, max) seconds with 0 <= min <= max")
        if type(max_list_items) is not int or max_list_items < 2:
            raise ValueError("Lists must show at least two items")
        if {p.name for p in space.parameters} & set(RESERVED_KEYS):
            raise ValueError(f"Parameter names {RESERVED_KEYS} are reserved for the response format")
        self.space, self.transport = space, transport
        self.ensemble_k = ensemble_k
        self.templates = templates if templates is not None else PromptTemplates.default()
        self.max_attempts, self.retry_wait, self.sleep = max_attempts, retry_wait, sleep
        self.require_complete, self.max_list_items, self.parallel = require_complete, max_list_items, parallel
        self.last_audit: dict[str, Any] = {}
        self.pending = False
        self._identity: dict[str, str | None] = {"model": None, "fingerprint": None}
        self._lock = threading.Lock()
        self._records: dict[int, dict[str, str]] = {}

    @property
    def identity(self) -> tuple[str | None, str | None]:
        return self._identity["model"], self._identity["fingerprint"]

    def ask(self, context: TrialContext, history: tuple[TrialResult, ...]) -> Proposal:
        if self.pending:
            raise RuntimeError("Previous LLM proposal has not been observed")
        messages = build_messages(
            self.templates, self.space, context, history, self._records, max_list_items=self.max_list_items
        )
        members: list[list[dict[str, Any]]] = [[] for _ in range(self.ensemble_k)]
        self.last_audit = {"controller": "llm", "messages": messages, "members": members}
        if self.parallel and self.ensemble_k > 1:
            with ThreadPoolExecutor(max_workers=self.ensemble_k) as pool:
                futures = [pool.submit(self._run_member, messages, attempts) for attempts in members]
                outcomes = [future.result() for future in futures]
        else:
            outcomes = [self._run_member(messages, attempts) for attempts in members]
        survivors = [(index, outcome) for index, outcome in enumerate(outcomes) if outcome is not None]
        self.last_audit["candidates"] = [
            {"member": index, "parameters": parameters, "reasoning": reasoning, "strategy": strategy}
            for index, (parameters, reasoning, strategy) in survivors
        ]
        dropped = self.ensemble_k - len(survivors)
        if dropped and self.require_complete:
            raise IncompleteEnsembleError(f"{dropped} of {self.ensemble_k} ensemble members exhausted their attempts")
        if not survivors:
            raise RuntimeError("No ensemble member produced a valid proposal")
        selected, statistics = self._medoid([outcome[0] for _, outcome in survivors])
        member, (parameters, reasoning, strategy) = survivors[selected]
        self.last_audit.update(
            selection={
                "requested": self.ensemble_k,
                "survivors": len(survivors),
                "members": [index for index, _ in survivors],
                "selected_member": member,
                **statistics,
            },
            reasoning=reasoning,
            strategy=strategy,
            identity={"model": self.identity[0], "fingerprint": self.identity[1]},
        )
        self._records[context.trial_id] = {"reasoning": reasoning, "strategy": strategy}
        self.pending = True
        return Proposal(parameters, dict(self.last_audit))

    def tell(self, result: TrialResult) -> None:
        if not self.pending:
            raise RuntimeError("No pending LLM proposal")
        self.pending = False

    def _run_member(
        self, messages: list[dict[str, str]], attempts: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str, str] | None:
        for attempt in range(1, self.max_attempts + 1):
            record: dict[str, Any] = {"attempt": attempt, "outcome": "error", "error": None}
            attempts.append(record)
            try:
                response = self.transport(messages)
                record.update(
                    content=response.content,
                    model=response.model,
                    fingerprint=response.fingerprint,
                    usage=response.usage,
                    reasoning=response.reasoning,
                )
                self._observe_identity(response)
                outcome = parse_proposal(response.content, self.space)
            except ServedModelChangedError as exc:
                record["error"] = str(exc)
                raise
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts:
                    low, high = self.retry_wait
                    self.sleep(min(high, low * 2 ** (attempt - 1)))
                continue
            record["outcome"] = "valid"
            return outcome
        return None

    def _observe_identity(self, response: Response) -> None:
        # A missing field establishes nothing; only the first non-None value of each field is locked.
        with self._lock:
            for name, value in (("model", response.model), ("fingerprint", response.fingerprint)):
                if value is None:
                    continue
                if self._identity[name] is None:
                    self._identity[name] = value
                elif self._identity[name] != value:
                    raise ServedModelChangedError(f"Served {name} changed: {self._identity[name]!r} -> {value!r}")

    def _medoid(self, candidates: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        count = len(candidates)
        sums = [0.0] * count
        pairwise = []
        for i, candidate in enumerate(candidates):
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                distance = self.space.distance(candidate, other)
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("Proposal distance must be finite and non-negative")
                sums[i] += distance
                if i < j:
                    pairwise.append(distance)
        statistics = {
            "distance_sums": sums,
            "mean_pairwise_distance": sum(pairwise) / len(pairwise) if pairwise else 0.0,
            "max_pairwise_distance": max(pairwise, default=0.0),
        }
        return min(range(count), key=sums.__getitem__), statistics


class OpenAITransport:
    """Optional SDK integration. Credentials remain on the client, outside run records."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        request_options: dict[str, Any] | None = None,
        max_retries: int = 0,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        if not model or not api_key:
            raise ValueError("Model and API key are required")
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)
        self.model = model
        self.options = dict(request_options or {})
        if set(self.options) & {"model", "messages", "api_key", "base_url", "stream"}:
            raise ValueError("Request options contain reserved fields")

    def __call__(self, messages: list[dict[str, str]]) -> Response:
        response = self.client.chat.completions.create(model=self.model, messages=messages, **self.options)
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("Provider returned no response content")
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        return Response(
            message.content,
            response.model,
            response.system_fingerprint,
            response.usage.model_dump() if response.usage is not None else {},
            reasoning if isinstance(reasoning, str) else None,
        )
