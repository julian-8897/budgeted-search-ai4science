"""Strict JSON records and local execution provenance."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def json_value(value: Any) -> Any:
    """Keep non-finite values explicit without emitting invalid JSON numbers.

    Array scalars and arrays are accepted by duck typing through `tolist`.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Record keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_value(tolist())
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.touch(exist_ok=False)

    def append(self, event: str, **payload: Any) -> None:
        record = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(), **payload}
        line = json.dumps(json_value(record), allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def build_manifest(project_root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(project_root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "metadata": metadata,
    }
