"""Export trial observations from a user-generated run to CSV without recomputing metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarise(run_dir: Path) -> Path:
    rows = []
    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event["event"] != "trial_finished":
                continue
            result = event["result"]
            score = result["score"]
            if isinstance(score, dict):
                score = score["nonfinite"]
            rows.append(
                {
                    "trial_id": result["trial_id"],
                    "seed": result["seed"],
                    "status": result["status"],
                    "score": score,
                    "wall_seconds": result["costs"].get("wall_seconds"),
                    "artefact": result["artefact"],
                    "error": result["error"],
                }
            )
    if not rows:
        raise ValueError("No trial observations in this run")
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    path = output / "trials.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(summarise(parser.parse_args().run_dir))


if __name__ == "__main__":
    main()
