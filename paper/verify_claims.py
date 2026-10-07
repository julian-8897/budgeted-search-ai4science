"""Verify reported numbers against released run records.

CLI: python verify_claims.py --records PATH

The script recomputes the paper's reported values from the released run records
and compares them with the values recorded in expected.json. It exits with
status 1 if any checked claim is outside tolerance, if a computation error
occurs, or if expected.json contains a claim with no computed result.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

SEEDS = (42, 43, 44)

PERFORMANCE_BATCHES = (
    ("2026-07-19_within_family_repaired_panel", "Within-family", 6),
    ("2026-07-20_xpde_repaired_panel", "Cross-family", 3),
    ("2026-07-28_reverse_xpde_panel", "Cross-family", 3),
)

COLDSTART_CAMPAIGN = "2026-07-03_coldstart_lr_prior"
WITHIN_SHAM_REPLAY = "2026-07-26_within_family_sham_replay"
WITHIN_FEEDBACK_REPLAY = "2026-07-22_within_family_feedback_replay"
CROSS_REPLAY_DIRS = (
    "2026-07-21_xpde_feedback_replay_nonidentity",
    "2026-07-30_reverse_xpde_feedback_replay",
)


def _norm_log10(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    safe_value = max(float(value), lo)
    return float(np.clip((math.log10(safe_value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)), 0.0, 1.0))


def _norm_weight_decay(value: float) -> float:
    hi = 0.01
    return float(np.clip(math.log1p(max(float(value), 0.0) * 1000.0) / math.log1p(hi * 1000.0), 0.0, 1.0))


def _norm_linear(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def _numeric_vector(cfg: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            _norm_log10(cfg["lr_base"], 1e-5, 1e-2),
            _norm_log10(cfg["lr_mult_lift"], 0.5, 2.0),
            _norm_log10(cfg["lr_mult_spectral"], 0.5, 2.0),
            _norm_log10(cfg["lr_mult_bypass"], 0.5, 2.0),
            _norm_log10(cfg["lr_mult_proj"], 0.5, 2.0),
            _norm_weight_decay(cfg["weight_decay"]),
            _norm_linear(cfg["conservation_loss_weight"], 0.0, 1.0),
            _norm_linear(cfg["semigroup_loss_weight"], 0.0, 1.0),
        ],
        dtype=float,
    )


def _categorical_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(
        (a["optimizer"] != b["optimizer"])
        + (a["scheduler"] != b["scheduler"])
        + (a["multistep_k"] != b["multistep_k"])
        + (a["loss_type"] != b["loss_type"])
    )


def action_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Mean absolute difference over 12 action coordinates (paper Equation 1)."""
    numeric = np.abs(_numeric_vector(a) - _numeric_vector(b))
    categorical = _categorical_distance(a, b)
    return float((numeric.sum() + categorical) / 12.0)


def normalise_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure a config dict contains every key action_distance expects."""
    out = dict(cfg)
    if "lr_base" not in out and "lr" in out:
        out["lr_base"] = out["lr"]
    out.setdefault("lr_base", 1e-3)
    out.setdefault("lr_mult_lift", 1.0)
    out.setdefault("lr_mult_spectral", 1.0)
    out.setdefault("lr_mult_bypass", 1.0)
    out.setdefault("lr_mult_proj", 1.0)
    out.setdefault("weight_decay", 1e-4)
    out.setdefault("conservation_loss_weight", 0.0)
    out.setdefault("semigroup_loss_weight", 0.0)
    out.setdefault("optimizer", "adamw")
    out.setdefault("scheduler", "none")
    out.setdefault("multistep_k", 1)
    out.setdefault("loss_type", "mse")
    return out


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _normalise_method(method: str) -> str:
    return "bayesian" if method in {"bayesian", "optuna"} else method


def _batch_for_cell(records: Path, cell: str) -> str:
    for batch_name, _, _ in PERFORMANCE_BATCHES:
        if (records / "batches" / batch_name / "alga" / cell).is_dir():
            return batch_name
    raise ValueError(f"no batch contains cell {cell}")


def load_performance(records: Path) -> list[dict]:
    """Return one record per cell with matched endpoints and the random pool."""
    cells: list[dict] = []
    for batch_name, _family, expected_cells in PERFORMANCE_BATCHES:
        batch = records / "batches" / batch_name
        names = sorted(path.name for path in (batch / "alga").iterdir() if path.is_dir())
        if len(names) != expected_cells:
            raise ValueError(f"expected {expected_cells} cells in {batch_name}; found {len(names)}")

        for name in names:
            llm_test: dict[int, float] = {}
            first_validation: dict[int, float] = {}
            for run in sorted((batch / "alga" / name).iterdir()):
                final_path = run / "final_results.json"
                history_path = run / "agent_history.json"
                if not final_path.exists() or not history_path.exists():
                    continue
                final = _load_json(final_path)
                history = _load_json(history_path)
                seed = int(final["config"]["seed"])
                llm_test[seed] = float(final["test_metrics"]["nrmse_clamped"])
                first_validation[seed] = float(history[0]["val_nrmse"])

            baseline_test: dict[str, dict[int, float]] = defaultdict(dict)
            random_pool: list[float] = []
            for path in sorted((batch / "benchmarks" / name).glob("*/*_results.json")):
                result = _load_json(path)
                method = _normalise_method(str(result["method"]))
                if method not in {"random", "bayesian"}:
                    continue
                seed = int(result["seed"])
                baseline_test[method][seed] = float(result["test"]["nrmse_clamped"])
                if method == "random":
                    random_pool.extend(float(row["metrics"]["val_nrmse"]) for row in result["all_iterations"])

            expected = set(SEEDS)
            if set(llm_test) != expected or set(first_validation) != expected:
                raise ValueError(f"incomplete LLM seeds for {batch_name}/{name}")
            if len(random_pool) != 60:
                raise ValueError(f"expected 60 random configs for {batch_name}/{name}; found {len(random_pool)}")
            for method in ("random", "bayesian"):
                if set(baseline_test[method]) != expected:
                    raise ValueError(f"incomplete {method} seeds for {batch_name}/{name}")

            cells.append(
                {
                    "cell": name,
                    "llm_test": llm_test,
                    "first_validation": first_validation,
                    "random_test": dict(baseline_test["random"]),
                    "tpe_test": dict(baseline_test["bayesian"]),
                    "random_pool": random_pool,
                }
            )
    return cells


def verify_endpoint_table(cells: list[dict]) -> list[dict]:
    """Recompute Table 2 medians and win counts from raw held-out test values."""
    out: list[dict] = []
    for cell_record in cells:
        cell = cell_record["cell"]
        for method, key in (("llm", "llm_test"), ("random", "random_test"), ("tpe", "tpe_test")):
            raw_values = [cell_record[key][s] for s in SEEDS]
            out.append(
                {
                    "claim": f"endpoint_median_{cell}_{method}",
                    "recomputed": float(median(raw_values)),
                    "unit": "nrmse_clamped",
                }
            )

    random_ratios = [cell["random_test"][s] / cell["llm_test"][s] for cell in cells for s in SEEDS]
    tpe_ratios = [cell["tpe_test"][s] / cell["llm_test"][s] for cell in cells for s in SEEDS]
    out.append(
        {
            "claim": "llm_vs_random_wins",
            "recomputed": f"{sum(r > 1 for r in random_ratios)}/{len(random_ratios)}",
            "unit": "count",
        }
    )
    out.append(
        {
            "claim": "llm_vs_tpe_wins",
            "recomputed": f"{sum(r > 1 for r in tpe_ratios)}/{len(tpe_ratios)}",
            "unit": "count",
        }
    )
    return out


def verify_first_proposal(cells: list[dict]) -> list[dict]:
    out: list[dict] = []
    all_percentiles: list[float] = []
    for cell_record in cells:
        cell = cell_record["cell"]
        pool = cell_record["random_pool"]
        for s in SEEDS:
            value = cell_record["first_validation"][s]
            worse = sum(item > value for item in pool)
            equal = sum(item == value for item in pool)
            pct = 100 * (worse + 0.5 * equal) / 60.0
            all_percentiles.append(pct)
            out.append(
                {
                    "claim": f"first_proposal_pct_{cell}_seed{s}",
                    "recomputed": pct,
                    "unit": "% random pool beaten",
                }
            )

    out.append(
        {
            "claim": "first_proposal_median_pct",
            "recomputed": float(median(all_percentiles)),
            "unit": "%",
        }
    )
    out.append(
        {
            "claim": "first_proposal_at_least_75pct",
            "recomputed": f"{sum(p >= 75 for p in all_percentiles)}/{len(all_percentiles)}",
            "unit": "count",
        }
    )
    out.append(
        {
            "claim": "first_proposal_at_least_50pct",
            "recomputed": f"{sum(p >= 50 for p in all_percentiles)}/{len(all_percentiles)}",
            "unit": "count",
        }
    )
    return out


def _coldstart_rows(records: Path) -> list[dict]:
    path = records / "campaigns" / COLDSTART_CAMPAIGN / "permutation_distances.csv"
    rows = _load_csv(path)
    parsed: list[dict] = []
    for row in rows:
        cfg = ast.literal_eval(row["proposed_config"])
        parsed.append(
            {
                "cell": row["cell"],
                "desc_arm": row["desc_arm"],
                "config": cfg,
                "lr": float(cfg["lr_base"]),
            }
        )
    return parsed


def _pool_by_shown_description(rows: list[dict]) -> tuple[list[float], list[float]]:
    adv_lrs: list[float] = []
    burg_lrs: list[float] = []
    for r in rows:
        if r["cell"] == "advection_beta4.0_fixed":
            if r["desc_arm"] == "true":
                adv_lrs.append(r["lr"])
            elif r["desc_arm"] == "wrong":
                burg_lrs.append(r["lr"])
        elif r["cell"] == "burgers_nu0.001_fixed":
            if r["desc_arm"] == "wrong":
                adv_lrs.append(r["lr"])
            elif r["desc_arm"] == "true":
                burg_lrs.append(r["lr"])
    return adv_lrs, burg_lrs


def _mann_whitney_u_and_exact_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """One-sided MW U (x expected larger than y) and exact permutation p with midranks."""
    pooled = sorted(x + y)
    counts: dict[float, int] = defaultdict(int)
    for v in pooled:
        counts[v] += 1

    rank = 1
    midranks: dict[float, float] = {}
    for v in sorted(counts):
        n = counts[v]
        midranks[v] = rank + (n - 1) / 2.0
        rank += n

    n1 = len(x)
    r1 = sum(midranks[v] for v in x)
    u_obs = r1 - n1 * (n1 + 1) / 2.0

    categories = sorted(counts)
    total_weight = 0.0
    hit_weight = 0.0

    def recurse(i: int, remaining_n1: int, current_r1: float, weight: int) -> None:
        nonlocal total_weight, hit_weight
        if i == len(categories):
            if remaining_n1 == 0:
                u = current_r1 - n1 * (n1 + 1) / 2.0
                total_weight += weight
                if u >= u_obs - 1e-12:
                    hit_weight += weight
            return
        cat = categories[i]
        n_cat = counts[cat]
        mr = midranks[cat]
        remaining_cats = sum(counts[categories[k]] for k in range(i + 1, len(categories)))
        min_j = max(0, remaining_n1 - remaining_cats)
        max_j = min(n_cat, remaining_n1)
        for j in range(min_j, max_j + 1):
            recurse(i + 1, remaining_n1 - j, current_r1 + j * mr, weight * math.comb(n_cat, j))

    recurse(0, n1, 0.0, 1)
    return u_obs, hit_weight / total_weight


def verify_coldstart(records: Path) -> list[dict]:
    rows = _coldstart_rows(records)
    adv_lrs, burg_lrs = _pool_by_shown_description(rows)

    if len(adv_lrs) != 90 or len(burg_lrs) != 90:
        raise ValueError(f"expected 90 proposals per description; got {len(adv_lrs)}, {len(burg_lrs)}")

    out: list[dict] = []

    for cell in ("advection_beta4.0_fixed", "burgers_nu0.001_fixed"):
        for label in ("advection_text", "burgers_text", "no_text"):
            if label == "advection_text":
                desc = "true" if cell == "advection_beta4.0_fixed" else "wrong"
            elif label == "burgers_text":
                desc = "wrong" if cell == "advection_beta4.0_fixed" else "true"
            else:
                desc = "removed"

            lrs = [r["lr"] for r in rows if r["cell"] == cell and r["desc_arm"] == desc]
            if len(lrs) != 45:
                raise ValueError(f"expected 45 samples for {cell}/{desc}")
            out.append(
                {
                    "claim": f"coldstart_median_lr_{cell}_{label}",
                    "recomputed": float(median(lrs)),
                    "unit": "lr_base",
                }
            )
            out.append(
                {
                    "claim": f"coldstart_share_0.001_{cell}_{label}",
                    "recomputed": sum(1 for v in lrs if v == 0.001) / len(lrs),
                    "unit": "fraction",
                }
            )

    out.append(
        {
            "claim": "coldstart_pooled_median_advection_text",
            "recomputed": float(median(adv_lrs)),
            "unit": "lr_base",
        }
    )
    out.append(
        {
            "claim": "coldstart_pooled_median_burgers_text",
            "recomputed": float(median(burg_lrs)),
            "unit": "lr_base",
        }
    )
    out.append(
        {
            "claim": "coldstart_pooled_share_0.001_advection_text",
            "recomputed": sum(1 for v in adv_lrs if v == 0.001) / len(adv_lrs),
            "unit": "fraction",
        }
    )
    out.append(
        {
            "claim": "coldstart_pooled_share_0.001_burgers_text",
            "recomputed": sum(1 for v in burg_lrs if v == 0.001) / len(burg_lrs),
            "unit": "fraction",
        }
    )

    u_adv, p_adv = _mann_whitney_u_and_exact_p(adv_lrs, burg_lrs)
    out.append(
        {
            "claim": "coldstart_mann_whitney_u",
            "recomputed": u_adv,
            "unit": "U statistic",
        }
    )
    out.append(
        {
            "claim": "coldstart_mann_whitney_p",
            "recomputed": p_adv,
            "unit": "p-value",
        }
    )

    # The seeded 20,000-draw shuffle (seed 0) depends on the CSV row order; keep it.
    cells = ("advection_beta4.0_fixed", "burgers_nu0.001_fixed")
    configs: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in rows:
        if row["cell"] not in cells or row["desc_arm"] not in ("true", "wrong"):
            continue
        configs.append(row["config"])
        shows_advection = (row["cell"] == cells[0] and row["desc_arm"] == "true") or (
            row["cell"] == cells[1] and row["desc_arm"] == "wrong"
        )
        labels.append(0 if shows_advection else 1)

    if len(configs) != 180:
        raise ValueError(f"expected 180 true/wrong cold-start samples; got {len(configs)}")
    if sum(labels) != 90 or len(labels) - sum(labels) != 90:
        raise ValueError(f"unexpected description counts: {sum(labels)} advection, {len(labels) - sum(labels)} burgers")

    y = np.array(labels)
    k = len(configs)
    dists = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            d = action_distance(configs[i], configs[j])
            dists[i, j] = d
            dists[j, i] = d
    upper = np.triu_indices(k, k=1)
    pair_distances = dists[upper]

    def statistic(group_labels: np.ndarray) -> float:
        same = group_labels[upper[0]] == group_labels[upper[1]]
        cross = pair_distances[~same].mean()
        within = pair_distances[same].mean()
        return float(cross - within)

    observed = statistic(y)
    rng = np.random.default_rng(0)
    n_perm = 20_000
    null = np.empty(n_perm)
    for idx in range(n_perm):
        null[idx] = statistic(rng.permutation(y))
    p_perm = (np.count_nonzero(null >= observed - 1e-12) + 1) / (n_perm + 1)

    out.append(
        {
            "claim": "coldstart_aggregate_action_distance",
            "recomputed": observed,
            "unit": "action distance",
        }
    )
    out.append(
        {
            "claim": "coldstart_aggregate_action_distance_p",
            "recomputed": p_perm,
            "unit": "p-value",
        }
    )
    return out


def _replay_rows(records: Path, campaign_dirs: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for d in campaign_dirs:
        path = records / "campaigns" / d / "permutation_distances.csv"
        rows.extend(_load_csv(path))
    return rows


def _trace_effects(rows: list[dict[str, str]], arm: str, baseline_arm: str) -> list[dict]:
    """Median paired gap per trace (cell x seed) for arm vs baseline."""
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["cell"], int(row["seed"]), row["window"], row["arm"])
        grouped[key].append(float(row["distance"]))
    medians = {key: median(vals) for key, vals in grouped.items()}
    traces = sorted({(row["cell"], int(row["seed"])) for row in rows})

    out: list[dict] = []
    for cell, seed in traces:
        gaps = []
        for window in ("early", "mid", "late"):
            tkey = (cell, seed, window, arm)
            bkey = (cell, seed, window, baseline_arm)
            if tkey not in medians or bkey not in medians:
                continue
            gaps.append(medians[tkey] - medians[bkey])
        if not gaps:
            raise ValueError(f"no paired replay windows for {arm} vs {baseline_arm}, {cell} seed {seed}")
        out.append({"cell": cell, "seed": seed, "gap": median(gaps)})
    return out


def _cell_clustered_signflip_p(traces: list[dict]) -> tuple[float, int]:
    cell_med: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        cell_med[t["cell"]].append(t["gap"])
    cell_medians = [median(v) for v in cell_med.values()]
    observed = sum(cell_medians)
    mags = [abs(g) for g in cell_medians]
    hits = 0
    for signs in itertools.product((-1, 1), repeat=len(cell_medians)):
        s = sum(sign * mag for sign, mag in zip(signs, mags, strict=True))
        if s >= observed - 1e-12:
            hits += 1
    return hits / 2 ** len(cell_medians), len(cell_medians)


def _hierarchical_bootstrap(traces: list[dict], n_boot: int, seed: int) -> np.ndarray:
    cells = sorted({t["cell"] for t in traces})
    per_cell = {cell: np.array([t["gap"] for t in traces if t["cell"] == cell]) for cell in cells}
    counts = {len(v) for v in per_cell.values()}
    if len(counts) != 1:
        raise ValueError(f"unbalanced panel: per-cell seed counts {sorted(counts)}")
    n_seeds = counts.pop()
    mat = np.array([per_cell[cell] for cell in cells])
    rng = np.random.default_rng(seed)
    cell_draws = rng.integers(0, len(cells), size=(n_boot, len(cells), 1))
    seed_draws = rng.integers(0, n_seeds, size=(n_boot, len(cells), n_seeds))
    return np.median(mat[cell_draws, seed_draws].reshape(n_boot, -1), axis=1)


def _verify_relabelled_best(records: Path, campaign_name: str) -> list[dict]:
    """Track whether replay proposals move toward the newly labelled best config."""
    campaign_dir = records / "campaigns" / campaign_name
    manifest_path = campaign_dir / "stimulus_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing stimulus manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    stimuli = {tuple(s["base_unit_identity"]): s for s in manifest["stimuli"]}

    histories: dict[tuple[str, str], list[dict]] = {}
    for uid in stimuli:
        key = (uid[0], uid[2])
        if key not in histories:
            batch_name = _batch_for_cell(records, key[0])
            history_path = records / "batches" / batch_name / "alga" / key[0] / key[1] / "agent_history.json"
            histories[key] = _load_json(history_path)

    rows = _load_csv(campaign_dir / "permutation_distances.csv")
    data: list[dict] = []
    for row in rows:
        uid = (row["cell"], int(row["seed"]), row["run_id"], row["window"], int(row["iteration"]), int(row["repeat"]))
        stim = stimuli.get(uid)
        if stim is None:
            continue
        hist = histories[(row["cell"], row["run_id"])]
        perm = stim["target_permutation"]["source_positions_by_destination"]
        finite = stim["target_permutation"]["finite_history_positions"]
        vals = [float(hist[p]["val_nrmse"]) for p in finite]
        b_old_local = min(range(len(finite)), key=lambda i: vals[i])
        b_old = finite[b_old_local]
        permuted_vals = [vals[perm[i]] for i in range(len(finite))]
        b_new_local = min(range(len(finite)), key=lambda i: permuted_vals[i])
        b_new = finite[b_new_local]
        changed = b_old != b_new
        bold_cfg = normalise_config(hist[b_old]["config"])
        bnew_cfg = normalise_config(hist[b_new]["config"])
        a = normalise_config(ast.literal_eval(row["proposed_config"]))
        s = action_distance(a, bold_cfg) - action_distance(a, bnew_cfg)
        data.append(
            {
                "cell": row["cell"],
                "seed": int(row["seed"]),
                "window": row["window"],
                "arm": row["arm"],
                "s": s,
                "changed": changed,
                "repeat": int(row["repeat"]),
            }
        )

    def _trace_level_effects(subset: list[dict]) -> list[float]:
        grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
        for d in subset:
            grouped[(d["cell"], d["seed"], d["window"], d["arm"])].append(d["s"])
        medians = {key: median(vals) for key, vals in grouped.items()}
        trace_effects: dict[tuple[str, int], list[float]] = defaultdict(list)
        for (cell, seed, window, arm), val in medians.items():
            if arm != "target":
                continue
            if (cell, seed, window, "sham") not in medians:
                continue
            trace_effects[(cell, seed)].append(val - medians[(cell, seed, window, "sham")])
        return [median(v) for v in trace_effects.values()]

    def _window_level_effects(subset: list[dict]) -> list[float]:
        grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
        for d in subset:
            grouped[(d["cell"], d["seed"], d["window"], d["arm"])].append(d["s"])
        medians = {key: median(vals) for key, vals in grouped.items()}
        gaps = []
        for (cell, seed, window, arm), val in medians.items():
            if arm != "target":
                continue
            if (cell, seed, window, "sham") not in medians:
                continue
            gaps.append(val - medians[(cell, seed, window, "sham")])
        return gaps

    all_effects = _trace_level_effects(data)
    changed_data = [d for d in data if d["changed"]]
    changed_effects = _trace_level_effects(changed_data)
    changed_window_gaps = _window_level_effects(changed_data)

    return [
        {
            "claim": "replay_relabelled_best_all_repeats_median",
            "recomputed": float(median(all_effects)),
            "unit": "action distance",
        },
        {
            "claim": "replay_relabelled_best_changed_median",
            "recomputed": float(median(changed_effects)),
            "unit": "action distance",
        },
        {
            "claim": "replay_relabelled_best_changed_window_median",
            "recomputed": float(median(changed_window_gaps)),
            "unit": "action distance",
        },
    ]


def verify_replay(records: Path) -> list[dict]:
    out: list[dict] = []

    within_rows = _replay_rows(records, (WITHIN_SHAM_REPLAY,))
    within_target = _trace_effects(within_rows, "target", "sham")
    within_sham = _trace_effects(within_rows, "sham", "control")

    out.append(
        {
            "claim": "replay_within_reassign_vs_notation_median",
            "recomputed": float(median(t["gap"] for t in within_target)),
            "unit": "action distance",
        }
    )
    p, n_cells = _cell_clustered_signflip_p(within_target)
    out.append(
        {
            "claim": "replay_within_reassign_vs_notation_p",
            "recomputed": p,
            "unit": "p-value",
            "n_cells": n_cells,
        }
    )
    boot = _hierarchical_bootstrap(within_target, 20_000, 20260725)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    out.append(
        {
            "claim": "replay_within_reassign_vs_notation_ci95",
            "recomputed": [lo, hi],
            "unit": "action distance",
        }
    )

    out.append(
        {
            "claim": "replay_within_notation_vs_resampling_median",
            "recomputed": float(median(t["gap"] for t in within_sham)),
            "unit": "action distance",
        }
    )
    p, n_cells = _cell_clustered_signflip_p(within_sham)
    out.append(
        {
            "claim": "replay_within_notation_vs_resampling_p",
            "recomputed": p,
            "unit": "p-value",
            "n_cells": n_cells,
        }
    )

    cross_rows = _replay_rows(records, CROSS_REPLAY_DIRS)
    cross_target = _trace_effects(cross_rows, "target", "control")
    out.append(
        {
            "claim": "replay_cross_reassign_vs_resampling_median",
            "recomputed": float(median(t["gap"] for t in cross_target)),
            "unit": "action distance",
        }
    )
    p, n_cells = _cell_clustered_signflip_p(cross_target)
    out.append(
        {
            "claim": "replay_cross_reassign_vs_resampling_p",
            "recomputed": p,
            "unit": "p-value",
            "n_cells": n_cells,
        }
    )

    cross_joint = _trace_effects(cross_rows, "joint", "control")
    out.append(
        {
            "claim": "replay_cross_feedback_bundle_median",
            "recomputed": float(median(t["gap"] for t in cross_joint)),
            "unit": "action distance",
        }
    )
    p, n_cells = _cell_clustered_signflip_p(cross_joint)
    out.append(
        {
            "claim": "replay_cross_feedback_bundle_p",
            "recomputed": p,
            "unit": "p-value",
            "n_cells": n_cells,
        }
    )

    feedback_rows = _replay_rows(records, (WITHIN_FEEDBACK_REPLAY,))
    feedback_target = _trace_effects(feedback_rows, "target", "control")
    feedback_joint = _trace_effects(feedback_rows, "joint", "control")
    out.append(
        {
            "claim": "replay_within_feedback_target_vs_control_median",
            "recomputed": float(median(t["gap"] for t in feedback_target)),
            "unit": "action distance",
        }
    )
    p, n_cells = _cell_clustered_signflip_p(feedback_target)
    out.append(
        {
            "claim": "replay_within_feedback_target_vs_control_p",
            "recomputed": p,
            "unit": "p-value",
            "n_cells": n_cells,
        }
    )
    out.append(
        {
            "claim": "replay_within_feedback_joint_vs_control_median",
            "recomputed": float(median(t["gap"] for t in feedback_joint)),
            "unit": "action distance",
        }
    )
    p, n_cells = _cell_clustered_signflip_p(feedback_joint)
    out.append(
        {
            "claim": "replay_within_feedback_joint_vs_control_p",
            "recomputed": p,
            "unit": "p-value",
            "n_cells": n_cells,
        }
    )

    out.extend(_verify_relabelled_best(records, WITHIN_SHAM_REPLAY))

    return out


def _benchmark_rows(result: dict[str, Any], method: str) -> list[dict]:
    """Return normalised trial rows for a benchmark result JSON."""
    if method == "random":
        rows = result.get("all_iterations") or []
        return [
            {
                "number": row["iteration"],
                "config": row["config"],
                "val_nrmse": row["metrics"]["val_nrmse"],
            }
            for row in rows
        ]
    if method == "bayesian":
        rows = result.get("all_trials") or []
        return [
            {
                "number": row["number"],
                "config": row["params"],
                "val_nrmse": row["value"],
            }
            for row in rows
        ]
    return []


def _configs_from_rows(rows: list[dict]) -> list[dict[str, Any]]:
    def _number(row: dict) -> int:
        if "number" in row:
            return int(row["number"])
        if "iteration" in row:
            return int(row["iteration"])
        return 0

    ordered = sorted(rows, key=_number)
    return [normalise_config(r["config"]) for r in ordered if r.get("config")]


def _consecutive_steps(configs: list[dict[str, Any]]) -> list[float]:
    return [action_distance(configs[i], configs[i + 1]) for i in range(len(configs) - 1)]


def _spearman_rho(values: list[float]) -> float:
    """Spearman correlation of values with their index."""
    n = len(values)
    if n < 2:
        return 0.0

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return out

    rx, ry = ranks(list(range(n))), ranks(values)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _median_window(rows: list[list[float]], lo: int, hi: int) -> float:
    return float(median([median(r[lo:hi]) for r in rows]))


def _trajectory_stats(traces: list[list[float]]) -> dict[str, Any]:
    rhos = [_spearman_rho(t) for t in traces]
    negative = sum(1 for r in rhos if r < 0)
    if len(traces) != 36:
        raise ValueError(f"expected 36 trajectory traces; got {len(traces)}")
    return {
        "n_traces": len(traces),
        "median_rho": float(median(rhos)),
        "negative_traces": f"{negative}/{len(traces)}",
        "window_1_5": _median_window(traces, 0, 5),
        "window_6_12": _median_window(traces, 5, 12),
        "window_13_19": _median_window(traces, 12, 19),
        "all": float(median([median(r) for r in traces])),
        "early_exact_repeats_pct": 100.0
        * sum(1 for r in traces for v in r[0:5] if v == 0)
        / sum(len(r[0:5]) for r in traces),
        "late_exact_repeats_pct": 100.0
        * sum(1 for r in traces for v in r[12:19] if v == 0)
        / sum(len(r[12:19]) for r in traces),
    }


def verify_tpe_startup(records: Path) -> list[dict]:
    tpe_startup_cells = {
        "burgers_nu0.001_fixed": "2026-07-19_within_family_repaired_panel",
        "advection_to_burgers_nu0.001": "2026-07-20_xpde_repaired_panel",
    }
    tpe_batch = records / "batches" / "2026-07-31_tpe_startup_control"
    llm_tests: dict[str, dict[int, float]] = {}
    tpe_tests: dict[str, dict[int, float]] = {}
    for cell, llm_batch in tpe_startup_cells.items():
        llm_dir = records / "batches" / llm_batch / "alga" / cell
        llm_tests[cell] = {}
        for run in llm_dir.iterdir():
            final = _load_json(run / "final_results.json")
            seed = int(final["config"]["seed"])
            llm_tests[cell][seed] = float(final["test_metrics"]["nrmse_clamped"])
        tpe_tests[cell] = {}
        for path in (tpe_batch / "benchmarks" / cell).glob("*/*_results.json"):
            result = _load_json(path)
            seed = int(result["seed"])
            tpe_tests[cell][seed] = float(result["test"]["nrmse_clamped"])

    wins = 0
    total = 0
    within_ratios: list[float] = []
    cross_ratios: list[float] = []
    for cell, seeds in tpe_tests.items():
        for seed, tpe_val in seeds.items():
            llm_val = llm_tests[cell][seed]
            total += 1
            if llm_val < tpe_val:
                wins += 1
            ratio = tpe_val / llm_val
            if cell == "burgers_nu0.001_fixed":
                within_ratios.append(ratio)
            else:
                cross_ratios.append(ratio)

    out: list[dict] = []
    out.append(
        {
            "claim": "tpe_startup_llm_wins",
            "recomputed": f"{wins}/{total}",
            "unit": "wins",
        }
    )
    out.append(
        {
            "claim": "tpe_startup_within_family_median_ratio",
            "recomputed": float(median(within_ratios)),
            "unit": "ratio",
        }
    )
    out.append(
        {
            "claim": "tpe_startup_cross_family_median_ratio",
            "recomputed": float(median(cross_ratios)),
            "unit": "ratio",
        }
    )
    out.append(
        {
            "claim": "tpe_startup_pooled_median_ratio",
            "recomputed": float(median(within_ratios + cross_ratios)),
            "unit": "ratio",
            "optional": True,
        }
    )
    return out


def verify_trajectory_stats(records: Path) -> list[dict]:
    trajectory_batches = (
        "2026-07-19_within_family_repaired_panel",
        "2026-07-20_xpde_repaired_panel",
        "2026-07-28_reverse_xpde_panel",
    )
    llm_traces: list[list[float]] = []
    random_traces: list[list[float]] = []
    tpe_traces: list[list[float]] = []
    for batch_name in trajectory_batches:
        batch_dir = records / "batches" / batch_name
        for path in sorted((batch_dir / "alga").rglob("agent_history.json")):
            configs = _configs_from_rows(_load_json(path))
            steps = _consecutive_steps(configs)
            if len(steps) >= 19:
                llm_traces.append(steps[:19])
        for path in sorted((batch_dir / "benchmarks").rglob("random*_results.json")):
            rows = _benchmark_rows(_load_json(path), "random")
            configs = _configs_from_rows(rows)
            steps = _consecutive_steps(configs)
            if len(steps) >= 19:
                random_traces.append(steps[:19])
        for path in sorted((batch_dir / "benchmarks").rglob("full_finetune_bayesian*_results.json")):
            rows = _benchmark_rows(_load_json(path), "bayesian")
            configs = _configs_from_rows(rows)
            steps = _consecutive_steps(configs)
            if len(steps) >= 19:
                tpe_traces.append(steps[:19])

    out: list[dict] = []
    for name, traces in (("llm", llm_traces), ("random", random_traces), ("tpe", tpe_traces)):
        stats = _trajectory_stats(traces)
        for key, value in stats.items():
            out.append(
                {
                    "claim": f"supp_trajectory_{name}_{key}",
                    "recomputed": value,
                    "unit": "varies",
                }
            )
    return out


def verify_early_budget(records: Path, cells: list[dict]) -> list[dict]:
    cell_to_batch: dict[str, str] = {}
    for batch_name, _, _ in PERFORMANCE_BATCHES:
        for path in (records / "batches" / batch_name / "alga").iterdir():
            if path.is_dir():
                cell_to_batch[path.name] = batch_name

    counts: dict[int, tuple[int, int]] = {}
    total_count = 0
    for k in (1, 3, 5):
        wins_random = 0
        wins_tpe = 0
        total = 0
        for cell in cells:
            batch_name = cell_to_batch[cell["cell"]]
            batch = records / "batches" / batch_name
            for s in SEEDS:
                llm_best = math.inf
                for run in (batch / "alga" / cell["cell"]).iterdir():
                    final = _load_json(run / "final_results.json")
                    if int(final["config"]["seed"]) != s:
                        continue
                    history = _load_json(run / "agent_history.json")
                    vals = [h["val_nrmse"] for h in history[:k] if h.get("val_nrmse") is not None]
                    if vals:
                        llm_best = min(llm_best, min(vals))
                random_best = math.inf
                tpe_best = math.inf
                for path in (batch / "benchmarks" / cell["cell"]).glob("*/*_results.json"):
                    result = _load_json(path)
                    if int(result["seed"]) != s:
                        continue
                    method = _normalise_method(str(result["method"]))
                    rows = _benchmark_rows(result, method)
                    best = min((r["val_nrmse"] for r in rows[:20]), default=math.inf)
                    if method == "random":
                        random_best = best
                    elif method == "bayesian":
                        tpe_best = best
                if llm_best != math.inf and random_best != math.inf and tpe_best != math.inf:
                    total += 1
                    if llm_best < random_best:
                        wins_random += 1
                    if llm_best < tpe_best:
                        wins_tpe += 1
        counts[k] = (wins_random, wins_tpe)
        total_count = total

    out: list[dict] = []
    for k, (wr, wt) in counts.items():
        out.append(
            {
                "claim": f"supp_early_budget_k{k}_vs_random_wins",
                "recomputed": f"{wr}/{total_count}",
                "unit": "wins",
            }
        )
        out.append(
            {
                "claim": f"supp_early_budget_k{k}_vs_tpe_wins",
                "recomputed": f"{wt}/{total_count}",
                "unit": "wins",
            }
        )
    return out


def _values_equal(recomputed: Any, expected: Any, tol: float | None) -> bool:
    if isinstance(expected, list) and isinstance(recomputed, list):
        if len(expected) != len(recomputed):
            return False
        return all(_values_equal(r, e, tol) for r, e in zip(recomputed, expected, strict=True))
    if isinstance(expected, str) and isinstance(recomputed, str):
        return expected == recomputed
    if isinstance(expected, int | float) and isinstance(recomputed, int | float):
        if tol is None:
            return float(recomputed) == float(expected)
        return abs(float(recomputed) - float(expected)) <= tol
    return False


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True, help="path to records root")
    ap.add_argument("--expected", type=Path, default=None, help="path to expected.json")
    args = ap.parse_args()

    records: Path = args.records
    expected_path: Path = args.expected or Path(__file__).with_name("expected.json")
    with expected_path.open() as handle:
        expected = json.load(handle)

    cells = load_performance(records)
    results: list[dict] = []
    results.extend(verify_endpoint_table(cells))
    results.extend(verify_first_proposal(cells))
    results.extend(verify_coldstart(records))
    results.extend(verify_replay(records))
    results.extend(verify_tpe_startup(records))
    results.extend(verify_trajectory_stats(records))
    results.extend(verify_early_budget(records, cells))

    computed: dict[str, dict] = {r["claim"]: r for r in results}
    table: list[dict] = []
    mismatches: list[dict] = []
    missing: list[str] = []
    optional: list[dict] = []
    failed = False

    for claim, exp in expected.items():
        if claim not in computed:
            missing.append(claim)
            failed = True
            continue
        item = computed[claim]
        tol = exp.get("tol")
        if tol is not None:
            tol = float(tol)
        ok = _values_equal(item["recomputed"], exp["value"], tol)
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failed = True
        row = {
            "claim": claim,
            "paper": _format_value(exp["value"]),
            "recomputed": _format_value(item["recomputed"]),
            "status": status,
            "source": exp.get("source", ""),
        }
        table.append(row)
        if not ok:
            mismatches.append({**row, "tol": tol})

    for item in results:
        if item.get("optional") and item["claim"] not in expected:
            optional.append(item)

    print(f"{'Claim':<55s} {'Paper':>18s} {'Recomputed':>18s} {'Status':>10s} {'Source':>20s}")
    print("-" * 125)
    for row in table:
        print(
            f"{row['claim']:<55s} {row['paper']:>18s} {row['recomputed']:>18s} "
            f"{row['status']:>10s} {row['source']:>20s}"
        )

    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(f"  {m['claim']}: paper={m['paper']} recomputed={m['recomputed']} tol={m['tol']}")

    if missing:
        print("\nMissing expected.json entries (computed value absent):")
        for claim in missing:
            print(f"  {claim}")

    if optional:
        print("\nOptional diagnostics (not checked against expected.json):")
        for item in optional:
            print(f"  {item['claim']}: {_format_value(item['recomputed'])}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
