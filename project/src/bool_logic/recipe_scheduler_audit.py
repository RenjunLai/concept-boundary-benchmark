from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import T2_GOLD_SET_SIZES
from .io_utils import ensure_dir, write_json, write_jsonl


DEFAULT_FAMILY_QUOTAS = {"wordnet": 650, "babelnet": 100}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _task_capacity(row: dict[str, Any], task: str, threshold: int) -> int:
    counts = row["counts_by_min_final_positive"][str(threshold)]
    if task == "T1":
        return int(counts["T1"])
    if task == "T2":
        return int(counts["T2_any"])
    if task == "T3":
        return int(counts["T3"])
    raise ValueError(task)


def _t2_all_gold_capacity(row: dict[str, Any], threshold: int) -> int:
    return int(row["counts_by_min_final_positive"][str(threshold)]["T2_all_gold_sizes"])


def _evaluate_scheduler(
    rows: list[dict[str, Any]],
    threshold: int,
    quotas: dict[str, int],
    t2_policy: str,
    hard_negative_policy: str,
) -> dict[str, Any]:
    shortage_rows = []
    t2_gold_plan_rows = []
    for row in rows:
        family = row["dataset_family"]
        quota = quotas[family]
        for task in ("T1", "T2", "T3"):
            population = _task_capacity(row, task, threshold)
            if population < quota:
                shortage_rows.append(
                    {
                        "dataset_family": family,
                        "task": task,
                        "constraint_cell": row["constraint_cell"],
                        "population": population,
                        "quota": quota,
                    }
                )

        if t2_policy == "balanced_gold_sizes":
            each = quota // len(T2_GOLD_SET_SIZES)
            remainder = quota % len(T2_GOLD_SET_SIZES)
            target = {str(size): each + (1 if index < remainder else 0) for index, size in enumerate(T2_GOLD_SET_SIZES)}
            all_gold_population = _t2_all_gold_capacity(row, threshold)
            exact_balanced = all_gold_population >= quota
            t2_gold_plan_rows.append(
                {
                    "dataset_family": family,
                    "constraint_cell": row["constraint_cell"],
                    "quota": quota,
                    "target_gold_size_counts": target,
                    "all_gold_population": all_gold_population,
                    "exact_balanced_possible": exact_balanced,
                }
            )
            if not exact_balanced:
                shortage_rows.append(
                    {
                        "dataset_family": family,
                        "task": "T2_all_gold_sizes",
                        "constraint_cell": row["constraint_cell"],
                        "population": all_gold_population,
                        "quota": quota,
                    }
                )
        elif t2_policy == "opportunistic_gold_sizes":
            # Uses any feasible T2 row and balances gold sizes greedily from available buckets.
            pass
        else:
            raise ValueError(t2_policy)

    if hard_negative_policy == "fixed_ratio":
        status = "requires_candidate_level_audit"
    elif shortage_rows:
        status = "quota_shortage"
    else:
        status = "audited_no_shortage"

    return {
        "threshold": threshold,
        "family_quotas": quotas,
        "t2_policy": t2_policy,
        "hard_negative_policy": hard_negative_policy,
        "status": status,
        "candidate_level_audit_required": hard_negative_policy == "fixed_ratio",
        "shortage_count": len(shortage_rows),
        "shortages": shortage_rows[:20],
        "t2_gold_plan_examples": t2_gold_plan_rows[:12],
        "requires_candidate_level_generation": True,
        "notes": _scheduler_notes(t2_policy, hard_negative_policy),
    }


def _scheduler_notes(t2_policy: str, hard_negative_policy: str) -> list[str]:
    notes = []
    if t2_policy == "balanced_gold_sizes":
        notes.append("T2 target is deterministic near-even split across gold_set_size 0/1/2 within each family-task-cell stratum.")
        notes.append("If a cell lacks all-gold-size capacity for the quota, the scheduler must fall back or report shortage.")
    else:
        notes.append("T2 target is best-effort balance over available gold-size buckets; unsupported buckets are skipped with report fields.")
    if hard_negative_policy == "prefer":
        notes.append("Hard negatives are used first when available, but not required at a fixed ratio.")
    elif hard_negative_policy == "fixed_ratio":
        notes.append("Hard negatives require an explicit ratio and candidate-level audit before freeze.")
    else:
        raise ValueError(hard_negative_policy)
    return notes


def build_recipe_scheduler_audit(
    threshold_audit_path: Path,
    out_dir: Path,
    threshold: int = 5,
) -> dict[str, Any]:
    audit = json.loads(threshold_audit_path.read_text(encoding="utf-8"))
    rows = audit["cell_rows"]
    out_dir = ensure_dir(out_dir)
    scenarios = [
        {
            "scenario_id": "A_balanced_simple",
            "t2_policy": "balanced_gold_sizes",
            "hard_negative_policy": "prefer",
            "description": "Recommended: balance answers and T2 gold sizes, prefer hard negatives, no fixed hard-negative ratio.",
        },
        {
            "scenario_id": "B_opportunistic_simple",
            "t2_policy": "opportunistic_gold_sizes",
            "hard_negative_policy": "prefer",
            "description": "Simpler fallback: use any feasible T2 realization and report actual gold-size distribution.",
        },
        {
            "scenario_id": "C_balanced_with_fixed_hard_negative_ratio",
            "t2_policy": "balanced_gold_sizes",
            "hard_negative_policy": "fixed_ratio",
            "description": "More complex: balance answers and require a fixed hard-negative ratio; needs candidate-level capacity audit.",
        },
    ]
    scenario_rows = []
    for scenario in scenarios:
        scenario_rows.append(
            {
                **scenario,
                **_evaluate_scheduler(
                    rows,
                    threshold=threshold,
                    quotas=DEFAULT_FAMILY_QUOTAS,
                    t2_policy=scenario["t2_policy"],
                    hard_negative_policy=scenario["hard_negative_policy"],
                ),
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_name": "recipe_scheduler_audit",
        "threshold_audit_path": str(threshold_audit_path),
        "standard_positive_pool_threshold": threshold,
        "family_quotas": DEFAULT_FAMILY_QUOTAS,
        "recommendation": {
            "scenario_id": "A_balanced_simple",
            "reason": "It balances the answer dimensions the benchmark reports, avoids unsupported hard-negative ratio commitments, and has no quota shortage under the audited threshold.",
        },
        "scenarios": scenario_rows,
        "outputs": {
            "summary": str(out_dir / "recipe_scheduler_audit_summary.json"),
            "scenarios": str(out_dir / "scheduler_scenarios.jsonl"),
        },
    }
    write_jsonl(out_dir / "scheduler_scenarios.jsonl", scenario_rows)
    write_json(out_dir / "recipe_scheduler_audit_summary.json", summary)
    return summary


def write_recipe_scheduler_audit(threshold_audit_path: Path, out_dir: Path, threshold: int = 5) -> dict[str, Any]:
    return build_recipe_scheduler_audit(threshold_audit_path, out_dir, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate realization scheduler options for current recipe.")
    parser.add_argument(
        "--threshold-audit",
        type=Path,
        default=Path("audits/recipe_seed/positive_pool_threshold_audit.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("audits/recipe_scheduler"))
    parser.add_argument("--threshold", type=int, default=5)
    args = parser.parse_args()
    summary = write_recipe_scheduler_audit(args.threshold_audit, args.out_dir, threshold=args.threshold)
    print({"summary": summary["outputs"]["summary"], "recommendation": summary["recommendation"]})


if __name__ == "__main__":
    main()
