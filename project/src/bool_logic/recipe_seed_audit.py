from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .constants import A_COUNTS, B_COUNTS, INFORMATION_LEVELS, POOL_SIZE, TASKS
from .io_utils import canonical_json, ensure_dir, stable_hash, write_json, write_jsonl
from .recipe_universe_audit import _core_capacity, _pct, _quantiles, _top_counter
from .resources import Inventory, load_inventories
from .taxonomy_bitsets import BitsetHelper


TOP_N = 12
PROBE_ROWS_PER_CELL = 80
POSITIVE_POOL_THRESHOLDS = (1, 2, 5, 10, 20, 50, 100)
DEFAULT_FAMILY_QUOTAS = {"wordnet": 650, "babelnet": 100}
TEST_FAMILY_QUOTAS = {"wordnet": 3, "babelnet": 1}


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _children_covering_seed(bitsets: BitsetHelper, a1_id: str, seed_bit: int) -> tuple[str, ...]:
    children = []
    for child_id in bitsets.usable_children(a1_id):
        if bitsets.closure_bits(child_id) & seed_bit:
            children.append(child_id)
    return tuple(sorted(children, key=lambda item: (bitsets.closure_size(item), item)))


def _children_not_covering_seed(bitsets: BitsetHelper, parent_id: str, seed_bit: int) -> tuple[str, ...]:
    children = []
    for child_id in bitsets.usable_children(parent_id):
        if bitsets.closure_bits(child_id) & seed_bit:
            continue
        children.append(child_id)
    return tuple(sorted(children, key=lambda item: (-bitsets.closure_size(item), item)))


def _select_branches(bitsets: BitsetHelper, a1_id: str, seed_id: str, branch_count: int) -> tuple[str, ...] | None:
    if branch_count == 0:
        return ()
    seed_bit = bitsets.bit(seed_id)
    covering = _children_covering_seed(bitsets, a1_id, seed_bit)
    if not covering:
        return None
    seed_branch = covering[0]
    branches = [seed_branch]
    if len(branches) == branch_count:
        return tuple(sorted(branches))
    for child_id in _children_not_covering_seed(bitsets, a1_id, seed_bit):
        if child_id != seed_branch and child_id not in branches:
            branches.append(child_id)
            if len(branches) == branch_count:
                return tuple(sorted(branches))
    # If a DAG gives multiple seed-covering children, allow additional covering
    # branches only after all non-covering sibling branches have been tried.
    for child_id in covering[1:]:
        if child_id not in branches:
            branches.append(child_id)
            if len(branches) == branch_count:
                return tuple(sorted(branches))
    return None


def _b_options_for_seed_core(
    bitsets: BitsetHelper,
    a1_id: str,
    branch_ids: tuple[str, ...],
    seed_id: str,
) -> tuple[dict[str, str], ...]:
    seed_bit = bitsets.bit(seed_id)
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    if branch_ids:
        for branch_id in branch_ids:
            for child_id in _children_not_covering_seed(bitsets, branch_id, seed_bit):
                if child_id in seen:
                    continue
                seen.add(child_id)
                options.append({"B_id": child_id, "paired_A_source": branch_id, "B_source": "inside_positive_branch"})
    for child_id in _children_not_covering_seed(bitsets, a1_id, seed_bit):
        if child_id in seen or child_id in branch_ids:
            continue
        seen.add(child_id)
        options.append({"B_id": child_id, "paired_A_source": a1_id, "B_source": "inside_A1"})
    return tuple(options)


def _choose_b(bitsets: BitsetHelper, options: tuple[dict[str, str], ...], b_count: int) -> tuple[dict[str, str], ...] | None:
    if b_count == 0:
        return ()
    if len(options) < b_count:
        return None
    return tuple(sorted(options, key=lambda item: (-bitsets.closure_size(item["B_id"]), item["paired_A_source"], item["B_id"]))[:b_count])


def _find_lowest_viable_seed_core(
    inventory: Inventory,
    bitsets: BitsetHelper,
    seed_id: str,
    a_count: int,
    b_count: int,
    min_final_positive_count: int = 1,
) -> tuple[dict[str, Any] | None, str]:
    branch_count = a_count - 1
    seed_bit = bitsets.bit(seed_id)
    ancestors = bitsets.ancestor_walk(seed_id)
    if not ancestors:
        return None, "no_usable_ancestor"
    reasons: Counter[str] = Counter()
    for a1_id, distance in ancestors:
        if not (bitsets.closure_bits(a1_id) & seed_bit):
            reasons["seed_not_in_A1_scope"] += 1
            continue
        branch_ids = _select_branches(bitsets, a1_id, seed_id, branch_count)
        if branch_ids is None:
            reasons["insufficient_positive_branches"] += 1
            continue
        b_options = _b_options_for_seed_core(bitsets, a1_id, branch_ids, seed_id)
        chosen_b = _choose_b(bitsets, b_options, b_count)
        if chosen_b is None:
            reasons["insufficient_B_options"] += 1
            continue
        capacity = _core_capacity(bitsets, a1_id, branch_ids, chosen_b)
        if capacity["final_positive_count"] < min_final_positive_count or not capacity["task_capacity"]["T1_true"]:
            reasons["empty_final_positive_scope"] += 1
            continue
        positive_bits_contains_seed = True
        # Reconstruct only the seed check cheaply from selected scope and B.
        if branch_ids:
            branch_bits = 0
            for branch_id in branch_ids:
                branch_bits |= bitsets.closure_bits(branch_id)
            if not (branch_bits & seed_bit):
                positive_bits_contains_seed = False
        b_bits = 0
        for option in chosen_b:
            b_bits |= bitsets.closure_bits(option["B_id"])
        if b_bits & seed_bit:
            positive_bits_contains_seed = False
        if not positive_bits_contains_seed:
            reasons["seed_lost_after_branch_or_B"] += 1
            continue
        return {
            "seed_id": seed_id,
            "seed_surface": inventory.get(seed_id).surface,
            "A1_id": a1_id,
            "A1_surface": inventory.get(a1_id).surface,
            "A_branch_ids": branch_ids,
            "A_branch_surfaces": tuple(inventory.get(item).surface for item in branch_ids),
            "B_ids": tuple(item["B_id"] for item in chosen_b),
            "B_surfaces": tuple(inventory.get(item["B_id"]).surface for item in chosen_b),
            "paired_A_ids": tuple(item["paired_A_source"] for item in chosen_b),
            "B_sources": tuple(item["B_source"] for item in chosen_b),
            "lowest_viable_distance": distance,
            "A1_scope_size": capacity["a1_scope_size"],
            "positive_pre_exclusion_count": capacity["positive_pre_exclusion_count"],
            "excluded_scope_count": capacity["excluded_scope_count"],
            "final_positive_count": capacity["final_positive_count"],
            "hits_b_count": capacity["hits_b_count"],
            "fails_positive_branch_count": capacity["fails_positive_branch_count"],
            "outside_A1_count": capacity["outside_a1_count"],
            "negative_count": capacity["negative_count"],
            "hard_negative_count": capacity["hard_negative_count"],
            "task_capacity": capacity["task_capacity"],
        }, "ok"
    if reasons:
        return None, reasons.most_common(1)[0][0]
    return None, "no_viable_A1"


class SeedCellAccumulator:
    def __init__(self, family: str, a_count: int, b_count: int):
        self.family = family
        self.a_count = a_count
        self.b_count = b_count
        self.seed_total = 0
        self.feasible = 0
        self.missing: Counter[str] = Counter()
        self.task_counts: Counter[str] = Counter()
        self.threshold_task_counts: dict[int, Counter[str]] = {threshold: Counter() for threshold in POSITIVE_POOL_THRESHOLDS}
        self.a1_counter: Counter[tuple[str, str]] = Counter()
        self.seed_counter: Counter[tuple[str, str]] = Counter()
        self.B_sources_counter: Counter[str] = Counter()
        self.distance_values: list[int] = []
        self.a1_scope_values: list[int] = []
        self.positive_values: list[int] = []
        self.excluded_values: list[int] = []
        self.final_positive_values: list[int] = []
        self.hits_b_values: list[int] = []
        self.fails_branch_values: list[int] = []
        self.outside_values: list[int] = []
        self.negative_values: list[int] = []
        self.hard_values: list[int] = []
        self.examples: list[dict[str, Any]] = []

    def add_missing(self, reason: str) -> None:
        self.seed_total += 1
        self.missing[reason] += 1

    def add_core(self, row: dict[str, Any]) -> None:
        self.seed_total += 1
        self.feasible += 1
        self.a1_counter[(row["A1_id"], row["A1_surface"])] += 1
        self.seed_counter[(row["seed_id"], row["seed_surface"])] += 1
        self.distance_values.append(row["lowest_viable_distance"])
        self.a1_scope_values.append(row["A1_scope_size"])
        self.positive_values.append(row["positive_pre_exclusion_count"])
        self.excluded_values.append(row["excluded_scope_count"])
        self.final_positive_values.append(row["final_positive_count"])
        self.hits_b_values.append(row["hits_b_count"])
        self.fails_branch_values.append(row["fails_positive_branch_count"])
        self.outside_values.append(row["outside_A1_count"])
        self.negative_values.append(row["negative_count"])
        self.hard_values.append(row["hard_negative_count"])
        for source in row["B_sources"]:
            self.B_sources_counter[source] += 1
        for key, ok in row["task_capacity"].items():
            if ok:
                self.task_counts[key] += 1
        for threshold in POSITIVE_POOL_THRESHOLDS:
            if row["final_positive_count"] < threshold:
                continue
            counts = self.threshold_task_counts[threshold]
            if row["task_capacity"].get("T1_balanced"):
                counts["T1"] += 1
            if row["task_capacity"].get("T2_any"):
                counts["T2_any"] += 1
            if row["task_capacity"].get("T2_all_gold_sizes"):
                counts["T2_all_gold_sizes"] += 1
            if row["task_capacity"].get("T3_balanced"):
                counts["T3"] += 1
        if len(self.examples) < 5:
            self.examples.append(
                {
                    "seed": {"id": row["seed_id"], "surface": row["seed_surface"]},
                    "A1": {"id": row["A1_id"], "surface": row["A1_surface"]},
                    "A_branches": [
                        {"id": object_id, "surface": surface}
                        for object_id, surface in zip(row["A_branch_ids"], row["A_branch_surfaces"])
                    ],
                    "B": [
                        {"id": object_id, "surface": surface, "paired_A_source": paired, "B_source": source}
                        for object_id, surface, paired, source in zip(
                            row["B_ids"], row["B_surfaces"], row["paired_A_ids"], row["B_sources"]
                        )
                    ],
                    "lowest_viable_distance": row["lowest_viable_distance"],
                    "final_positive_count": row["final_positive_count"],
                    "hard_negative_count": row["hard_negative_count"],
                }
            )

    def to_row(self) -> dict[str, Any]:
        status = "available" if self.feasible else "infeasible"
        if self.feasible and self.feasible < max(10, self.seed_total // 100):
            status = "shortage"
        quota = DEFAULT_FAMILY_QUOTAS.get(self.family, 0)
        threshold_counts: dict[str, dict[str, Any]] = {}
        for threshold, counts in self.threshold_task_counts.items():
            t1 = int(counts.get("T1", 0))
            t2_any = int(counts.get("T2_any", 0))
            t2_all = int(counts.get("T2_all_gold_sizes", 0))
            t3 = int(counts.get("T3", 0))
            threshold_counts[str(threshold)] = {
                "T1": t1,
                "T2_any": t2_any,
                "T2_all_gold_sizes": t2_all,
                "T3": t3,
                "base_samples_T1_T2any_T3": t1 + t2_any + t3,
                "quota_shortage_with_current_standard": bool(quota and min(t1, t2_any, t2_all, t3) < quota),
            }
        return {
            "dataset_family": self.family,
            "a_count": self.a_count,
            "b_count": self.b_count,
            "constraint_cell": [self.a_count, self.b_count],
            "seed_total": self.seed_total,
            "feasible_seed_cell_count": self.feasible,
            "feasible_rate": _pct(self.feasible, self.seed_total),
            "status": status,
            "missing_reasons": _counter_dict(self.missing),
            "task_feasible_counts": _counter_dict(self.task_counts),
            "counts_by_min_final_positive": threshold_counts,
            "top_A1": _top_counter(self.a1_counter, self.feasible),
            "top_seed": _top_counter(self.seed_counter, self.feasible),
            "B_sources_distribution": _counter_dict(self.B_sources_counter),
            "lowest_viable_distance": _quantiles(self.distance_values),
            "A1_scope_size": _quantiles(self.a1_scope_values),
            "positive_pre_exclusion_count": _quantiles(self.positive_values),
            "excluded_scope_count": _quantiles(self.excluded_values),
            "final_positive_count": _quantiles(self.final_positive_values),
            "hits_b_count": _quantiles(self.hits_b_values),
            "fails_positive_branch_count": _quantiles(self.fails_branch_values),
            "outside_A1_count": _quantiles(self.outside_values),
            "negative_count": _quantiles(self.negative_values),
            "hard_negative_count": _quantiles(self.hard_values),
            "hard_negative_presence": {
                "ge_1": sum(1 for value in self.hard_values if value >= 1),
                "ge_2": sum(1 for value in self.hard_values if value >= 2),
                "ge_3": sum(1 for value in self.hard_values if value >= 3),
                "ge_4": sum(1 for value in self.hard_values if value >= 4),
            },
            "examples": self.examples,
        }


def _task_population(row: dict[str, Any], task: str) -> int:
    counts = row["task_feasible_counts"]
    if task == "T1":
        return int(counts.get("T1_balanced", 0))
    if task == "T2":
        return int(counts.get("T2_any", 0))
    if task == "T3":
        return int(counts.get("T3_balanced", 0))
    raise ValueError(task)


def _build_sampling_recommendations(cell_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    populations = []
    test_base = 0
    main_base = 0
    test_shortages: list[dict[str, Any]] = []
    main_shortages: list[dict[str, Any]] = []
    for row in cell_rows:
        for task in TASKS:
            population = _task_population(row, task)
            test_quota = TEST_FAMILY_QUOTAS.get(row["dataset_family"], 0)
            test_selected = min(test_quota, population)
            main_quota = DEFAULT_FAMILY_QUOTAS.get(row["dataset_family"], 0)
            main_selected = min(main_quota, population)
            test_base += test_selected
            main_base += main_selected
            if population < test_quota:
                test_shortages.append(
                    {
                        "dataset_family": row["dataset_family"],
                        "task": task,
                        "a_count": row["a_count"],
                        "b_count": row["b_count"],
                        "constraint_cell": row["constraint_cell"],
                        "population": population,
                        "quota": test_quota,
                    }
                )
            if population < main_quota:
                main_shortages.append(
                    {
                        "dataset_family": row["dataset_family"],
                        "task": task,
                        "a_count": row["a_count"],
                        "b_count": row["b_count"],
                        "constraint_cell": row["constraint_cell"],
                        "population": population,
                        "quota": main_quota,
                    }
                )
            populations.append(population)
            rows.append(
                {
                    "dataset_family": row["dataset_family"],
                    "task": task,
                    "a_count": row["a_count"],
                    "b_count": row["b_count"],
                    "constraint_cell": row["constraint_cell"],
                    "population": population,
                    "test_quota": test_quota,
                    "test_selected_base_samples": test_selected,
                    "test_quota_shortage": population < test_quota,
                    "test_inclusion_probability_if_sampled": test_selected / population if population else 0.0,
                    "test_sampling_weight_if_sampled": population / test_selected if test_selected else 0.0,
                    "main_quota": main_quota,
                    "main_selected_base_samples": main_selected,
                    "main_quota_shortage": population < main_quota,
                    "main_inclusion_probability_if_sampled": main_selected / population if population else 0.0,
                    "main_sampling_weight_if_sampled": population / main_selected if main_selected else 0.0,
                }
            )
    nonzero_min = min((value for value in populations if value > 0), default=0)
    full_base = sum(row["population"] for row in rows)
    summary = {
        "strata": len(rows),
        "minimum_nonzero_population": nonzero_min,
        "test": {
            "purpose": "engineering connectivity and data-product sanity only",
            "family_quotas": dict(TEST_FAMILY_QUOTAS),
            "base_samples": test_base,
            "rendered_requests": test_base * len(INFORMATION_LEVELS),
            "quota_shortage_count": len(test_shortages),
            "quota_shortages_top": test_shortages[:TOP_N],
            "weight_population": "post_gate_eligible_population",
        },
        "main": {
            "family_quotas": dict(DEFAULT_FAMILY_QUOTAS),
            "base_samples": main_base,
            "rendered_requests": main_base * len(INFORMATION_LEVELS),
            "quota_shortage_count": len(main_shortages),
            "quota_shortages_top": main_shortages[:TOP_N],
            "weight_population": "post_gate_eligible_population",
        },
        "full": {
            "base_samples_if_one_realization_per_feasible_seed_cell_task": full_base,
            "rendered_requests": full_base * len(INFORMATION_LEVELS),
            "sampling_weight": 1.0,
        },
    }
    return rows, summary


def _threshold_cell_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_family": row["dataset_family"],
        "a_count": row["a_count"],
        "b_count": row["b_count"],
        "constraint_cell": row["constraint_cell"],
        "feasible": row["feasible_seed_cell_count"],
        "missing": row["missing_reasons"],
        "final_positive_count": row["final_positive_count"],
        "counts_by_min_final_positive": row["counts_by_min_final_positive"],
    }


def _build_positive_pool_threshold_audit(cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_rows = [_threshold_cell_row(row) for row in cell_rows]
    families = {
        family: {
            "seed_cell_total": sum(row["seed_total"] for row in cell_rows if row["dataset_family"] == family),
            "feasible": sum(row["feasible_seed_cell_count"] for row in cell_rows if row["dataset_family"] == family),
        }
        for family in sorted({row["dataset_family"] for row in cell_rows})
    }
    threshold_summary: dict[str, Any] = {}
    for threshold in POSITIVE_POOL_THRESHOLDS:
        key = str(threshold)
        shortages: list[dict[str, Any]] = []
        family_summary: dict[str, Any] = {}
        for family in sorted(families):
            family_rows = [row for row in threshold_rows if row["dataset_family"] == family]
            quota = DEFAULT_FAMILY_QUOTAS.get(family, 0)
            min_by_task = {
                "T1": min((row["counts_by_min_final_positive"][key]["T1"] for row in family_rows), default=0),
                "T2_any": min((row["counts_by_min_final_positive"][key]["T2_any"] for row in family_rows), default=0),
                "T3": min((row["counts_by_min_final_positive"][key]["T3"] for row in family_rows), default=0),
            }
            min_t2_all = min(
                (row["counts_by_min_final_positive"][key]["T2_all_gold_sizes"] for row in family_rows),
                default=0,
            )
            shortage_count = 0
            for row in family_rows:
                counts = row["counts_by_min_final_positive"][key]
                for task_key in ("T1", "T2_any", "T3", "T2_all_gold_sizes"):
                    population = int(counts[task_key])
                    if quota and population < quota:
                        shortage_count += 1
                        shortages.append(
                            {
                                "dataset_family": family,
                                "constraint_cell": row["constraint_cell"],
                                "task": task_key,
                                "population": population,
                                "quota": quota,
                            }
                        )
            total_base = sum(row["counts_by_min_final_positive"][key]["base_samples_T1_T2any_T3"] for row in family_rows)
            family_summary[family] = {
                "min_population_by_task": min_by_task,
                "min_T2_all_gold_sizes": min_t2_all,
                "total_base_samples_after_threshold": total_base,
                "rendered_requests_after_threshold": total_base * len(INFORMATION_LEVELS),
                "quota_shortage_count": shortage_count,
            }
        threshold_summary[key] = {
            "families": family_summary,
            "shortages": shortages,
            "shortage_count_total": len(shortages),
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_name": "recipe_positive_pool_threshold_audit",
        "thresholds": list(POSITIVE_POOL_THRESHOLDS),
        "quota_policy": dict(DEFAULT_FAMILY_QUOTAS),
        "families": families,
        "threshold_summary": threshold_summary,
        "cell_rows": threshold_rows,
    }


def build_recipe_seed_audit(config_path: Path, out_dir: Path, max_seed_per_family: int = 0) -> dict[str, Any]:
    config = load_config(config_path)
    inventories = load_inventories(config.resources)
    out_dir = ensure_dir(out_dir)
    cell_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    family_summary: dict[str, Any] = {}

    for family, inventory in inventories.items():
        bitsets = BitsetHelper(inventory)
        seed_ids = tuple(sorted(inventory.objects))
        if max_seed_per_family:
            seed_ids = seed_ids[:max_seed_per_family]
        family_feasible_any: Counter[str] = Counter()
        family_a1_counter: Counter[tuple[str, str]] = Counter()
        family_distance_values: list[int] = []
        for a_count in A_COUNTS:
            for b_count in B_COUNTS:
                acc = SeedCellAccumulator(family, a_count, b_count)
                probe_cap = 0
                for seed_id in seed_ids:
                    found, reason = _find_lowest_viable_seed_core(
                        inventory,
                        bitsets,
                        seed_id,
                        a_count,
                        b_count,
                        config.resources.min_final_positive_count,
                    )
                    if not found:
                        acc.add_missing(reason)
                        continue
                    acc.add_core(found)
                    family_feasible_any[seed_id] += 1
                    family_a1_counter[(found["A1_id"], found["A1_surface"])] += 1
                    family_distance_values.append(found["lowest_viable_distance"])
                    if probe_cap < PROBE_ROWS_PER_CELL:
                        probe = {
                            "core_id": "seed_"
                            + stable_hash(
                                {
                                    "dataset_family": family,
                                    "seed_id": seed_id,
                                    "a_count": a_count,
                                    "b_count": b_count,
                                    "A1": found["A1_id"],
                                    "A_branches": found["A_branch_ids"],
                                    "B": found["B_ids"],
                                }
                            ),
                            "dataset_family": family,
                            "a_count": a_count,
                            "b_count": b_count,
                            **found,
                        }
                        probe_rows.append(probe)
                        probe_cap += 1
                cell_rows.append(acc.to_row())
        family_summary[family] = {
            "seed_count": len(seed_ids),
            "seed_with_any_feasible_cell": sum(1 for value in family_feasible_any.values() if value > 0),
            "seed_with_any_feasible_cell_rate": _pct(sum(1 for value in family_feasible_any.values() if value > 0), len(seed_ids)),
            "top_A1_across_cells": _top_counter(family_a1_counter, sum(family_a1_counter.values())),
            "lowest_viable_distance": _quantiles(family_distance_values),
        }

    sampling_rows, sampling_summary = _build_sampling_recommendations(cell_rows)
    positive_pool_threshold_audit = _build_positive_pool_threshold_audit(cell_rows)
    write_jsonl(out_dir / "lowest_viable_cells.jsonl", cell_rows)
    write_jsonl(out_dir / "lowest_viable_core_probe_frame.jsonl", probe_rows)
    write_jsonl(out_dir / "sampling_frame.jsonl", sampling_rows)
    write_json(out_dir / "positive_pool_threshold_audit.json", positive_pool_threshold_audit)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_name": "recipe_seed_object_first_audit",
        "config_path": str(config_path),
        "policy": {
            "entry": "seed_object_first_nearest_viable_A1",
            "A_semantics": "A1 base plus optional OR branches A2...Ak",
            "B_policy": "prefer internal branches that do not exclude seed object",
            "candidate_policy": "audits generation capacity; freeze assigns concrete candidate ids",
            "information_policy": "paired I0/I1/I2",
        },
        "families": family_summary,
        "cell_count": len(cell_rows),
        "probe_core_rows": len(probe_rows),
        "sampling_summary": sampling_summary,
        "positive_pool_threshold_summary": positive_pool_threshold_audit["threshold_summary"],
        "outputs": {
            "summary": str(out_dir / "recipe_seed_audit_summary.json"),
            "lowest_viable_cells": str(out_dir / "lowest_viable_cells.jsonl"),
            "lowest_viable_core_probe_frame": str(out_dir / "lowest_viable_core_probe_frame.jsonl"),
            "sampling_frame": str(out_dir / "sampling_frame.jsonl"),
            "positive_pool_threshold_audit": str(out_dir / "positive_pool_threshold_audit.json"),
        },
    }
    write_json(out_dir / "recipe_seed_audit_summary.json", summary)
    return summary


def write_recipe_seed_audit(config_path: Path, out_dir: Path, max_seed_per_family: int = 0) -> dict[str, Any]:
    return build_recipe_seed_audit(config_path, out_dir, max_seed_per_family=max_seed_per_family)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit seed-object-first current recipe sampling frame.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/test.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("audits/recipe_seed"))
    parser.add_argument("--max-seed-per-family", type=int, default=0)
    args = parser.parse_args()
    summary = write_recipe_seed_audit(args.config, args.out_dir, max_seed_per_family=args.max_seed_per_family)
    print(
        {
            "summary": summary["outputs"]["summary"],
            "probe_core_rows": summary["probe_core_rows"],
            "main_rendered_requests": summary["sampling_summary"]["main"]["rendered_requests"],
            "full_rendered_requests": summary["sampling_summary"]["full"]["rendered_requests"],
        }
    )


if __name__ == "__main__":
    main()
