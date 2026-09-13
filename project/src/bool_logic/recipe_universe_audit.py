from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

from .config import load_config
from .constants import A_COUNTS, B_COUNTS, INFORMATION_LEVELS, POOL_SIZE, TASKS, T2_GOLD_SET_SIZES
from .io_utils import canonical_json, ensure_dir, stable_hash, write_json, write_jsonl
from .resources import Inventory, load_inventories
from .taxonomy_bitsets import BitsetHelper


TOP_N = 12
PROBE_CORE_ROWS_PER_CELL = 80
TEST_FAMILY_QUOTAS = {"wordnet": 3, "babelnet": 1}
MAIN_FAMILY_QUOTAS = {"wordnet": 650, "babelnet": 100}


def _pct(part: int, total: int) -> float:
    return part / total if total else 0.0


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(values)

    def pick(q: float) -> int:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _top_counter(counter: Counter[Any], total: int, limit: int = TOP_N) -> list[dict[str, Any]]:
    out = []
    for key, count in counter.most_common(limit):
        item: dict[str, Any] = {"count": int(count), "share": _pct(count, total)}
        if isinstance(key, tuple):
            item["key"] = list(key)
        else:
            item["key"] = key
        out.append(item)
    return out


def _safe_comb(n: int, k: int) -> int:
    return comb(n, k) if n >= k >= 0 else 0


def _direct_children(bitsets: BitsetHelper, object_id: str) -> tuple[str, ...]:
    return bitsets.usable_children(object_id)


def _branch_b_set_count(branch_b_option_counts: list[int], branch_count: int, b_count: int) -> int:
    """Exact count of (positive branch set, internal B set) pairs.

    For a selected positive branch subset of size k, B can be selected from the
    union of direct usable children under those selected branches. This computes:
    sum_{branch subset size k} C(sum(child_counts in subset), b_count)
    without enumerating branch subsets.
    """
    if branch_count == 0:
        return 0
    dp = [[0 for _ in range(b_count + 1)] for _ in range(branch_count + 1)]
    dp[0][0] = 1
    for option_count in branch_b_option_counts:
        next_dp = [row[:] for row in dp]
        choices = [_safe_comb(option_count, take) for take in range(b_count + 1)]
        for used_branches in range(branch_count):
            for used_b in range(b_count + 1):
                base = dp[used_branches][used_b]
                if not base:
                    continue
                for take_b in range(b_count - used_b + 1):
                    next_dp[used_branches + 1][used_b + take_b] += base * choices[take_b]
        dp = next_dp
    return dp[branch_count][b_count]


def _spread_ids(values: tuple[str, ...], count: int, bitsets: BitsetHelper) -> tuple[str, ...]:
    if count <= 0:
        return ()
    if len(values) <= count:
        return tuple(values)
    ordered = sorted(values, key=lambda item: (bitsets.closure_size(item), item))
    if count == 1:
        return (ordered[len(ordered) // 2],)
    indexes = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return tuple(ordered[index] for index in indexes)


def _probe_branch_sets(bitsets: BitsetHelper, a1_id: str, branch_count: int) -> tuple[tuple[str, ...], ...]:
    if branch_count == 0:
        return ((),)
    children = _direct_children(bitsets, a1_id)
    if len(children) < branch_count:
        return ()
    largest = tuple(sorted(children, key=lambda item: (-bitsets.closure_size(item), item))[:branch_count])
    smallest = tuple(sorted(children, key=lambda item: (bitsets.closure_size(item), item))[:branch_count])
    lexical = tuple(sorted(children)[:branch_count])
    spread = _spread_ids(children, branch_count, bitsets)
    hashed = tuple(sorted(children, key=lambda item: stable_hash({"a1": a1_id, "child": item}))[:branch_count])
    out = []
    seen: set[tuple[str, ...]] = set()
    for branch_set in (largest, smallest, lexical, spread, hashed):
        normalized = tuple(sorted(branch_set))
        if len(normalized) == branch_count and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def _b_options_for_core(bitsets: BitsetHelper, a1_id: str, branch_ids: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    if not branch_ids:
        for child_id in _direct_children(bitsets, a1_id):
            if child_id not in seen:
                seen.add(child_id)
                options.append({"B_id": child_id, "paired_A_source": a1_id, "B_source": "inside_A1"})
        return tuple(options)
    for branch_id in branch_ids:
        for child_id in _direct_children(bitsets, branch_id):
            if child_id in seen:
                continue
            seen.add(child_id)
            options.append({"B_id": child_id, "paired_A_source": branch_id, "B_source": "inside_positive_branch"})
    return tuple(options)


def _choose_b_options(bitsets: BitsetHelper, options: tuple[dict[str, str], ...], b_count: int) -> tuple[dict[str, str], ...] | None:
    if b_count == 0:
        return ()
    if len(options) < b_count:
        return None
    return tuple(sorted(options, key=lambda item: (-bitsets.closure_size(item["B_id"]), item["paired_A_source"], item["B_id"]))[:b_count])


def _core_capacity(
    bitsets: BitsetHelper,
    a1_id: str,
    branch_ids: tuple[str, ...],
    b_options: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    a1_bits = bitsets.closure_bits(a1_id)
    if branch_ids:
        positive_pre_bits = 0
        for branch_id in branch_ids:
            positive_pre_bits |= bitsets.closure_bits(branch_id)
        positive_pre_bits &= a1_bits
    else:
        positive_pre_bits = a1_bits

    b_bits = 0
    for option in b_options:
        b_bits |= bitsets.closure_bits(option["B_id"])
    hits_b_bits = positive_pre_bits & b_bits
    positive_bits = positive_pre_bits & ~b_bits

    if branch_ids:
        fails_positive_branch_bits = a1_bits & ~positive_pre_bits
        near_sibling_bits = fails_positive_branch_bits
    else:
        fails_positive_branch_bits = 0
        near_sibling_bits = bitsets.sibling_bits(a1_id) & ~positive_pre_bits & ~hits_b_bits

    outside_a1_bits = bitsets.all_mask & ~a1_bits & ~bitsets.bit(a1_id)
    negative_bits = hits_b_bits | fails_positive_branch_bits | near_sibling_bits | outside_a1_bits
    hard_negative_bits = hits_b_bits | fails_positive_branch_bits | near_sibling_bits

    positive_count = positive_bits.bit_count()
    negative_count = negative_bits.bit_count()
    hard_count = hard_negative_bits.bit_count()
    task_capacity = {
        "T1_true": positive_count >= 1,
        "T1_false": negative_count >= 1,
        "T1_balanced": positive_count >= 1 and negative_count >= 1,
        "T2_gold_0": negative_count >= POOL_SIZE,
        "T2_gold_1": positive_count >= 1 and negative_count >= POOL_SIZE - 1,
        "T2_gold_2": positive_count >= 2 and negative_count >= POOL_SIZE - 2,
        "T2_any": (
            negative_count >= POOL_SIZE
            or (positive_count >= 1 and negative_count >= POOL_SIZE - 1)
            or (positive_count >= 2 and negative_count >= POOL_SIZE - 2)
        ),
        "T2_all_gold_sizes": (
            negative_count >= POOL_SIZE
            and positive_count >= 1
            and negative_count >= POOL_SIZE - 1
            and positive_count >= 2
            and negative_count >= POOL_SIZE - 2
        ),
        "T2_hard_gold_0": hard_count >= POOL_SIZE,
        "T2_hard_gold_1": positive_count >= 1 and hard_count >= POOL_SIZE - 1,
        "T2_hard_gold_2": positive_count >= 2 and hard_count >= POOL_SIZE - 2,
        "T3_true": positive_count >= 1 and negative_count >= POOL_SIZE - 1,
        "T3_false": negative_count >= POOL_SIZE,
        "T3_balanced": positive_count >= 1 and negative_count >= POOL_SIZE,
        "T3_hard_true": positive_count >= 1 and hard_count >= POOL_SIZE - 1,
        "T3_hard_false": hard_count >= POOL_SIZE,
    }
    return {
        "a1_scope_size": a1_bits.bit_count(),
        "positive_pre_exclusion_count": positive_pre_bits.bit_count(),
        "excluded_scope_count": b_bits.bit_count(),
        "hits_b_count": hits_b_bits.bit_count(),
        "final_positive_count": positive_count,
        "fails_positive_branch_count": fails_positive_branch_bits.bit_count(),
        "near_sibling_count": near_sibling_bits.bit_count(),
        "outside_a1_count": outside_a1_bits.bit_count(),
        "negative_count": negative_count,
        "hard_negative_count": hard_count,
        "task_capacity": task_capacity,
    }


class CellAccumulator:
    def __init__(self, family: str, a_count: int, b_count: int):
        self.family = family
        self.a_count = a_count
        self.b_count = b_count
        self.branch_count = a_count - 1
        self.a1_candidates = 0
        self.structural_branch_set_count = 0
        self.structural_core_count = 0
        self.probe_core_count = 0
        self.probe_missing: Counter[str] = Counter()
        self.task_counts: Counter[str] = Counter()
        self.a1_counter: Counter[tuple[str, str]] = Counter()
        self.B_sources_counter: Counter[str] = Counter()
        self.paired_slots_values: list[int] = []
        self.a1_scope_values: list[int] = []
        self.branch_option_values: list[int] = []
        self.b_option_values: list[int] = []
        self.positive_pre_values: list[int] = []
        self.excluded_values: list[int] = []
        self.final_positive_values: list[int] = []
        self.hits_b_values: list[int] = []
        self.fails_branch_values: list[int] = []
        self.outside_a1_values: list[int] = []
        self.negative_values: list[int] = []
        self.hard_values: list[int] = []
        self.examples: list[dict[str, Any]] = []

    def add_structural(self, branch_option_count: int, branch_set_count: int, core_count: int) -> None:
        self.a1_candidates += 1
        self.branch_option_values.append(branch_option_count)
        self.structural_branch_set_count += branch_set_count
        self.structural_core_count += core_count

    def add_missing(self, reason: str) -> None:
        self.probe_missing[reason] += 1

    def add_probe(
        self,
        inventory: Inventory,
        a1_id: str,
        branch_ids: tuple[str, ...],
        b_options: tuple[dict[str, str], ...],
        capacity: dict[str, Any],
    ) -> None:
        self.probe_core_count += 1
        self.a1_counter[(a1_id, inventory.get(a1_id).surface)] += 1
        self.paired_slots_values.append(1 + len(branch_ids))
        self.a1_scope_values.append(capacity["a1_scope_size"])
        self.b_option_values.append(len(b_options))
        self.positive_pre_values.append(capacity["positive_pre_exclusion_count"])
        self.excluded_values.append(capacity["excluded_scope_count"])
        self.final_positive_values.append(capacity["final_positive_count"])
        self.hits_b_values.append(capacity["hits_b_count"])
        self.fails_branch_values.append(capacity["fails_positive_branch_count"])
        self.outside_a1_values.append(capacity["outside_a1_count"])
        self.negative_values.append(capacity["negative_count"])
        self.hard_values.append(capacity["hard_negative_count"])
        for option in b_options:
            self.B_sources_counter[option["B_source"]] += 1
        for key, ok in capacity["task_capacity"].items():
            if ok:
                self.task_counts[key] += 1
        if len(self.examples) < 5:
            self.examples.append(
                {
                    "A1": {"id": a1_id, "surface": inventory.get(a1_id).surface},
                    "A_branches": [{"id": item, "surface": inventory.get(item).surface} for item in branch_ids],
                    "B": [
                        {
                            "id": option["B_id"],
                            "surface": inventory.get(option["B_id"]).surface,
                            "paired_A_source": option["paired_A_source"],
                            "B_source": option["B_source"],
                        }
                        for option in b_options
                    ],
                    "positive_pre_exclusion_count": capacity["positive_pre_exclusion_count"],
                    "final_positive_count": capacity["final_positive_count"],
                    "hits_b_count": capacity["hits_b_count"],
                    "hard_negative_count": capacity["hard_negative_count"],
                }
            )

    def to_row(self) -> dict[str, Any]:
        status = "available" if self.probe_core_count else "shortage"
        if not self.structural_core_count:
            status = "infeasible"
        return {
            "dataset_family": self.family,
            "a_count": self.a_count,
            "b_count": self.b_count,
            "constraint_cell": [self.a_count, self.b_count],
            "branch_count": self.branch_count,
            "a1_candidate_count": self.a1_candidates,
            "structural_a_branch_set_count": self.structural_branch_set_count,
            "structural_constraint_core_count": self.structural_core_count,
            "probe_core_count": self.probe_core_count,
            "probe_status": status,
            "probe_missing_reasons": _counter_dict(self.probe_missing),
            "task_feasible_probe_counts": _counter_dict(self.task_counts),
            "top_A1_by_probe_count": _top_counter(self.a1_counter, self.probe_core_count),
            "B_sources_probe_distribution": _counter_dict(self.B_sources_counter),
            "paired_slot_count": _quantiles(self.paired_slots_values),
            "branch_option_count": _quantiles(self.branch_option_values),
            "b_option_count": _quantiles(self.b_option_values),
            "A1_scope_size": _quantiles(self.a1_scope_values),
            "positive_pre_exclusion_count": _quantiles(self.positive_pre_values),
            "excluded_scope_count": _quantiles(self.excluded_values),
            "final_positive_count": _quantiles(self.final_positive_values),
            "hits_b_count": _quantiles(self.hits_b_values),
            "fails_positive_branch_count": _quantiles(self.fails_branch_values),
            "outside_A1_count": _quantiles(self.outside_a1_values),
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


def _relation_graph_audit(inventory: Inventory, bitsets: BitsetHelper) -> dict[str, Any]:
    child_counts: list[int] = []
    parent_counts: list[int] = []
    closure_sizes: list[int] = []
    top_children: Counter[tuple[str, str]] = Counter()
    top_closure: Counter[tuple[str, str]] = Counter()
    roots = 0
    leaves = 0
    multi_parent = 0
    for object_id in inventory.objects:
        child_count = len(_direct_children(bitsets, object_id))
        parent_count = len(bitsets.usable_parents(object_id))
        closure_size = bitsets.closure_size(object_id)
        child_counts.append(child_count)
        parent_counts.append(parent_count)
        closure_sizes.append(closure_size)
        top_children[(object_id, inventory.get(object_id).surface)] = child_count
        top_closure[(object_id, inventory.get(object_id).surface)] = closure_size
        if parent_count == 0:
            roots += 1
        if child_count == 0:
            leaves += 1
        if parent_count > 1:
            multi_parent += 1
    return {
        "usable_object_count": len(inventory.objects),
        "audit_stats": inventory.audit_stats,
        "root_count": roots,
        "leaf_count": leaves,
        "multi_parent_count": multi_parent,
        "direct_child_count": _quantiles(child_counts),
        "parent_count": _quantiles(parent_counts),
        "closure_size": _quantiles(closure_sizes),
        "top_direct_child_nodes": _top_counter(top_children, sum(child_counts) or 1),
        "top_closure_nodes": _top_counter(top_closure, sum(closure_sizes) or 1),
    }


def _structural_counts_for_a1(bitsets: BitsetHelper, a1_id: str, branch_count: int, b_count: int) -> tuple[int, int, int]:
    children = _direct_children(bitsets, a1_id)
    if bitsets.closure_size(a1_id) <= 0:
        return len(children), 0, 0
    if branch_count == 0:
        branch_set_count = 1
        core_count = _safe_comb(len(children), b_count)
        return len(children), branch_set_count, core_count
    if len(children) < branch_count:
        return len(children), 0, 0
    branch_set_count = _safe_comb(len(children), branch_count)
    branch_b_option_counts = [len(_direct_children(bitsets, child_id)) for child_id in children]
    core_count = _branch_b_set_count(branch_b_option_counts, branch_count, b_count)
    return len(children), branch_set_count, core_count


def _write_object_inventory(path: Path, inventories: dict[str, Inventory], bitsets_by_family: dict[str, BitsetHelper]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for family, inventory in inventories.items():
            bitsets = bitsets_by_family[family]
            for object_id in sorted(inventory.objects):
                obj = inventory.get(object_id)
                row = {
                    "dataset_family": family,
                    "object_id": object_id,
                    "surface": obj.surface,
                    "gloss_available": bool(obj.gloss),
                    "synonym_count": len(obj.synonyms),
                    "direct_usable_child_count": len(_direct_children(bitsets, object_id)),
                    "usable_parent_count": len(bitsets.usable_parents(object_id)),
                    "closure_size": bitsets.closure_size(object_id),
                    "usable_flags": obj.usable_flags,
                }
                handle.write(canonical_json(row))
                handle.write("\n")


def _task_population(cell_row: dict[str, Any], task: str) -> int:
    counts = cell_row["task_feasible_probe_counts"]
    scale_source = max(1, cell_row["probe_core_count"])
    structural = int(cell_row["structural_constraint_core_count"])
    if task == "T1":
        probe = int(counts.get("T1_balanced", 0))
    elif task == "T2":
        probe = int(counts.get("T2_any", 0))
    elif task == "T3":
        probe = int(counts.get("T3_balanced", 0))
    else:
        raise ValueError(task)
    return int(structural * _pct(probe, scale_source))


def _build_sampling_frame(cell_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "test": {
            "status": "top_down_reference_only",
            "family_quotas": dict(TEST_FAMILY_QUOTAS),
            "information_policy": "paired I0/I1/I2 per base_sample_id",
        },
        "main": {
            "status": "top_down_reference_only",
            "family_quotas": dict(MAIN_FAMILY_QUOTAS),
            "information_policy": "paired I0/I1/I2 per base_sample_id",
        },
        "full": {
            "status": "structural universe estimate, not an API target",
            "base_samples_estimate": 0,
            "rendered_requests_estimate": 0,
        },
    }
    test_shortages: list[dict[str, Any]] = []
    main_shortages: list[dict[str, Any]] = []
    extended_base = 0
    for row in cell_rows:
        for task in TASKS:
            population = _task_population(row, task)
            extended_base += population
            test_quota = TEST_FAMILY_QUOTAS.get(row["dataset_family"], 0)
            main_quota = MAIN_FAMILY_QUOTAS.get(row["dataset_family"], 0)
            test_selected = min(test_quota, population)
            main_selected = min(main_quota, population)
            item = {
                "dataset_family": row["dataset_family"],
                "task": task,
                "a_count": row["a_count"],
                "b_count": row["b_count"],
                "constraint_cell": row["constraint_cell"],
                "estimated_population": population,
                "test_reference_quota": test_selected,
                "test_quota_shortage": population < test_quota,
                "test_inclusion_probability_if_sampled": test_selected / population if population else 0.0,
                "test_sampling_weight_if_sampled": population / test_selected if test_selected else 0.0,
                "main_reference_quota": main_selected,
                "main_quota_shortage": population < main_quota,
                "main_inclusion_probability_if_sampled": main_selected / population if population else 0.0,
                "main_sampling_weight_if_sampled": population / main_selected if main_selected else 0.0,
            }
            rows.append(item)
            if population < test_quota:
                test_shortages.append(item)
            if population < main_quota:
                main_shortages.append(item)
    test_base = sum(row["test_reference_quota"] for row in rows)
    main_base = sum(row["main_reference_quota"] for row in rows)
    summary["test"].update(
        {
            "base_samples": test_base,
            "rendered_requests": test_base * len(INFORMATION_LEVELS),
            "quota_shortage_count": len(test_shortages),
            "quota_shortages_top": test_shortages[:TOP_N],
        }
    )
    summary["main"].update(
        {
            "base_samples": main_base,
            "rendered_requests": main_base * len(INFORMATION_LEVELS),
            "quota_shortage_count": len(main_shortages),
            "quota_shortages_top": main_shortages[:TOP_N],
        }
    )
    summary["full"].update(
        {
            "base_samples_estimate": extended_base,
            "rendered_requests_estimate": extended_base * len(INFORMATION_LEVELS),
        }
    )
    return rows, summary


def build_recipe_universe_audit(config_path: Path, out_dir: Path, max_a1_per_family: int = 0) -> dict[str, Any]:
    config = load_config(config_path)
    inventories = load_inventories(config.resources)
    out_dir = ensure_dir(out_dir)
    bitsets_by_family = {family: BitsetHelper(inventory) for family, inventory in inventories.items()}
    relation_audits = {
        family: _relation_graph_audit(inventory, bitsets_by_family[family]) for family, inventory in inventories.items()
    }

    object_inventory_path = out_dir / "object_inventory.jsonl"
    _write_object_inventory(object_inventory_path, inventories, bitsets_by_family)

    cell_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for family, inventory in inventories.items():
        bitsets = bitsets_by_family[family]
        a1_ids = tuple(sorted(inventory.objects))
        if max_a1_per_family:
            a1_ids = a1_ids[:max_a1_per_family]
        for a_count in A_COUNTS:
            branch_count = a_count - 1
            for b_count in B_COUNTS:
                acc = CellAccumulator(family, a_count, b_count)
                row_cap = 0
                for a1_id in a1_ids:
                    branch_option_count, branch_set_count, core_count = _structural_counts_for_a1(
                        bitsets, a1_id, branch_count, b_count
                    )
                    if branch_set_count:
                        acc.add_structural(branch_option_count, branch_set_count, core_count)
                    if not core_count:
                        continue
                    branch_sets = _probe_branch_sets(bitsets, a1_id, branch_count)
                    if not branch_sets:
                        acc.add_missing("insufficient_A_branch_options")
                        continue
                    for branch_ids in branch_sets:
                        b_options = _b_options_for_core(bitsets, a1_id, branch_ids)
                        chosen_b = _choose_b_options(bitsets, b_options, b_count)
                        if chosen_b is None:
                            acc.add_missing("insufficient_B_options")
                            continue
                        capacity = _core_capacity(bitsets, a1_id, branch_ids, chosen_b)
                        if capacity["positive_pre_exclusion_count"] <= 0:
                            acc.add_missing("empty_positive_pre_exclusion_scope")
                            continue
                        if capacity["final_positive_count"] <= 0:
                            acc.add_missing("empty_final_positive_scope")
                            continue
                        acc.add_probe(inventory, a1_id, branch_ids, chosen_b, capacity)
                        if row_cap < PROBE_CORE_ROWS_PER_CELL:
                            probe_row = {
                                "core_id": "core_"
                                + stable_hash(
                                    {
                                        "dataset_family": family,
                                        "A1": a1_id,
                                        "A_branches": branch_ids,
                                        "B": [item["B_id"] for item in chosen_b],
                                        "a_count": a_count,
                                        "b_count": b_count,
                                    }
                                ),
                                "dataset_family": family,
                                "a_count": a_count,
                                "b_count": b_count,
                                "A1_id": a1_id,
                                "A1_surface": inventory.get(a1_id).surface,
                                "A_branch_ids": list(branch_ids),
                                "A_branch_surfaces": [inventory.get(item).surface for item in branch_ids],
                                "B_ids": [item["B_id"] for item in chosen_b],
                                "B_surfaces": [inventory.get(item["B_id"]).surface for item in chosen_b],
                                "paired_A_ids": [item["paired_A_source"] for item in chosen_b],
                                "B_sources": [item["B_source"] for item in chosen_b],
                                **{key: value for key, value in capacity.items() if key != "task_capacity"},
                                "task_capacity": capacity["task_capacity"],
                            }
                            probe_rows.append(probe_row)
                            row_cap += 1
                cell_rows.append(acc.to_row())

    sampling_rows, sampling_summary = _build_sampling_frame(cell_rows)
    resource_snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "audit_name": "recipe_universe_audit",
        "topic_definition": "A1 base concept plus optional OR positive branches A2...Ak, minus audited B exclusions",
        "truth_scope_policy": {
            "primary": "non_reflexive_descendant_scope_to_match_current_runtime_resources",
            "note": "Reflexive membership is outside the current test/main/full product definition.",
        },
        "matrix": {"a_count": list(A_COUNTS), "b_count": list(B_COUNTS), "tasks": list(TASKS), "information": list(INFORMATION_LEVELS)},
        "relation_graph_audit": relation_audits,
    }
    write_json(out_dir / "resource_snapshot.json", resource_snapshot)
    write_jsonl(out_dir / "constraint_core_universe.jsonl", cell_rows)
    write_jsonl(out_dir / "constraint_core_probe_frame.jsonl", probe_rows)
    write_jsonl(out_dir / "candidate_capacity.jsonl", cell_rows)
    feasibility_rows = []
    for row in cell_rows:
        for task in TASKS:
            population = _task_population(row, task)
            quota = MAIN_FAMILY_QUOTAS.get(row["dataset_family"], 0)
            status = "available" if population >= quota else ("shortage" if population else "infeasible")
            feasibility_rows.append(
                {
                    "dataset_family": row["dataset_family"],
                    "task": task,
                    "constraint_cell": row["constraint_cell"],
                    "a_count": row["a_count"],
                    "b_count": row["b_count"],
                    "information": "paired_I0_I1_I2",
                    "estimated_population": population,
                    "main_quota": quota,
                    "status": status,
                    "reason": "probe_scaled_population" if population else "no_probe_feasible_core",
                    "structural_constraint_core_count": row["structural_constraint_core_count"],
                    "probe_core_count": row["probe_core_count"],
                    "hard_negative_count": row["hard_negative_count"],
                    "final_positive_count": row["final_positive_count"],
                }
            )
    write_jsonl(out_dir / "feasibility_table.jsonl", feasibility_rows)
    write_jsonl(out_dir / "sampling_frame.jsonl", sampling_rows)

    summary = {
        "created_at": resource_snapshot["created_at"],
        "outputs": {
            "resource_snapshot": str(out_dir / "resource_snapshot.json"),
            "object_inventory": str(object_inventory_path),
            "constraint_core_universe": str(out_dir / "constraint_core_universe.jsonl"),
            "constraint_core_probe_frame": str(out_dir / "constraint_core_probe_frame.jsonl"),
            "candidate_capacity": str(out_dir / "candidate_capacity.jsonl"),
            "feasibility_table": str(out_dir / "feasibility_table.jsonl"),
            "sampling_frame": str(out_dir / "sampling_frame.jsonl"),
            "summary": str(out_dir / "recipe_universe_summary.json"),
        },
        "resource_snapshot": resource_snapshot,
        "cell_count": len(cell_rows),
        "probe_core_rows": len(probe_rows),
        "structural_constraint_core_count": sum(int(row["structural_constraint_core_count"]) for row in cell_rows),
        "sampling_summary": sampling_summary,
        "current_policy": {
            "truth_scope": "non_reflexive_descendant_closure",
            "B_structure": "paired_A_ids are provenance fields, not a matrix axis",
            "main_sampling": "post_gate_seed_object_first_frame",
            "closure_policy": "coverage field only, no hard closure band filter",
        },
    }
    write_json(out_dir / "recipe_universe_summary.json", summary)
    return summary


def write_recipe_universe_audit(config_path: Path, out_dir: Path, max_a1_per_family: int = 0) -> dict[str, Any]:
    return build_recipe_universe_audit(config_path, out_dir, max_a1_per_family=max_a1_per_family)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the current recipe generation universe under A1 + OR-branches - B semantics.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/test.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("audits/recipe_universe"))
    parser.add_argument("--max-a1-per-family", type=int, default=0)
    args = parser.parse_args()
    summary = write_recipe_universe_audit(args.config, args.out_dir, max_a1_per_family=args.max_a1_per_family)
    print(
        {
            "summary": summary["outputs"]["summary"],
            "structural_constraint_core_count": summary["structural_constraint_core_count"],
            "probe_core_rows": summary["probe_core_rows"],
            "main_rendered_requests": summary["sampling_summary"]["main"]["rendered_requests"],
            "full_rendered_requests_estimate": summary["sampling_summary"]["full"]["rendered_requests_estimate"],
        }
    )


if __name__ == "__main__":
    main()
