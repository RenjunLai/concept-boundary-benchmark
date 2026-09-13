from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from .config import ExperimentConfig
from .constants import POOL_SIZE
from .io_utils import to_plain
from .recipe_seed_audit import _find_lowest_viable_seed_core
from .resources import Inventory, load_inventories
from .taxonomy_bitsets import BitsetHelper


@dataclass(frozen=True)
class FeasibilityRow:
    dataset_family: str
    task: str
    constraint_cell: tuple[int, int]
    a_count: int
    b_count: int
    information_policy: str
    closure_size_distribution: dict[str, int]
    usable_seed_count: int
    feasible_seed_cell_count: int
    usable_candidate_count: int
    direct_usable_child_count: int
    B_sources: tuple[str, ...]
    available_primary_negative_source: tuple[str, ...]
    hard_negative_available: bool
    easy_negative_available: bool
    t1_capacity: int
    t2_pool_space_estimate: int
    t3_pool_space_estimate: int
    status: str
    reason: str
    surface_coverage: float = 1.0
    gloss_coverage: float = 1.0
    final_positive_min: int = 0
    final_positive_max: int = 0
    negative_pool_min: int = 0
    negative_pool_max: int = 0
    recommended_pool_size: int = POOL_SIZE
    budget_seed_limit: int = 0
    budget_max_base_samples: int = 0


def _closure_size_distribution(inventory: Inventory, bitsets: BitsetHelper) -> dict[str, int]:
    buckets = {"0": 0, "1-19": 0, "20-99": 0, "100-499": 0, "500-1999": 0, "2000+": 0}
    for object_id in inventory.objects:
        size = bitsets.closure_size(object_id)
        if size == 0:
            buckets["0"] += 1
        elif size < 20:
            buckets["1-19"] += 1
        elif size < 100:
            buckets["20-99"] += 1
        elif size < 500:
            buckets["100-499"] += 1
        elif size < 2000:
            buckets["500-1999"] += 1
        else:
            buckets["2000+"] += 1
    return buckets


def _candidate_space(bitsets: BitsetHelper, row: dict[str, Any]) -> dict[str, int]:
    final_positive = int(row["final_positive_count"])
    negative = int(row["negative_count"])
    hard = int(row["hard_negative_count"])
    return {
        "t1_capacity": final_positive + negative,
        "t2_pool_space_estimate": max(0, negative >= POOL_SIZE)
        + max(0, final_positive >= 1 and negative >= POOL_SIZE - 1)
        + max(0, final_positive >= 2 and negative >= POOL_SIZE - 2),
        "t3_pool_space_estimate": int(final_positive >= 1 and negative >= POOL_SIZE - 1) + int(negative >= POOL_SIZE),
        "hard_available": hard >= 1,
        "easy_available": int(row["outside_A1_count"]) >= POOL_SIZE,
    }


def build_resource_audit(config: ExperimentConfig, inventories: dict[str, Inventory]) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for family, inventory in inventories.items():
        bitsets = BitsetHelper(inventory)
        resources[family] = {
            "audit_stats": inventory.audit_stats,
            "usable_object_count": len(inventory),
            "closure_size_distribution": _closure_size_distribution(inventory, bitsets),
            "max_seed_per_family": config.resources.max_seed_per_family,
            "min_final_positive_count": config.resources.min_final_positive_count,
        }
    return {
        "experiment": config.name,
        "matrix": to_plain(config.matrix),
        "information_policy": "paired_I0_I1_I2_per_base_sample_id",
        "resources": resources,
    }


def build_feasibility(config: ExperimentConfig, inventories: dict[str, Inventory]) -> list[FeasibilityRow]:
    rows: list[FeasibilityRow] = []
    for family, inventory in inventories.items():
        bitsets = BitsetHelper(inventory)
        seed_ids = tuple(sorted(inventory.objects))[: config.resources.max_seed_per_family or None]
        closure_distribution = _closure_size_distribution(inventory, bitsets)
        direct_child_max = max((len(bitsets.usable_children(object_id)) for object_id in inventory.objects), default=0)
        for a_count, b_count in product(config.matrix.a_count, config.matrix.b_count):
            feasible_rows: list[dict[str, Any]] = []
            missing_reasons: dict[str, int] = {}
            for seed_id in seed_ids:
                found, reason = _find_lowest_viable_seed_core(
                    inventory,
                    bitsets,
                    seed_id,
                    a_count,
                    b_count,
                    config.resources.min_final_positive_count,
                )
                if found:
                    feasible_rows.append(found)
                else:
                    missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
            final_positive_values = [int(row["final_positive_count"]) for row in feasible_rows]
            negative_values = [int(row["negative_count"]) for row in feasible_rows]
            source_values = sorted({source for row in feasible_rows for source in row["B_sources"]})
            negative_sources = {"outside_A1"}
            if any(int(row["hits_b_count"]) > 0 for row in feasible_rows):
                negative_sources.add("hits_B")
            if any(int(row["fails_positive_branch_count"]) > 0 for row in feasible_rows):
                negative_sources.add("fails_positive_branch")
            capacity = [_candidate_space(bitsets, row) for row in feasible_rows]
            feasible_count = len(feasible_rows)
            if feasible_count == 0:
                status = "infeasible"
                reason = ",".join(sorted(missing_reasons)) or "no_feasible_seed_cell"
            elif feasible_count < max(1, len(seed_ids) // 100):
                status = "shortage"
                reason = "low_feasible_seed_cell_count"
            else:
                status = "available"
                reason = "ok"
            for task in config.tasks:
                rows.append(
                    FeasibilityRow(
                        dataset_family=family,
                        task=task,
                        constraint_cell=(a_count, b_count),
                        a_count=a_count,
                        b_count=b_count,
                        information_policy="paired_I0_I1_I2_per_base_sample_id",
                        closure_size_distribution=closure_distribution,
                        usable_seed_count=len(seed_ids),
                        feasible_seed_cell_count=feasible_count,
                        usable_candidate_count=sum(final_positive_values) + sum(negative_values),
                        direct_usable_child_count=direct_child_max,
                        B_sources=tuple(source_values),
                        available_primary_negative_source=tuple(sorted(negative_sources)),
                        hard_negative_available=any(item["hard_available"] for item in capacity),
                        easy_negative_available=any(item["easy_available"] for item in capacity),
                        t1_capacity=sum(item["t1_capacity"] for item in capacity) if task == "T1" else 0,
                        t2_pool_space_estimate=sum(item["t2_pool_space_estimate"] for item in capacity),
                        t3_pool_space_estimate=sum(item["t3_pool_space_estimate"] for item in capacity),
                        status=status,
                        reason=reason,
                        surface_coverage=float(inventory.audit_stats.get("surface_coverage", 1.0)),
                        gloss_coverage=float(inventory.audit_stats.get("gloss_coverage", 1.0)),
                        final_positive_min=min(final_positive_values) if final_positive_values else 0,
                        final_positive_max=max(final_positive_values) if final_positive_values else 0,
                        negative_pool_min=min(negative_values) if negative_values else 0,
                        negative_pool_max=max(negative_values) if negative_values else 0,
                        budget_seed_limit=config.resources.max_seed_per_family,
                        budget_max_base_samples=config.budgets.max_base_samples,
                    )
                )
    return rows


def audit_experiment(config: ExperimentConfig) -> tuple[dict[str, Any], list[FeasibilityRow], dict[str, Inventory]]:
    inventories = load_inventories(config.resources)
    return build_resource_audit(config, inventories), build_feasibility(config, inventories), inventories
