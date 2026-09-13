from __future__ import annotations

from typing import Any

from .config import ExperimentConfig


def family_quota_map(config: ExperimentConfig) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for family in config.resources.dataset_families:
        if family == "wordnet":
            quotas[family] = config.product.wordnet_quota_per_stratum
        elif family == "babelnet":
            quotas[family] = config.product.babelnet_quota_per_stratum
        else:
            quotas[family] = 0
    return quotas


def stratified_product_size_target(config: ExperimentConfig) -> dict[str, Any]:
    family_quotas = family_quota_map(config)
    constraint_cell_count = len(config.matrix.a_count) * len(config.matrix.b_count)
    task_count = len(config.tasks)
    information_level_count = len(config.matrix.information)
    family_quota_sum = sum(family_quotas.values())
    target_base_samples = task_count * constraint_cell_count * family_quota_sum
    return {
        "scale_unit": "task_constraint_cell_with_family_quota_sum",
        "task_count": task_count,
        "constraint_cell_count": constraint_cell_count,
        "family_quotas": family_quotas,
        "family_quota_sum_per_task_cell": family_quota_sum,
        "dataset_family_count_is_not_multiplier": True,
        "target_base_samples": target_base_samples,
        "information_level_count": information_level_count,
        "target_rendered_requests": target_base_samples * information_level_count,
    }
