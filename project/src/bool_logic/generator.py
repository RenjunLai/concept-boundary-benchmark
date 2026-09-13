from __future__ import annotations

from collections.abc import Callable
from collections import Counter
from dataclasses import replace
from typing import Any, Iterator

from .audit import FeasibilityRow
from .config import ExperimentConfig
from .constants import INFORMATION_LEVELS, POOL_SIZE, RENDERING_TEMPLATE_VERSION, T2_GOLD_SET_SIZES
from .io_utils import stable_hash, to_plain
from .oracle import gold_bool, gold_pool
from .product_scale import stratified_product_size_target
from .resources import Inventory
from .schemas import FrozenSample
from .taxonomy_bitsets import BitsetHelper

FRAME_VERSION = "seed_object_first"
ORACLE_VERSION = "a1_or_branches_minus_b"
PRODUCT_ID = "test"
ProgressCallback = Callable[[dict[str, Any]], None]


def _ids_from_mask(
    bitsets: BitsetHelper,
    mask: int,
    limit: int = 64,
    salt: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if limit <= 0 or not mask:
        return ()
    ids: list[str] = []
    remaining = mask
    scan_limit = max(limit * 8, limit)
    while remaining and len(ids) < scan_limit:
        low_bit = remaining & -remaining
        index = low_bit.bit_length() - 1
        ids.append(bitsets.object_ids[index])
        remaining ^= low_bit
    if salt is None:
        return tuple(ids[:limit])
    return tuple(sorted(ids, key=lambda object_id: stable_hash({"salt": salt, "object_id": object_id}))[:limit])


def _mask_from_ids(bitsets: BitsetHelper, object_ids: tuple[str, ...]) -> int:
    bits = 0
    for object_id in object_ids:
        bits |= bitsets.closure_bits(object_id)
    return bits


def _children_covering_seed(bitsets: BitsetHelper, a1_id: str, seed_bit: int) -> tuple[str, ...]:
    children = [
        child_id
        for child_id in bitsets.usable_children(a1_id)
        if bitsets.closure_bits(child_id) & seed_bit
    ]
    return tuple(sorted(children, key=lambda item: (bitsets.closure_size(item), item)))


def _children_not_covering_seed(bitsets: BitsetHelper, parent_id: str, seed_bit: int) -> tuple[str, ...]:
    children = [
        child_id
        for child_id in bitsets.usable_children(parent_id)
        if not (bitsets.closure_bits(child_id) & seed_bit)
    ]
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
    for child_id in covering[1:]:
        if child_id not in branches:
            branches.append(child_id)
            if len(branches) == branch_count:
                return tuple(sorted(branches))
    return None


def _b_options(
    bitsets: BitsetHelper,
    a1_id: str,
    branch_ids: tuple[str, ...],
    seed_id: str,
) -> tuple[dict[str, str], ...]:
    seed_bit = bitsets.bit(seed_id)
    options: list[dict[str, str]] = []
    seen: set[str] = set()
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


def _core_masks(bitsets: BitsetHelper, a1_id: str, branch_ids: tuple[str, ...], b_ids: tuple[str, ...]) -> dict[str, int]:
    a1_bits = bitsets.closure_bits(a1_id)
    if branch_ids:
        positive_pre_bits = _mask_from_ids(bitsets, branch_ids) & a1_bits
        fails_branch_bits = a1_bits & ~positive_pre_bits
        near_sibling_bits = fails_branch_bits
    else:
        positive_pre_bits = a1_bits
        fails_branch_bits = 0
        near_sibling_bits = bitsets.sibling_bits(a1_id) & ~positive_pre_bits
    b_bits = _mask_from_ids(bitsets, b_ids)
    hits_b_bits = positive_pre_bits & b_bits
    positive_bits = positive_pre_bits & ~b_bits
    outside_bits = bitsets.all_mask & ~a1_bits & ~bitsets.bit(a1_id)
    hard_bits = hits_b_bits | fails_branch_bits | near_sibling_bits
    negative_bits = hard_bits | outside_bits
    return {
        "positive": positive_bits,
        "negative": negative_bits,
        "hard_negative": hard_bits,
        "easy_negative": outside_bits,
        "hits_B": hits_b_bits,
        "fails_positive_branch": fails_branch_bits,
        "near_sibling": near_sibling_bits,
        "outside_A1": outside_bits,
    }


def _find_seed_core(
    inventory: Inventory,
    bitsets: BitsetHelper,
    seed_id: str,
    a_count: int,
    b_count: int,
    min_final_positive_count: int,
) -> dict[str, Any] | None:
    seed_bit = bitsets.bit(seed_id)
    branch_count = a_count - 1
    for a1_id, distance in bitsets.ancestor_walk(seed_id):
        if not (bitsets.closure_bits(a1_id) & seed_bit):
            continue
        branch_ids = _select_branches(bitsets, a1_id, seed_id, branch_count)
        if branch_ids is None:
            continue
        chosen_b = _choose_b(bitsets, _b_options(bitsets, a1_id, branch_ids, seed_id), b_count)
        if chosen_b is None:
            continue
        b_ids = tuple(item["B_id"] for item in chosen_b)
        masks = _core_masks(bitsets, a1_id, branch_ids, b_ids)
        if not (masks["positive"] & seed_bit):
            continue
        if masks["positive"].bit_count() < min_final_positive_count:
            continue
        if not masks["negative"]:
            continue
        return {
            "seed_id": seed_id,
            "A1_id": a1_id,
            "A_branch_ids": branch_ids,
            "B_ids": b_ids,
            "paired_A_ids": tuple(item["paired_A_source"] for item in chosen_b),
            "B_sources": tuple(item["B_source"] for item in chosen_b),
            "lowest_viable_distance": distance,
            "masks": masks,
        }
    return None


def _stable_select(
    object_ids: tuple[str, ...],
    count: int,
    salt: dict[str, Any],
    usage: Counter[str] | None = None,
    avoid: set[str] | None = None,
) -> tuple[str, ...]:
    avoid = avoid or set()
    candidates = [object_id for object_id in object_ids if object_id not in avoid]
    if len(candidates) < count:
        return ()
    usage = usage or Counter()
    return tuple(sorted(candidates, key=lambda object_id: (usage[object_id], stable_hash({"salt": salt, "object_id": object_id})))[:count])


def _negative_source_for_candidate(bitsets: BitsetHelper, masks: dict[str, int], candidate_id: str) -> str:
    candidate_bit = bitsets.bit(candidate_id)
    if masks["hits_B"] & candidate_bit:
        return "hits_B"
    if masks["fails_positive_branch"] & candidate_bit:
        return "fails_positive_branch"
    if masks["near_sibling"] & candidate_bit:
        return "near_sibling"
    return "outside_A1"


def _negative_source_counts(bitsets: BitsetHelper, masks: dict[str, int], candidate_ids: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate_id in candidate_ids:
        if masks["positive"] & bitsets.bit(candidate_id):
            continue
        counts[_negative_source_for_candidate(bitsets, masks, candidate_id)] += 1
    return dict(counts)


def _sample_payload(
    dataset_family: str,
    task: str,
    a_count: int,
    b_count: int,
    core: dict[str, Any],
    candidate_ids: tuple[str, ...],
    gold_answer: bool | tuple[int, ...],
) -> dict[str, Any]:
    return {
        "dataset_family": dataset_family,
        "task": task,
        "constraint_cell": [a_count, b_count],
        "seed_id": core["seed_id"],
        "A1_id": core["A1_id"],
        "A_branch_ids": core["A_branch_ids"],
        "B_ids": core["B_ids"],
        "candidate_ids": candidate_ids,
        "gold_answer": gold_answer,
        "oracle_version": ORACLE_VERSION,
        "frame_version": FRAME_VERSION,
    }


def _build_base_samples(
    inventory: Inventory,
    bitsets: BitsetHelper,
    dataset_family: str,
    task: str,
    a_count: int,
    b_count: int,
    core: dict[str, Any],
    variant_index: int,
    usage: Counter[str],
    product_id: str = PRODUCT_ID,
    candidate_window_limit: int = 64,
    t2_gold_set_size: int | None = None,
    t3_force_answer: bool | None = None,
) -> list[FrozenSample] | None:
    masks = core["masks"]
    window_salt = {
        "dataset_family": dataset_family,
        "task": task,
        "a_count": a_count,
        "b_count": b_count,
        "seed_id": core["seed_id"],
        "A1_id": core["A1_id"],
        "A_branch_ids": core["A_branch_ids"],
        "B_ids": core["B_ids"],
        "variant_index": variant_index,
        "product_id": product_id,
    }
    positive_pool = _ids_from_mask(bitsets, masks["positive"], candidate_window_limit, {**window_salt, "stream": "positive"})
    negative_pool = _ids_from_mask(bitsets, masks["negative"], candidate_window_limit, {**window_salt, "stream": "negative"})
    hard_pool = _ids_from_mask(bitsets, masks["hard_negative"], candidate_window_limit, {**window_salt, "stream": "hard_negative"})
    easy_pool = _ids_from_mask(bitsets, masks["easy_negative"], candidate_window_limit, {**window_salt, "stream": "easy_negative"})

    candidate_id: str | None = None
    pool_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = {
        "lowest_viable_distance": core["lowest_viable_distance"],
        "final_positive_count": masks["positive"].bit_count(),
        "negative_count": masks["negative"].bit_count(),
        "hard_negative_count": masks["hard_negative"].bit_count(),
    }

    if task == "T1":
        make_true = variant_index % 2 == 0
        source_pool = positive_pool if make_true else (hard_pool or negative_pool)
        candidate = _stable_select(source_pool, 1, {"task": task, "variant": variant_index, "truth": make_true}, usage)
        if not candidate:
            return None
        candidate_id = candidate[0]
        gold_answer: bool | tuple[int, ...] = gold_bool(inventory, candidate_id, core["A1_id"], core["A_branch_ids"], core["B_ids"])
        if not gold_answer:
            metadata["negative_source"] = _negative_source_for_candidate(bitsets, masks, candidate_id)
    elif task == "T2":
        gold_size = t2_gold_set_size if t2_gold_set_size is not None else T2_GOLD_SET_SIZES[variant_index % len(T2_GOLD_SET_SIZES)]
        positives = _stable_select(positive_pool, gold_size, {"task": task, "variant": variant_index, "kind": "positive"}, usage)
        negative_source_pool = hard_pool if len(hard_pool) >= POOL_SIZE - gold_size else negative_pool
        negatives = _stable_select(negative_source_pool, POOL_SIZE - gold_size, {"task": task, "variant": variant_index, "kind": "negative"}, usage, set(positives))
        if len(positives) != gold_size or len(negatives) != POOL_SIZE - gold_size:
            return None
        pool_ids = tuple(sorted(positives + negatives, key=lambda object_id: stable_hash({"pool": task, "variant": variant_index, "object_id": object_id})))
        gold_answer = gold_pool(inventory, pool_ids, core["A1_id"], core["A_branch_ids"], core["B_ids"])
        metadata["gold_set_size"] = len(gold_answer)
        metadata["negative_source_counts"] = _negative_source_counts(bitsets, masks, pool_ids)
    else:
        make_true = t3_force_answer if t3_force_answer is not None else variant_index % 2 == 0
        if make_true:
            positives = _stable_select(positive_pool, 1, {"task": task, "variant": variant_index, "kind": "positive"}, usage)
            negative_source_pool = hard_pool if len(hard_pool) >= POOL_SIZE - 1 else negative_pool
            negatives = _stable_select(negative_source_pool, POOL_SIZE - 1, {"task": task, "variant": variant_index, "kind": "negative"}, usage, set(positives))
            if len(positives) != 1 or len(negatives) != POOL_SIZE - 1:
                return None
            pool_ids = tuple(sorted(positives + negatives, key=lambda object_id: stable_hash({"pool": task, "variant": variant_index, "object_id": object_id})))
        else:
            negative_source_pool = hard_pool if len(hard_pool) >= POOL_SIZE else negative_pool
            pool_ids = _stable_select(negative_source_pool, POOL_SIZE, {"task": task, "variant": variant_index, "kind": "negative"}, usage)
            if len(pool_ids) != POOL_SIZE:
                return None
        gold_set = gold_pool(inventory, pool_ids, core["A1_id"], core["A_branch_ids"], core["B_ids"])
        gold_answer = bool(gold_set)
        metadata["positive_count"] = len(gold_set)
        metadata["negative_source_counts"] = _negative_source_counts(bitsets, masks, pool_ids)

    candidate_ids = (candidate_id,) if task == "T1" and candidate_id else pool_ids
    base_payload = _sample_payload(dataset_family, task, a_count, b_count, core, candidate_ids, gold_answer)
    base_hash = stable_hash(base_payload, length=24)
    base_sample_id = f"base_{base_hash}"
    samples: list[FrozenSample] = []
    for information in INFORMATION_LEVELS:
        payload = {**base_payload, "information": information, "renderer_version": RENDERING_TEMPLATE_VERSION}
        sample_hash = stable_hash(payload, length=32)
        request_hash = stable_hash({**payload, "request_layer": "rendered_request"}, length=32)
        samples.append(
            FrozenSample(
                base_sample_id=base_sample_id,
                sample_id=f"sample_{sample_hash[:24]}",
                product_id=product_id,
                task=task,
                dataset_family=dataset_family,
                constraint_cell=(a_count, b_count),
                a_count=a_count,
                b_count=b_count,
                condition_count=a_count + b_count,
                information=information,
                seed_id=core["seed_id"],
                A1_id=core["A1_id"],
                A_branch_ids=core["A_branch_ids"],
                B_ids=core["B_ids"],
                paired_A_ids=core["paired_A_ids"],
                B_sources=core["B_sources"],
                candidate_ids=candidate_ids,
                gold_answer=gold_answer,
                oracle_version=ORACLE_VERSION,
                renderer_version=RENDERING_TEMPLATE_VERSION,
                frame_version=FRAME_VERSION,
                stratum_id=f"{dataset_family}:{task}:{a_count}:{b_count}",
                inclusion_probability=1.0,
                sampling_weight=1.0,
                sample_hash=sample_hash,
                request_hash=request_hash,
                reuse_metadata={"variant_index": variant_index},
                metadata=metadata,
            )
        )
    return samples


def _base_reuse_keys(sample: FrozenSample) -> dict[str, tuple[str, ...]]:
    return {
        "seed": (sample.seed_id,),
        "A1": (sample.A1_id,),
        "constraint": (stable_hash({"A1": sample.A1_id, "branches": sample.A_branch_ids, "B": sample.B_ids}),),
        "candidate": sample.candidate_ids,
        "pool": (stable_hash(sample.candidate_ids),) if sample.task in ("T2", "T3") else (),
    }


def _would_exceed_reuse(keys: dict[str, tuple[str, ...]], counters: dict[str, Counter], config: ExperimentConfig) -> bool:
    limits = {
        "seed": config.budgets.max_seed_reuse,
        "A1": config.budgets.max_a1_reuse,
        "constraint": config.budgets.max_constraint_reuse,
        "candidate": config.budgets.max_candidate_reuse,
        "pool": config.budgets.max_pool_reuse,
    }
    for name, values in keys.items():
        limit = limits[name]
        if limit <= 0:
            continue
        if any(counters[name][value] >= limit for value in values):
            return True
    return False


def _record_reuse(keys: dict[str, tuple[str, ...]], counters: dict[str, Counter]) -> None:
    for name, values in keys.items():
        for value in values:
            counters[name][value] += 1


def _product_quota(config: ExperimentConfig, family: str) -> int:
    if family == "wordnet":
        return config.product.wordnet_quota_per_stratum
    if family == "babelnet":
        return config.product.babelnet_quota_per_stratum
    return 0


def _ordered_cores(
    config: ExperimentConfig,
    dataset_family: str,
    task: str,
    a_count: int,
    b_count: int,
    seed_ids: tuple[str, ...],
    cores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seed_order = {seed_id: index for index, seed_id in enumerate(seed_ids)}
    return sorted(
        cores,
        key=lambda core: stable_hash(
            {
                "product_id": config.product.product_id,
                "tier": config.product.tier,
                "scheduler": config.product.candidate_scheduler,
                "seed": config.budgets.seed,
                "dataset_family": dataset_family,
                "task": task,
                "a_count": a_count,
                "b_count": b_count,
                "seed_index": seed_order[core["seed_id"]],
                "seed_id": core["seed_id"],
            },
            length=32,
        ),
    )


def _ordered_seed_ids(
    config: ExperimentConfig,
    dataset_family: str,
    task: str,
    a_count: int,
    b_count: int,
    seed_ids: tuple[str, ...],
) -> tuple[str, ...]:
    seed_order = {seed_id: index for index, seed_id in enumerate(seed_ids)}
    return tuple(
        sorted(
            seed_ids,
            key=lambda seed_id: stable_hash(
                {
                    "product_id": config.product.product_id,
                    "tier": config.product.tier,
                    "scheduler": config.product.candidate_scheduler,
                    "seed": config.budgets.seed,
                    "dataset_family": dataset_family,
                    "task": task,
                    "a_count": a_count,
                    "b_count": b_count,
                    "seed_index": seed_order[seed_id],
                    "seed_id": seed_id,
                },
                length=32,
            ),
        )
    )


def _apply_sampling_weight(samples: list[FrozenSample], inclusion_probability: float, sampling_weight: float) -> list[FrozenSample]:
    return [
        replace(sample, inclusion_probability=inclusion_probability, sampling_weight=sampling_weight)
        for sample in samples
    ]


def _target_for_task(task: str, accepted_count: int) -> int | bool | None:
    if task == "T2":
        return T2_GOLD_SET_SIZES[accepted_count % len(T2_GOLD_SET_SIZES)]
    if task == "T3":
        return accepted_count % 2 == 0
    return None


def _generation_manifest_base(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "product_id": config.product.product_id,
        "product_tier": config.product.tier,
        "frame_version": FRAME_VERSION,
        "oracle_version": ORACLE_VERSION,
        "renderer_version": RENDERING_TEMPLATE_VERSION,
        "matrix": to_plain(config.matrix),
        "resources": to_plain(config.resources),
        "budgets": to_plain(config.budgets),
        "product": to_plain(config.product),
        "paired_information_levels": list(INFORMATION_LEVELS),
        "analysis_unit_rule": "information_effects_use_base_sample_id_as_the_paired_or_cluster_unit",
        "candidate_scheduler": {
            "name": config.product.candidate_scheduler,
            "candidate_window_limit": config.product.candidate_window_limit,
            "answer_balancing": "T1/T3 alternate boolean targets; T2 cycles gold set sizes 0/1/2.",
            "candidate_reuse_order": "stable deterministic selection with reuse-aware ordering inside candidate pools.",
            "B_depth_policy": "nearest feasible B only",
        },
    }


def _generate_stratified_product_samples(
    config: ExperimentConfig,
    inventories: dict[str, Inventory],
    feasibility_rows: list[FeasibilityRow],
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[FrozenSample], dict[str, Any]]:
    samples: list[FrozenSample] = []
    base_count = 0
    reuse_counters: dict[str, Counter] = {
        "seed": Counter(),
        "A1": Counter(),
        "constraint": Counter(),
        "candidate": Counter(),
        "pool": Counter(),
    }
    stratum_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    bitsets_by_family = {family: BitsetHelper(inventory) for family, inventory in inventories.items()}
    seed_ids_by_family = {
        family: tuple(sorted(inventory.objects))[: config.resources.max_seed_per_family or None]
        for family, inventory in inventories.items()
    }
    population_lookup = {
        (row.dataset_family, row.task, row.a_count, row.b_count): row.feasible_seed_cell_count
        for row in feasibility_rows
    }
    usage = reuse_counters["candidate"]
    for dataset_family in config.resources.dataset_families:
        if dataset_family not in inventories:
            continue
        quota = _product_quota(config, dataset_family)
        if quota <= 0:
            continue
        inventory = inventories[dataset_family]
        bitsets = bitsets_by_family[dataset_family]
        seed_ids = seed_ids_by_family[dataset_family]
        for task in config.tasks:
            for a_count in config.matrix.a_count:
                for b_count in config.matrix.b_count:
                    population = population_lookup.get((dataset_family, task, a_count, b_count))
                    if population is None:
                        population = sum(
                            1
                            for seed_id in seed_ids
                            if _find_seed_core(
                                inventory,
                                bitsets,
                                seed_id,
                                a_count,
                                b_count,
                                config.resources.min_final_positive_count,
                            )
                        )
                    ordered_seed_ids = _ordered_seed_ids(config, dataset_family, task, a_count, b_count, seed_ids)
                    target = min(quota, population)
                    accepted = 0
                    scanned = 0
                    for seed_id in ordered_seed_ids:
                        if accepted >= target:
                            break
                        scanned += 1
                        core = _find_seed_core(
                            inventory,
                            bitsets,
                            seed_id,
                            a_count,
                            b_count,
                            config.resources.min_final_positive_count,
                        )
                        if not core:
                            continue
                        requested_target = _target_for_task(task, accepted)
                        built = _build_base_samples(
                            inventory,
                            bitsets,
                            dataset_family,
                            task,
                            a_count,
                            b_count,
                            core,
                            accepted,
                            usage,
                            product_id=config.product.product_id,
                            candidate_window_limit=config.product.candidate_window_limit,
                            t2_gold_set_size=requested_target if task == "T2" else None,
                            t3_force_answer=requested_target if task == "T3" else None,
                        )
                        if not built:
                            continue
                        keys = _base_reuse_keys(built[0])
                        if _would_exceed_reuse(keys, reuse_counters, config):
                            continue
                        inclusion = 1.0 if population <= target or target == 0 else target / population
                        weight = 1.0 if target == 0 else population / target
                        samples.extend(_apply_sampling_weight(built, inclusion, weight))
                        _record_reuse(keys, reuse_counters)
                        accepted += 1
                        base_count += 1
                    if accepted < target:
                        skipped.append(
                            {
                                "dataset_family": dataset_family,
                                "task": task,
                                "a_count": a_count,
                                "b_count": b_count,
                                "reason": "quota_not_filled_after_candidate_realization",
                                "eligible_population": population,
                                "quota": quota,
                                "target": target,
                                "accepted": accepted,
                            }
                        )
                    stratum_rows.append(
                        {
                            "stratum_id": f"{dataset_family}:{task}:{a_count}:{b_count}",
                            "dataset_family": dataset_family,
                            "task": task,
                            "a_count": a_count,
                            "b_count": b_count,
                            "post_gate_population": population,
                            "quota": quota,
                            "selected_base_samples": accepted,
                            "inclusion_probability": 0.0 if population == 0 else min(1.0, accepted / population),
                            "sampling_weight": None if accepted == 0 else population / accepted,
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "dataset_family": dataset_family,
                                "task": task,
                                "a_count": a_count,
                                "b_count": b_count,
                                "post_gate_population": population,
                                "quota": quota,
                                "target": target,
                                "accepted": accepted,
                                "scanned_seed_count": scanned,
                                "base_sample_count": base_count,
                                "sample_count": len(samples),
                            }
                        )
    manifest = {
        **_generation_manifest_base(config),
        "sampling_policy": {
            "type": "post_gate_stratified_without_replacement",
            "gate": f"final_positive_count >= {config.resources.min_final_positive_count}",
            "wordnet_quota_per_stratum": config.product.wordnet_quota_per_stratum,
            "babelnet_quota_per_stratum": config.product.babelnet_quota_per_stratum,
            "weight_population": "post_gate_eligible_population",
            "small_positive_pool_policy": "excluded_from_test_and_main; retained by full",
        },
        "size_target": stratified_product_size_target(config),
        "base_sample_count": base_count,
        "sample_count": len(samples),
        "strata": stratum_rows,
        "reuse_cap_unit": "base_sample",
        "reuse_counts": {name: dict(counter) for name, counter in reuse_counters.items()},
        "skipped": skipped,
    }
    validate_samples(samples, inventories)
    return samples, manifest


def iter_extended_samples(config: ExperimentConfig, inventories: dict[str, Inventory]) -> Iterator[list[FrozenSample]]:
    reuse_counters: dict[str, Counter] = {
        "seed": Counter(),
        "A1": Counter(),
        "constraint": Counter(),
        "candidate": Counter(),
        "pool": Counter(),
    }
    usage = reuse_counters["candidate"]
    bitsets_by_family = {family: BitsetHelper(inventory) for family, inventory in inventories.items()}
    seed_ids_by_family = {
        family: tuple(sorted(inventory.objects))[: config.resources.max_seed_per_family or None]
        for family, inventory in inventories.items()
    }
    variant_counts: Counter[tuple[str, str, int, int]] = Counter()
    for dataset_family in config.resources.dataset_families:
        if dataset_family not in inventories:
            continue
        inventory = inventories[dataset_family]
        bitsets = bitsets_by_family[dataset_family]
        for task in config.tasks:
            for a_count in config.matrix.a_count:
                for b_count in config.matrix.b_count:
                    cell_key = (dataset_family, task, a_count, b_count)
                    for seed_id in seed_ids_by_family[dataset_family]:
                        core = _find_seed_core(
                            inventory,
                            bitsets,
                            seed_id,
                            a_count,
                            b_count,
                            config.resources.min_final_positive_count,
                        )
                        if not core:
                            continue
                        variant_index = variant_counts[cell_key]
                        requested_target = _target_for_task(task, variant_index)
                        built = _build_base_samples(
                            inventory,
                            bitsets,
                            dataset_family,
                            task,
                            a_count,
                            b_count,
                            core,
                            variant_index,
                            usage,
                            product_id=config.product.product_id,
                            candidate_window_limit=config.product.candidate_window_limit,
                            t2_gold_set_size=requested_target if task == "T2" else None,
                            t3_force_answer=requested_target if task == "T3" else None,
                        )
                        if not built:
                            continue
                        keys = _base_reuse_keys(built[0])
                        _record_reuse(keys, reuse_counters)
                        variant_counts[cell_key] += 1
                        yield built


def extended_manifest_template(config: ExperimentConfig) -> dict[str, Any]:
    return {
        **_generation_manifest_base(config),
        "sampling_policy": {
            "type": "full_seed_object_first_frame_realization",
            "gate": f"final_positive_count >= {config.resources.min_final_positive_count}",
            "inclusion_probability": 1.0,
            "sampling_weight": 1.0,
            "small_positive_pool_policy": "retained",
        },
        "reuse_cap_unit": "base_sample",
        "streaming_policy": {
            "samples_are_written_to_shards": True,
            "shard_size": config.product.shard_size,
        },
    }


def generate_frozen_samples(
    config: ExperimentConfig,
    inventories: dict[str, Inventory],
    feasibility_rows: list[FeasibilityRow],
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[FrozenSample], dict[str, Any]]:
    if config.product.tier in {"test", "main"}:
        return _generate_stratified_product_samples(config, inventories, feasibility_rows, progress_callback)
    if config.product.tier == "full":
        raise ValueError("full must be generated through the streaming freeze path")
    raise ValueError(f"unsupported product tier: {config.product.tier}")


def validate_samples(samples: list[FrozenSample], inventories: dict[str, Inventory] | None = None) -> None:
    sample_hashes = [sample.sample_hash for sample in samples]
    if len(sample_hashes) != len(set(sample_hashes)):
        raise ValueError("duplicate sample_hash")
    request_hashes = [sample.request_hash for sample in samples]
    if len(request_hashes) != len(set(request_hashes)):
        raise ValueError("duplicate request_hash")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id")
    by_base: dict[str, list[FrozenSample]] = {}
    for sample in samples:
        by_base.setdefault(sample.base_sample_id, []).append(sample)
        if sample.a_count not in (1, 2, 3, 4, 5):
            raise ValueError(f"invalid a_count: {sample.a_count}")
        if sample.b_count not in (0, 1, 2, 3, 4, 5):
            raise ValueError(f"invalid b_count: {sample.b_count}")
        if sample.condition_count != sample.a_count + sample.b_count:
            raise ValueError(f"invalid condition_count for {sample.sample_id}")
        if len(sample.A_branch_ids) != max(0, sample.a_count - 1):
            raise ValueError(f"a_count does not match A_branch_ids for {sample.sample_id}")
        if len(sample.B_ids) != sample.b_count:
            raise ValueError(f"b_count does not match B_ids for {sample.sample_id}")
        if len(sample.paired_A_ids) != sample.b_count or len(sample.B_sources) != sample.b_count:
            raise ValueError(f"B provenance does not match b_count for {sample.sample_id}")
        if sample.task == "T1":
            if len(sample.candidate_ids) != 1:
                raise ValueError(f"invalid T1 candidate fields for {sample.sample_id}")
        elif sample.task in ("T2", "T3"):
            if len(sample.candidate_ids) != POOL_SIZE:
                raise ValueError(f"invalid {sample.task} pool size for {sample.sample_id}")
        if inventories:
            inventory = inventories[sample.dataset_family]
            if sample.task == "T1":
                expected_gold = gold_bool(inventory, sample.candidate_ids[0], sample.A1_id, sample.A_branch_ids, sample.B_ids)
            elif sample.task == "T2":
                expected_gold = gold_pool(inventory, sample.candidate_ids, sample.A1_id, sample.A_branch_ids, sample.B_ids)
            else:
                expected_gold = bool(gold_pool(inventory, sample.candidate_ids, sample.A1_id, sample.A_branch_ids, sample.B_ids))
            if sample.gold_answer != expected_gold:
                raise ValueError(f"gold_answer oracle mismatch for {sample.sample_id}: {sample.gold_answer} != {expected_gold}")
    for base_id, group in by_base.items():
        infos = sorted(sample.information for sample in group)
        if infos != ["I0", "I1", "I2"]:
            raise ValueError(f"base sample {base_id} is not I0/I1/I2 paired: {infos}")
        payloads = {
            (
                sample.dataset_family,
                sample.task,
                sample.constraint_cell,
                sample.seed_id,
                sample.A1_id,
                sample.A_branch_ids,
                sample.B_ids,
                sample.candidate_ids,
                str(sample.gold_answer),
            )
            for sample in group
        }
        if len(payloads) != 1:
            raise ValueError(f"base sample {base_id} has inconsistent paired payloads")
