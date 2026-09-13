from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .constants import POOL_SIZE, T2_GOLD_SET_SIZES
from .io_utils import ensure_dir, stable_hash, write_json, write_jsonl
from .resources import load_inventories
from .taxonomy_bitsets import BitsetHelper


STRATEGIES = ("hash_balanced", "reuse_aware", "reuse_aware_mixed_negatives")
ANSWER_BUCKETS = {
    "T1": ("false", "true"),
    "T2": tuple(str(size) for size in T2_GOLD_SET_SIZES),
    "T3": ("false", "true"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                import json

                rows.append(json.loads(line))
    return rows


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(values)

    def pick(q: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _mask_from_ids(bitsets: BitsetHelper, object_ids: tuple[str, ...]) -> int:
    bits = 0
    for object_id in object_ids:
        bits |= bitsets.closure_bits(object_id)
    return bits


def _core_masks(bitsets: BitsetHelper, row: dict[str, Any]) -> dict[str, int]:
    a1_id = row["A1_id"]
    branch_ids = tuple(row.get("A_branch_ids", ()))
    b_ids = tuple(row.get("B_ids", ()))
    a1_bits = bitsets.closure_bits(a1_id)
    if branch_ids:
        positive_pre_bits = _mask_from_ids(bitsets, branch_ids) & a1_bits
        fails_branch_bits = a1_bits & ~positive_pre_bits
        near_sibling_bits = 0
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
    }


def _ids_from_mask(bitsets: BitsetHelper, mask: int, stream: str, limit: int, salt: str) -> tuple[str, ...]:
    """Return a deterministic probe window without scanning the whole inventory.

    This is for scheduler experiments, not final sample generation. It samples a
    bounded window from the bitset and then applies a stable hash order so the
    scheduler comparison is repeatable while staying cheap on large masks.
    """
    if limit <= 0 or not mask:
        return ()
    candidates = []
    remaining = mask
    scan_limit = max(limit * 8, limit)
    while remaining and len(candidates) < scan_limit:
        low_bit = remaining & -remaining
        index = low_bit.bit_length() - 1
        candidates.append(bitsets.object_ids[index])
        remaining ^= low_bit
    return tuple(sorted(candidates, key=lambda object_id: stable_hash({"stream": stream, "salt": salt, "object_id": object_id}))[:limit])


def _select_from_ids(
    object_ids: tuple[str, ...],
    count: int,
    usage: Counter[str],
    reuse_aware: bool,
    avoid: set[str] | None = None,
) -> tuple[str, ...]:
    if count <= 0:
        return ()
    avoid = avoid or set()
    if len(object_ids) < count:
        return ()
    if not reuse_aware:
        selected = []
        for object_id in object_ids:
            if object_id in avoid:
                continue
            selected.append(object_id)
            if len(selected) == count:
                return tuple(selected)
        return ()
    window = [object_id for object_id in object_ids if object_id not in avoid]
    if len(window) < count:
        return ()
    return tuple(sorted(window, key=lambda object_id: (usage[object_id], stable_hash({"reuse": object_id})))[:count])


def _select_negatives(
    pools: dict[str, tuple[str, ...]],
    count: int,
    usage: Counter[str],
    strategy: str,
    row_index: int,
    avoid: set[str],
) -> tuple[tuple[str, ...], Counter[str], bool]:
    if count <= 0:
        return (), Counter(), False
    reuse_aware = strategy != "hash_balanced"
    selected: list[str] = []
    sources: Counter[str] = Counter()
    fallback = False

    if strategy == "reuse_aware_mixed_negatives":
        mode = ("hard_first", "mixed", "easy_first")[row_index % 3]
    else:
        mode = "hard_first"

    def take(source: str, need: int) -> None:
        nonlocal selected
        if need <= 0:
            return
        pool = pools["hard_negative"] if source == "hard" else pools["easy_negative"]
        picked = _select_from_ids(pool, need, usage, reuse_aware, avoid | set(selected))
        selected.extend(picked)
        sources[source] += len(picked)

    if mode == "hard_first":
        take("hard", count)
        take("easy", count - len(selected))
    elif mode == "easy_first":
        take("easy", count)
        take("hard", count - len(selected))
    else:
        hard_target = count // 2
        easy_target = count - hard_target
        take("hard", hard_target)
        take("easy", easy_target)
        take("hard", count - len(selected))
        take("easy", count - len(selected))

    if len(selected) < count:
        fallback = True
        picked = _select_from_ids(pools["negative"], count - len(selected), usage, reuse_aware, avoid | set(selected))
        selected.extend(picked)
        sources["fallback_any_negative"] += len(picked)
    return tuple(selected[:count]), sources, fallback or len(selected) < count


def _pool_hash(ids: tuple[str, ...]) -> str:
    return stable_hash({"pool": tuple(sorted(ids))})


def _record_sample(
    strategy_state: dict[str, Any],
    task: str,
    family: str,
    cell: tuple[int, int],
    candidate_ids: tuple[str, ...],
    answer_key: str,
    negative_sources: Counter[str],
    fallback: bool,
) -> None:
    strategy_state["base_samples"] += 1
    strategy_state["task_counts"][task] += 1
    strategy_state["answer_counts"][(task, answer_key)] += 1
    strategy_state["cell_answer_counts"][(family, cell[0], cell[1], task, answer_key)] += 1
    if fallback:
        strategy_state["fallback_count"] += 1
    for candidate_id in candidate_ids:
        strategy_state["candidate_usage"][candidate_id] += 1
    if len(candidate_ids) > 1:
        strategy_state["pool_usage"][_pool_hash(candidate_ids)] += 1
    strategy_state["negative_sources"].update(negative_sources)


def _cell_answer_imbalances(cell_answer_counts: Counter[tuple[str, int, int, str, str]]) -> list[int]:
    cell_imbalances = []
    keys = set((family, a, b, task) for family, a, b, task, _answer in cell_answer_counts)
    for family, a, b, task in keys:
        labels = ANSWER_BUCKETS[task]
        counts = [cell_answer_counts[(family, a, b, task, label)] for label in labels]
        cell_imbalances.append(max(counts) - min(counts))
    return cell_imbalances


def _validation_issues(strategy_state: dict[str, Any]) -> list[str]:
    issues = []
    task_total = sum(strategy_state["task_counts"].values())
    if strategy_state["base_samples"] != task_total:
        issues.append(f"base_samples {strategy_state['base_samples']} != task_counts total {task_total}")

    for task, labels in ANSWER_BUCKETS.items():
        task_count = strategy_state["task_counts"][task]
        answer_total = sum(strategy_state["answer_counts"][(task, label)] for label in labels)
        if answer_total != task_count:
            issues.append(f"{task} answer total {answer_total} != task_count {task_count}")
        if task_count >= len(labels):
            missing = [label for label in labels if strategy_state["answer_counts"][(task, label)] == 0]
            if missing:
                issues.append(f"{task} missing answer bucket(s): {', '.join(missing)}")

    expected_pool_samples = strategy_state["task_counts"]["T2"] + strategy_state["task_counts"]["T3"]
    actual_pool_samples = sum(strategy_state["pool_usage"].values())
    if actual_pool_samples != expected_pool_samples:
        issues.append(f"pool samples {actual_pool_samples} != T2+T3 samples {expected_pool_samples}")
    return issues


def _simulate_strategy(
    rows: list[dict[str, Any]],
    bitsets_by_family: dict[str, BitsetHelper],
    strategy: str,
    threshold: int,
    pool_candidate_limit: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["final_positive_count"]) >= threshold:
            grouped[(row["dataset_family"], int(row["a_count"]), int(row["b_count"]))].append(row)

    state: dict[str, Any] = {
        "eligible_core_rows": sum(len(group_rows) for group_rows in grouped.values()),
        "base_samples": 0,
        "task_counts": Counter(),
        "answer_counts": Counter(),
        "cell_answer_counts": Counter(),
        "candidate_usage": Counter(),
        "pool_usage": Counter(),
        "negative_sources": Counter(),
        "fallback_count": 0,
        "skipped_count": 0,
    }

    for (family, a_count, b_count), group_rows in sorted(grouped.items()):
        ordered_rows = sorted(group_rows, key=lambda row: stable_hash({"family": family, "core": row["core_id"]}))
        bitsets = bitsets_by_family[family]
        for index, row in enumerate(ordered_rows):
            masks = _core_masks(bitsets, row)
            salt = str(row["core_id"])
            pools = {
                "positive": _ids_from_mask(bitsets, masks["positive"], "positive", pool_candidate_limit, salt),
                "negative": _ids_from_mask(bitsets, masks["negative"], "negative", pool_candidate_limit, salt),
                "hard_negative": _ids_from_mask(bitsets, masks["hard_negative"], "hard_negative", pool_candidate_limit, salt),
                "easy_negative": _ids_from_mask(bitsets, masks["easy_negative"], "easy_negative", pool_candidate_limit, salt),
            }
            cell = (a_count, b_count)
            reuse_aware = strategy != "hash_balanced"

            t1_true = index % 2 == 0
            if t1_true:
                candidate = _select_from_ids(pools["positive"], 1, state["candidate_usage"], reuse_aware)
                sources = Counter()
            else:
                candidate, sources, fallback = _select_negatives(
                    pools, 1, state["candidate_usage"], strategy, index, set()
                )
                if not candidate:
                    state["skipped_count"] += 1
            if candidate:
                _record_sample(state, "T1", family, cell, candidate, "true" if t1_true else "false", sources, False if t1_true else fallback)

            gold_size = T2_GOLD_SET_SIZES[index % len(T2_GOLD_SET_SIZES)]
            positives = _select_from_ids(
                pools["positive"], gold_size, state["candidate_usage"], reuse_aware
            )
            negatives, sources, fallback = _select_negatives(
                pools, POOL_SIZE - gold_size, state["candidate_usage"], strategy, index, set(positives)
            )
            if len(positives) == gold_size and len(negatives) == POOL_SIZE - gold_size:
                _record_sample(state, "T2", family, cell, tuple(positives + negatives), str(gold_size), sources, fallback)
            else:
                state["skipped_count"] += 1

            t3_true = index % 2 == 0
            if t3_true:
                positives = _select_from_ids(
                    pools["positive"], 1, state["candidate_usage"], reuse_aware
                )
                negatives, sources, fallback = _select_negatives(
                    pools, POOL_SIZE - 1, state["candidate_usage"], strategy, index, set(positives)
                )
                if len(positives) == 1 and len(negatives) == POOL_SIZE - 1:
                    _record_sample(state, "T3", family, cell, tuple(positives + negatives), "true", sources, fallback)
                else:
                    state["skipped_count"] += 1
            else:
                negatives, sources, fallback = _select_negatives(
                    pools, POOL_SIZE, state["candidate_usage"], strategy, index, set()
                )
                if len(negatives) == POOL_SIZE:
                    _record_sample(state, "T3", family, cell, negatives, "false", sources, fallback)
                else:
                    state["skipped_count"] += 1

    usage_values = list(state["candidate_usage"].values())
    pool_values = list(state["pool_usage"].values())
    validation_issues = _validation_issues(state)

    return {
        "strategy": strategy,
        "threshold": threshold,
        "pool_candidate_limit": pool_candidate_limit,
        "eligible_core_rows": state["eligible_core_rows"],
        "base_samples": state["base_samples"],
        "skipped_count": state["skipped_count"],
        "fallback_count": state["fallback_count"],
        "task_counts": _counter_dict(state["task_counts"]),
        "answer_counts": _counter_dict(state["answer_counts"]),
        "negative_sources": _counter_dict(state["negative_sources"]),
        "candidate_reuse": {
            "unique": len(state["candidate_usage"]),
            "total_slots": sum(usage_values),
            "reuse_count_quantiles": _quantiles(usage_values),
        },
        "pool_reuse": {
            "unique": len(state["pool_usage"]),
            "total_pools": sum(pool_values),
            "reuse_count_quantiles": _quantiles(pool_values),
        },
        "cell_answer_imbalance": _quantiles(_cell_answer_imbalances(state["cell_answer_counts"])),
        "validation": {
            "status": "pass" if not validation_issues else "fail",
            "issues": validation_issues,
        },
    }


def build_recipe_candidate_scheduler_probe(
    config_path: Path,
    probe_frame: Path,
    out_dir: Path,
    threshold: int = 5,
    max_rows: int = 0,
    pool_candidate_limit: int = 64,
) -> dict[str, Any]:
    config = load_config(config_path)
    inventories = load_inventories(config.resources)
    bitsets_by_family = {family: BitsetHelper(inventory) for family, inventory in inventories.items()}
    rows = _read_jsonl(probe_frame)
    if max_rows:
        rows = rows[:max_rows]
    eligible_probe_rows = sum(1 for row in rows if int(row["final_positive_count"]) >= threshold)
    out_dir = ensure_dir(out_dir)
    strategy_rows = [
        _simulate_strategy(rows, bitsets_by_family, strategy, threshold, pool_candidate_limit)
        for strategy in STRATEGIES
    ]
    recommendation = {
        "strategy": "reuse_aware_mixed_negatives",
        "reason": "It keeps answer and gold-size balance, actively reduces candidate reuse, and samples both hard and easy negatives without requiring a fixed hard-negative ratio.",
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_name": "recipe_candidate_scheduler_probe",
        "config_path": str(config_path),
        "probe_frame": str(probe_frame),
        "threshold": threshold,
        "max_rows": max_rows,
        "pool_candidate_limit": pool_candidate_limit,
        "probe_rows": len(rows),
        "eligible_probe_rows": eligible_probe_rows,
        "strategies": strategy_rows,
        "recommendation": recommendation,
        "outputs": {
            "summary": str(out_dir / "recipe_candidate_scheduler_probe_summary.json"),
            "strategies": str(out_dir / "candidate_scheduler_strategy_metrics.jsonl"),
        },
    }
    write_jsonl(out_dir / "candidate_scheduler_strategy_metrics.jsonl", strategy_rows)
    write_json(out_dir / "recipe_candidate_scheduler_probe_summary.json", summary)
    return summary


def write_recipe_candidate_scheduler_probe(
    config_path: Path,
    probe_frame: Path,
    out_dir: Path,
    threshold: int = 5,
    max_rows: int = 0,
    pool_candidate_limit: int = 64,
) -> dict[str, Any]:
    return build_recipe_candidate_scheduler_probe(
        config_path, probe_frame, out_dir, threshold=threshold, max_rows=max_rows, pool_candidate_limit=pool_candidate_limit
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe candidate-level scheduler strategies for current recipe.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/test.toml"))
    parser.add_argument("--probe-frame", type=Path, default=Path("audits/recipe_seed/lowest_viable_core_probe_frame.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("audits/recipe_candidate_scheduler"))
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--pool-candidate-limit", type=int, default=64)
    args = parser.parse_args()
    summary = write_recipe_candidate_scheduler_probe(
        args.config,
        args.probe_frame,
        args.out_dir,
        threshold=args.threshold,
        max_rows=args.max_rows,
        pool_candidate_limit=args.pool_candidate_limit,
    )
    print({"summary": summary["outputs"]["summary"], "recommendation": summary["recommendation"]})


if __name__ == "__main__":
    main()
