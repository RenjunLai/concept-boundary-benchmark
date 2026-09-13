from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import lgamma, log
from pathlib import Path
from statistics import mean
from typing import Any

from .config import load_config
from .io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from .recipe_universe_audit import _core_capacity, _quantiles
from .resources import load_inventories
from .taxonomy_bitsets import BitsetHelper


LOG10 = log(10)


def _log10_comb(n: int, k: int) -> float:
    if n < k or k < 0:
        return float("-inf")
    if k == 0:
        return 0.0
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / LOG10


def _descendant_candidates(
    bitsets: BitsetHelper,
    source_id: str,
    seed_bit: int,
    max_depth: int | None,
) -> tuple[tuple[str, int], ...]:
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(child_id, 1) for child_id in bitsets.usable_children(source_id)]
    while queue:
        object_id, depth = queue.pop(0)
        if object_id in seen:
            continue
        seen.add(object_id)
        if max_depth is None or depth <= max_depth:
            if not (bitsets.closure_bits(object_id) & seed_bit):
                rows.append((object_id, depth))
        if max_depth is None or depth < max_depth:
            for child_id in bitsets.usable_children(object_id):
                if child_id not in seen:
                    queue.append((child_id, depth + 1))
    return tuple(rows)


def _b_options_by_depth(
    bitsets: BitsetHelper,
    a1_id: str,
    branch_ids: tuple[str, ...],
    seed_id: str,
    max_depth: int | None,
) -> tuple[dict[str, Any], ...]:
    seed_bit = bitsets.bit(seed_id)
    options: list[dict[str, Any]] = []
    seen: set[str] = set()

    for branch_id in branch_ids:
        for B_id, depth in _descendant_candidates(bitsets, branch_id, seed_bit, max_depth):
            if B_id in seen:
                continue
            seen.add(B_id)
            options.append(
                {
                    "B_id": B_id,
                    "paired_A_source": branch_id,
                    "B_source": "inside_positive_branch",
                    "depth_from_source": depth,
                }
            )

    for B_id, depth in _descendant_candidates(bitsets, a1_id, seed_bit, max_depth):
        if B_id in seen or B_id in branch_ids:
            continue
        seen.add(B_id)
        options.append(
            {
                "B_id": B_id,
                "paired_A_source": a1_id,
                "B_source": "inside_A1",
                "depth_from_source": depth,
            }
        )
    return tuple(options)


def _choose_b(
    bitsets: BitsetHelper,
    options: tuple[dict[str, Any], ...],
    b_count: int,
    policy: str,
) -> tuple[dict[str, Any], ...] | None:
    if b_count == 0:
        return ()
    if len(options) < b_count:
        return None
    if policy == "largest_closure":
        key = lambda item: (-bitsets.closure_size(item["B_id"]), item["depth_from_source"], item["paired_A_source"], item["B_id"])
    elif policy == "smallest_closure":
        key = lambda item: (bitsets.closure_size(item["B_id"]), -item["depth_from_source"], item["paired_A_source"], item["B_id"])
    elif policy == "deepest_then_smallest":
        key = lambda item: (-item["depth_from_source"], bitsets.closure_size(item["B_id"]), item["paired_A_source"], item["B_id"])
    else:
        raise ValueError(policy)
    return tuple(sorted(options, key=key)[:b_count])


def _capacity_row(bitsets: BitsetHelper, a1_id: str, branch_ids: tuple[str, ...], chosen_b: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    capacity = _core_capacity(bitsets, a1_id, branch_ids, chosen_b)
    depths = [int(item.get("depth_from_source", 1)) for item in chosen_b]
    return {
        "B_ids": [item["B_id"] for item in chosen_b],
        "B_sources": [item["B_source"] for item in chosen_b],
        "B_depth": {
            "min": min(depths) if depths else 0,
            "max": max(depths) if depths else 0,
            "mean": mean(depths) if depths else 0.0,
        },
        "excluded_scope_count": capacity["excluded_scope_count"],
        "hits_b_count": capacity["hits_b_count"],
        "final_positive_count": capacity["final_positive_count"],
        "hard_negative_count": capacity["hard_negative_count"],
        "T1_balanced": capacity["task_capacity"]["T1_balanced"],
        "T2_any": capacity["task_capacity"]["T2_any"],
        "T3_balanced": capacity["task_capacity"]["T3_balanced"],
    }


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_recipe_b_depth_audit(config_path: Path, probe_frame: Path, out_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inventories = load_inventories(config.resources)
    bitsets_by_family = {family: BitsetHelper(inventory) for family, inventory in inventories.items()}
    rows = read_jsonl(probe_frame)
    out_dir = ensure_dir(out_dir)

    comparison_rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {}
    by_family_acc: dict[str, dict[str, list[float] | Counter[str]]] = defaultdict(
        lambda: {
            "direct_option_count": [],
            "depth2_option_count": [],
            "all_option_count": [],
            "direct_combo_log10": [],
            "depth2_combo_log10": [],
            "all_combo_log10": [],
            "all_vs_direct_combo_log10_delta": [],
            "direct_hits_b": [],
            "deepest_hits_b": [],
            "largest_same_as_direct": Counter(),
            "deepest_preserves_feasibility": Counter(),
        }
    )
    by_cell_acc: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)

    for row in rows:
        family = row["dataset_family"]
        bitsets = bitsets_by_family[family]
        a1_id = row["A1_id"]
        branch_ids = tuple(row.get("A_branch_ids", ()))
        seed_id = row["seed_id"]
        b_count = int(row["b_count"])

        direct_options = _b_options_by_depth(bitsets, a1_id, branch_ids, seed_id, max_depth=1)
        depth2_options = _b_options_by_depth(bitsets, a1_id, branch_ids, seed_id, max_depth=2)
        all_options = _b_options_by_depth(bitsets, a1_id, branch_ids, seed_id, max_depth=None)

        direct_chosen = tuple(
            {
                "B_id": B_id,
                "paired_A_source": paired,
                "B_source": source,
                "depth_from_source": 1,
            }
            for B_id, paired, source in zip(row.get("B_ids", ()), row.get("paired_A_ids", ()), row.get("B_sources", ()))
        )
        largest_all = _choose_b(bitsets, all_options, b_count, "largest_closure")
        smallest_all = _choose_b(bitsets, all_options, b_count, "smallest_closure")
        deepest_all = _choose_b(bitsets, all_options, b_count, "deepest_then_smallest")

        direct_combo = _log10_comb(len(direct_options), b_count)
        depth2_combo = _log10_comb(len(depth2_options), b_count)
        all_combo = _log10_comb(len(all_options), b_count)

        direct_capacity = _capacity_row(bitsets, a1_id, branch_ids, direct_chosen)
        largest_capacity = _capacity_row(bitsets, a1_id, branch_ids, largest_all) if largest_all is not None else None
        smallest_capacity = _capacity_row(bitsets, a1_id, branch_ids, smallest_all) if smallest_all is not None else None
        deepest_capacity = _capacity_row(bitsets, a1_id, branch_ids, deepest_all) if deepest_all is not None else None

        same_largest = largest_capacity is not None and set(direct_capacity["B_ids"]) == set(largest_capacity["B_ids"])
        deepest_ok = deepest_capacity is not None and all(
            deepest_capacity[key] for key in ("T1_balanced", "T2_any", "T3_balanced")
        )

        acc = by_family_acc[family]
        acc["direct_option_count"].append(len(direct_options))  # type: ignore[union-attr]
        acc["depth2_option_count"].append(len(depth2_options))  # type: ignore[union-attr]
        acc["all_option_count"].append(len(all_options))  # type: ignore[union-attr]
        acc["direct_combo_log10"].append(direct_combo)  # type: ignore[union-attr]
        acc["depth2_combo_log10"].append(depth2_combo)  # type: ignore[union-attr]
        acc["all_combo_log10"].append(all_combo)  # type: ignore[union-attr]
        acc["all_vs_direct_combo_log10_delta"].append(all_combo - direct_combo)  # type: ignore[union-attr]
        acc["direct_hits_b"].append(direct_capacity["hits_b_count"])  # type: ignore[union-attr]
        if deepest_capacity is not None:
            acc["deepest_hits_b"].append(deepest_capacity["hits_b_count"])  # type: ignore[union-attr]
        acc["largest_same_as_direct"].update(["yes" if same_largest else "no"])  # type: ignore[union-attr]
        acc["deepest_preserves_feasibility"].update(["yes" if deepest_ok else "no"])  # type: ignore[union-attr]
        by_cell_acc[(family, int(row["a_count"]), b_count)]["rows"] += 1
        if all_combo - direct_combo >= 3:
            by_cell_acc[(family, int(row["a_count"]), b_count)]["combo_delta_ge_3"] += 1
        if deepest_ok:
            by_cell_acc[(family, int(row["a_count"]), b_count)]["deepest_ok"] += 1

        comparison_rows.append(
            {
                "dataset_family": family,
                "a_count": int(row["a_count"]),
                "b_count": b_count,
                "seed_id": seed_id,
                "A1_id": a1_id,
                "A_branch_ids": list(branch_ids),
                "option_counts": {
                    "direct_depth1": len(direct_options),
                    "depth2": len(depth2_options),
                    "all_depths": len(all_options),
                },
                "combo_log10": {
                    "direct_depth1": direct_combo,
                    "depth2": depth2_combo,
                    "all_depths": all_combo,
                    "all_minus_direct": all_combo - direct_combo,
                },
                "direct_current": direct_capacity,
                "deep_all_largest": largest_capacity,
                "deep_all_smallest": smallest_capacity,
                "deep_all_deepest": deepest_capacity,
            }
        )

    for family, acc in by_family_acc.items():
        by_family[family] = {
            "direct_option_count": _quantiles([int(value) for value in acc["direct_option_count"]]),  # type: ignore[arg-type]
            "depth2_option_count": _quantiles([int(value) for value in acc["depth2_option_count"]]),  # type: ignore[arg-type]
            "all_option_count": _quantiles([int(value) for value in acc["all_option_count"]]),  # type: ignore[arg-type]
            "direct_combo_log10": _quantiles([round(float(value)) for value in acc["direct_combo_log10"]]),  # type: ignore[arg-type]
            "depth2_combo_log10": _quantiles([round(float(value)) for value in acc["depth2_combo_log10"]]),  # type: ignore[arg-type]
            "all_combo_log10": _quantiles([round(float(value)) for value in acc["all_combo_log10"]]),  # type: ignore[arg-type]
            "all_vs_direct_combo_log10_delta": _quantiles(
                [round(float(value)) for value in acc["all_vs_direct_combo_log10_delta"]]  # type: ignore[arg-type]
            ),
            "direct_hits_b": _quantiles([int(value) for value in acc["direct_hits_b"]]),  # type: ignore[arg-type]
            "deepest_hits_b": _quantiles([int(value) for value in acc["deepest_hits_b"]]),  # type: ignore[arg-type]
            "largest_same_as_direct": _counter_dict(acc["largest_same_as_direct"]),  # type: ignore[arg-type]
            "deepest_preserves_feasibility": _counter_dict(acc["deepest_preserves_feasibility"]),  # type: ignore[arg-type]
        }

    by_cell_rows = []
    for key, counter in sorted(by_cell_acc.items()):
        family, a_count, b_count = key
        total = counter["rows"]
        by_cell_rows.append(
            {
                "dataset_family": family,
                "a_count": a_count,
                "b_count": b_count,
                "rows": total,
                "combo_delta_ge_3": counter["combo_delta_ge_3"],
                "combo_delta_ge_3_rate": counter["combo_delta_ge_3"] / total if total else 0.0,
                "deepest_ok": counter["deepest_ok"],
                "deepest_ok_rate": counter["deepest_ok"] / total if total else 0.0,
            }
        )

    write_jsonl(out_dir / "b_depth_core_comparison.jsonl", comparison_rows)
    write_jsonl(out_dir / "b_depth_cell_summary.jsonl", by_cell_rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_name": "recipe_b_depth_audit",
        "config_path": str(config_path),
        "probe_frame": str(probe_frame),
        "probe_rows": len(rows),
        "policy_notes": {
            "direct_depth1": "current standard: B candidates are direct usable child boundaries under selected positive branches or A1",
            "all_depths": "experimental: B candidates may be any descendant boundary that does not remove the seed",
            "combo_log10": "log10 of the number of possible B sets for the fixed A1/A-branch core; materializing these alternatives would multiply core count",
        },
        "families": by_family,
        "outputs": {
            "summary": str(out_dir / "recipe_b_depth_audit_summary.json"),
            "core_comparison": str(out_dir / "b_depth_core_comparison.jsonl"),
            "cell_summary": str(out_dir / "b_depth_cell_summary.jsonl"),
        },
    }
    write_json(out_dir / "recipe_b_depth_audit_summary.json", summary)
    return summary


def write_recipe_b_depth_audit(config_path: Path, probe_frame: Path, out_dir: Path) -> dict[str, Any]:
    return build_recipe_b_depth_audit(config_path, probe_frame, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit B-depth alternatives for current recipe probe cores.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/test.toml"))
    parser.add_argument("--probe-frame", type=Path, default=Path("audits/recipe_seed/lowest_viable_core_probe_frame.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("audits/recipe_b_depth"))
    args = parser.parse_args()
    summary = write_recipe_b_depth_audit(args.config, args.probe_frame, args.out_dir)
    print({"summary": summary["outputs"]["summary"], "probe_rows": summary["probe_rows"]})


if __name__ == "__main__":
    main()
