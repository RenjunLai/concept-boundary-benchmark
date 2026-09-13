from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .audit import FeasibilityRow, audit_experiment
from .config import ExperimentConfig, provider_config_for_run
from .constants import INFORMATION_LEVELS, RENDERING_TEMPLATE_VERSION
from .evaluation_reporting import generate_evaluation_report
from .generator import extended_manifest_template, generate_frozen_samples, iter_extended_samples
from .io_utils import ensure_dir, read_json, read_jsonl, stable_hash, to_plain, write_json, write_jsonl
from .parsing import build_predictions_and_scores
from .product_scale import stratified_product_size_target
from .providers import get_provider
from .rendering import render_samples
from .resources import Inventory, load_inventories
from .run_control import read_response_run_state, run_provider_requests


def resource_snapshot(inventories: dict[str, Inventory]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        family: {
            object_id: {
                "object_id": obj.object_id,
                "surface": obj.surface,
                "gloss": obj.gloss,
                "synonyms": obj.synonyms,
                "children": obj.children,
                "parents": obj.parents,
                "usable_flags": obj.usable_flags,
                "metadata": obj.metadata,
            }
            for object_id, obj in inventory.objects.items()
        }
        for family, inventory in inventories.items()
    }


def write_config_snapshot(config: ExperimentConfig, out_dir: Path) -> None:
    write_json(out_dir / "config_snapshot.json", to_plain(config))


def dataset_config_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    snapshot = to_plain(config)
    snapshot.pop("provider", None)
    return snapshot


def write_dataset_config_snapshot(config: ExperimentConfig, out_dir: Path) -> None:
    write_json(out_dir / "dataset_config_snapshot.json", dataset_config_snapshot(config))


def run_audit(config: ExperimentConfig, out_dir: Path) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    audit, feasibility, inventories = audit_experiment(config)
    write_json(out_dir / "resource_audit.json", audit)
    write_jsonl(out_dir / "feasibility.jsonl", feasibility)
    write_json(out_dir / "resource_snapshot.json", resource_snapshot(inventories))
    write_dataset_config_snapshot(config, out_dir)
    write_json(out_dir / "stage_status.json", {"stage": "audit", "complete": False, "reason": "audit/freeze/render directory is not a completed run"})
    return {
        "resource_audit": str(out_dir / "resource_audit.json"),
        "feasibility": str(out_dir / "feasibility.jsonl"),
        "resource_snapshot": str(out_dir / "resource_snapshot.json"),
        "base_feasibility_rows": len(feasibility),
        "feasibility_unit": "dataset_family x task x constraint_cell",
        "stratified_product_size_target": (
            stratified_product_size_target(config) if config.product.tier in {"test", "main"} else None
        ),
    }


def _feasibility_rows(path: Path) -> list[FeasibilityRow]:
    rows = []
    for row in read_jsonl(path):
        rows.append(
            FeasibilityRow(
                dataset_family=row["dataset_family"],
                task=row["task"],
                constraint_cell=tuple(row["constraint_cell"]),
                a_count=row["a_count"],
                b_count=row["b_count"],
                information_policy=row.get("information_policy", "paired_I0_I1_I2_per_base_sample_id"),
                closure_size_distribution=dict(row["closure_size_distribution"]),
                usable_seed_count=row["usable_seed_count"],
                feasible_seed_cell_count=row["feasible_seed_cell_count"],
                usable_candidate_count=row["usable_candidate_count"],
                direct_usable_child_count=row["direct_usable_child_count"],
                B_sources=tuple(row["B_sources"]),
                available_primary_negative_source=tuple(row["available_primary_negative_source"]),
                hard_negative_available=row["hard_negative_available"],
                easy_negative_available=row["easy_negative_available"],
                t1_capacity=row["t1_capacity"],
                t2_pool_space_estimate=row["t2_pool_space_estimate"],
                t3_pool_space_estimate=row["t3_pool_space_estimate"],
                status=row["status"],
                reason=row["reason"],
                surface_coverage=float(row.get("surface_coverage", 1.0)),
                gloss_coverage=float(row.get("gloss_coverage", 1.0)),
                final_positive_min=int(row.get("final_positive_min", 0)),
                final_positive_max=int(row.get("final_positive_max", 0)),
                negative_pool_min=int(row.get("negative_pool_min", 0)),
                negative_pool_max=int(row.get("negative_pool_max", 0)),
                recommended_pool_size=int(row.get("recommended_pool_size", 4)),
                budget_seed_limit=int(row.get("budget_seed_limit", 0)),
                budget_max_base_samples=int(row.get("budget_max_base_samples", 0)),
            )
        )
    return rows


def _update_full_stream_validation(
    validation: dict[str, Any],
    sample: Any,
    inventories: dict[str, Inventory],
) -> None:
    sample_hash = sample.sample_hash
    request_hash = sample.request_hash
    sample_id = sample.sample_id
    base_id = sample.base_sample_id
    if sample_hash in validation["sample_hashes"]:
        validation["duplicate_sample_hashes"] += 1
    validation["sample_hashes"].add(sample_hash)
    if request_hash in validation["request_hashes"]:
        validation["duplicate_request_hashes"] += 1
    validation["request_hashes"].add(request_hash)
    if sample_id in validation["sample_ids"]:
        validation["duplicate_sample_ids"] += 1
    validation["sample_ids"].add(sample_id)
    base_entry = validation["base_samples"].setdefault(
        base_id,
        {
            "information": set(),
            "payload": None,
            "payload_mismatch": False,
        },
    )
    base_entry["information"].add(sample.information)
    payload = (
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
    if base_entry["payload"] is None:
        base_entry["payload"] = payload
    elif base_entry["payload"] != payload:
        base_entry["payload_mismatch"] = True
    inventory = inventories[sample.dataset_family]
    from .oracle import gold_bool, gold_pool

    if sample.task == "T1":
        expected_gold = gold_bool(inventory, sample.candidate_ids[0], sample.A1_id, sample.A_branch_ids, sample.B_ids)
    elif sample.task == "T2":
        expected_gold = gold_pool(inventory, sample.candidate_ids, sample.A1_id, sample.A_branch_ids, sample.B_ids)
    else:
        expected_gold = bool(gold_pool(inventory, sample.candidate_ids, sample.A1_id, sample.A_branch_ids, sample.B_ids))
    if sample.gold_answer != expected_gold:
        validation["oracle_mismatch_count"] += 1


def _finalize_full_stream_validation(validation: dict[str, Any]) -> dict[str, Any]:
    missing_pairs = 0
    payload_mismatch = 0
    for item in validation["base_samples"].values():
        if item["information"] != {"I0", "I1", "I2"}:
            missing_pairs += 1
        if item["payload_mismatch"]:
            payload_mismatch += 1
    return {
        "base_sample_count": len(validation["base_samples"]),
        "duplicate_sample_hashes": validation["duplicate_sample_hashes"],
        "duplicate_request_hashes": validation["duplicate_request_hashes"],
        "duplicate_sample_ids": validation["duplicate_sample_ids"],
        "unpaired_base_sample_count": missing_pairs,
        "paired_payload_mismatch_count": payload_mismatch,
        "oracle_mismatch_count": validation["oracle_mismatch_count"],
        "passed": not any(
            [
                validation["duplicate_sample_hashes"],
                validation["duplicate_request_hashes"],
                validation["duplicate_sample_ids"],
                missing_pairs,
                payload_mismatch,
                validation["oracle_mismatch_count"],
            ]
        ),
    }


def _empty_full_light_validation() -> dict[str, Any]:
    return {
        "unpaired_base_sample_count": 0,
        "paired_payload_mismatch_count": 0,
        "oracle_mismatch_count": 0,
    }


def _update_full_light_validation(
    validation: dict[str, Any],
    base_samples: list[Any],
    inventories: dict[str, Inventory],
) -> None:
    if len(base_samples) != len(INFORMATION_LEVELS):
        validation["unpaired_base_sample_count"] += 1
        return
    information = {sample.information for sample in base_samples}
    if information != set(INFORMATION_LEVELS):
        validation["unpaired_base_sample_count"] += 1
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
        for sample in base_samples
    }
    if len(payloads) != 1:
        validation["paired_payload_mismatch_count"] += 1
    sample = base_samples[0]
    inventory = inventories[sample.dataset_family]
    from .oracle import gold_bool, gold_pool

    if sample.task == "T1":
        expected_gold = gold_bool(inventory, sample.candidate_ids[0], sample.A1_id, sample.A_branch_ids, sample.B_ids)
    elif sample.task == "T2":
        expected_gold = gold_pool(inventory, sample.candidate_ids, sample.A1_id, sample.A_branch_ids, sample.B_ids)
    else:
        expected_gold = bool(gold_pool(inventory, sample.candidate_ids, sample.A1_id, sample.A_branch_ids, sample.B_ids))
    if sample.gold_answer != expected_gold:
        validation["oracle_mismatch_count"] += 1


def _finalize_full_light_validation(validation: dict[str, Any], base_count: int) -> dict[str, Any]:
    return {
        "base_sample_count": base_count,
        "duplicate_check_policy": "not_materialized_during_full_stream_freeze",
        "unpaired_base_sample_count": validation["unpaired_base_sample_count"],
        "paired_payload_mismatch_count": validation["paired_payload_mismatch_count"],
        "oracle_mismatch_count": validation["oracle_mismatch_count"],
        "passed": not any(
            [
                validation["unpaired_base_sample_count"],
                validation["paired_payload_mismatch_count"],
                validation["oracle_mismatch_count"],
            ]
        ),
    }


def _full_sample_hashes_from_shards(shard_paths: list[Path]) -> Iterator[str]:
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)["sample_hash"]


def _scan_existing_full_sample_shards(samples_dir: Path) -> dict[str, Any]:
    shard_paths = sorted(samples_dir.glob("part-*.jsonl"))
    sample_hasher = hashlib.sha256()
    sample_count = 0
    information_count = len(INFORMATION_LEVELS)
    for expected_index, shard_path in enumerate(shard_paths):
        expected_name = f"part-{expected_index:05d}.jsonl"
        if shard_path.name != expected_name:
            raise ValueError(f"non-contiguous full sample shard sequence: expected {expected_name}, got {shard_path.name}")
        shard_count = 0
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample_hash = json.loads(line)["sample_hash"]
                sample_hasher.update(sample_hash.encode("utf-8"))
                sample_hasher.update(b"\n")
                shard_count += 1
        if shard_count == 0:
            raise ValueError(f"empty full sample shard: {shard_path}")
        if shard_count % information_count != 0:
            raise ValueError(f"full sample shard does not align to paired information groups: {shard_path}")
        sample_count += shard_count
    return {
        "shard_paths": shard_paths,
        "sample_count": sample_count,
        "base_count": sample_count // information_count,
        "sample_hash_state": sample_hasher,
    }


def run_freeze(config: ExperimentConfig, audit_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    feasibility_path = audit_dir / "feasibility.jsonl"
    if config.product.tier == "full":
        inventories = load_inventories(config.resources)
        samples_dir = ensure_dir(out_dir / "samples")
        shard_size = max(1, config.product.shard_size)
        existing = _scan_existing_full_sample_shards(samples_dir)
        current_rows: list[Any] = []
        shard_paths: list[str] = [str(path) for path in existing["shard_paths"]]
        sample_hasher = existing["sample_hash_state"]
        base_count = int(existing["base_count"])
        sample_count = int(existing["sample_count"])
        shard_index = len(shard_paths)
        validation = _empty_full_light_validation()
        resumed_base_count = base_count
        existing_hashes = _full_sample_hashes_from_shards(existing["shard_paths"])

        def write_full_progress(latest_shard: str | None = None, phase: str = "generating") -> None:
            write_json(
                out_dir / "freeze_progress.json",
                {
                    "product_id": config.product.product_id,
                    "product_tier": config.product.tier,
                    "phase": phase,
                    "resumed_base_sample_count": resumed_base_count,
                    "prefix_replay_base_sample_count": skipped_base_count,
                    "base_sample_count": base_count,
                    "sample_count": sample_count,
                    "sample_shard_count": len(shard_paths),
                    "latest_shard": latest_shard,
                    "complete": False,
                },
            )

        skipped_base_count = 0
        for base_samples in iter_extended_samples(config, inventories):
            if skipped_base_count < resumed_base_count:
                for sample in base_samples:
                    expected_hash = next(existing_hashes, None)
                    if expected_hash != sample.sample_hash:
                        raise ValueError(
                            "existing full sample shard prefix does not match regenerated deterministic sequence"
                        )
                _update_full_light_validation(validation, base_samples, inventories)
                skipped_base_count += 1
                if skipped_base_count == resumed_base_count or skipped_base_count % 10000 == 0:
                    write_full_progress(phase="replaying_existing_shards")
                continue
            base_count += 1
            _update_full_light_validation(validation, base_samples, inventories)
            for sample in base_samples:
                current_rows.append(sample)
                sample_hasher.update(sample.sample_hash.encode("utf-8"))
                sample_hasher.update(b"\n")
                sample_count += 1
            if len(current_rows) >= shard_size:
                shard_path = samples_dir / f"part-{shard_index:05d}.jsonl"
                write_jsonl(shard_path, current_rows)
                shard_paths.append(str(shard_path))
                current_rows = []
                shard_index += 1
                write_full_progress(str(shard_path))
        if current_rows:
            shard_path = samples_dir / f"part-{shard_index:05d}.jsonl"
            write_jsonl(shard_path, current_rows)
            shard_paths.append(str(shard_path))
            write_full_progress(str(shard_path))
        if next(existing_hashes, None) is not None:
            raise ValueError("existing full sample shards contain more rows than the regenerated deterministic prefix")
        manifest = extended_manifest_template(config)
        manifest["base_sample_count"] = base_count
        manifest["sample_count"] = sample_count
        manifest["sample_shards"] = shard_paths
        manifest["dataset_sample_hash"] = sample_hasher.hexdigest()
        manifest["stream_validation"] = _finalize_full_light_validation(validation, base_count)
        manifest["resume"] = {
            "resumed_from_existing_shards": resumed_base_count > 0,
            "resumed_base_sample_count": resumed_base_count,
            "resumed_sample_count": int(existing["sample_count"]),
        }
        manifest["source_feasibility"] = str(feasibility_path) if feasibility_path.exists() else None
        write_json(out_dir / "sample_manifest.json", manifest)
        write_dataset_config_snapshot(config, out_dir)
        write_json(out_dir / "stage_status.json", {"stage": "freeze", "complete": False, "reason": "frozen sample shards are not a completed run"})
        return {"samples": str(samples_dir), "sample_count": sample_count, "manifest": str(out_dir / "sample_manifest.json")}
    if not feasibility_path.exists():
        raise FileNotFoundError(f"feasibility table not found: {feasibility_path}")
    inventories = load_inventories(config.resources)
    rows = _feasibility_rows(feasibility_path)
    total_strata = len(config.resources.dataset_families) * len(config.tasks) * len(config.matrix.a_count) * len(config.matrix.b_count)
    completed_strata = 0

    def write_freeze_progress(event: dict[str, Any]) -> None:
        nonlocal completed_strata
        completed_strata += 1
        write_json(
            out_dir / "freeze_progress.json",
            {
                "product_id": config.product.product_id,
                "product_tier": config.product.tier,
                "completed_strata": completed_strata,
                "total_strata": total_strata,
                "base_sample_count": event.get("base_sample_count", 0),
                "sample_count": event.get("sample_count", 0),
                "latest_stratum": event,
            },
        )

    samples, manifest = generate_frozen_samples(config, inventories, rows, progress_callback=write_freeze_progress)
    write_jsonl(out_dir / "samples.jsonl", samples)
    manifest["dataset_sample_hash"] = stable_hash([sample.sample_hash for sample in samples], length=32)
    manifest["source_feasibility"] = "feasibility.jsonl" if feasibility_path.parent.resolve() == out_dir.resolve() else str(feasibility_path)
    write_json(out_dir / "sample_manifest.json", manifest)
    write_dataset_config_snapshot(config, out_dir)
    write_json(out_dir / "stage_status.json", {"stage": "freeze", "complete": False, "reason": "frozen samples are not a completed run"})
    return {"samples": str(out_dir / "samples.jsonl"), "sample_count": len(samples), "manifest": str(out_dir / "sample_manifest.json")}


def run_render(samples_path: Path, resource_snapshot_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    resources = __import__("json").load(resource_snapshot_path.open("r", encoding="utf-8"))
    if samples_path.is_dir():
        requests_dir = ensure_dir(out_dir / "requests")
        shard_paths: list[str] = []
        request_hasher = hashlib.sha256()
        request_count = 0
        for shard_index, sample_shard in enumerate(sorted(samples_path.glob("*.jsonl"))):
            samples = read_jsonl(sample_shard)
            requests = render_samples(samples, resources)
            request_path = requests_dir / f"part-{shard_index:05d}.jsonl"
            write_jsonl(request_path, requests)
            shard_paths.append(str(request_path))
            for request in requests:
                request_hasher.update(request.request_hash.encode("utf-8"))
                request_hasher.update(b"\n")
            request_count += len(requests)
        write_json(
            out_dir / "render_manifest.json",
            {
                "samples_path": str(samples_path),
                "sample_shards": [str(path) for path in sorted(samples_path.glob("*.jsonl"))],
                "request_shards": shard_paths,
                "resource_snapshot_path": "resource_snapshot.json" if resource_snapshot_path.parent.resolve() == out_dir.resolve() else str(resource_snapshot_path),
                "renderer_version": RENDERING_TEMPLATE_VERSION,
                "language_policy": "wordnet=en, babelnet=zh",
                "information_policy": "I0=no gloss, I1=concept gloss only, I2=concept and candidate gloss",
                "request_count": request_count,
                "request_hash": request_hasher.hexdigest(),
            },
        )
        write_json(out_dir / "stage_status.json", {"stage": "render", "complete": False, "reason": "rendered request shards are not a completed run"})
        return {"requests": str(requests_dir), "request_count": request_count}
    samples = read_jsonl(samples_path)
    requests = render_samples(samples, resources)
    write_jsonl(out_dir / "requests.jsonl", requests)
    write_json(
        out_dir / "render_manifest.json",
        {
            "samples_path": "samples.jsonl" if samples_path.parent.resolve() == out_dir.resolve() else str(samples_path),
            "resource_snapshot_path": "resource_snapshot.json" if resource_snapshot_path.parent.resolve() == out_dir.resolve() else str(resource_snapshot_path),
            "renderer_version": RENDERING_TEMPLATE_VERSION,
            "language_policy": "wordnet=en, babelnet=zh",
            "information_policy": "I0=no gloss, I1=concept gloss only, I2=concept and candidate gloss",
            "request_count": len(requests),
            "request_hash": stable_hash([request.request_hash for request in requests], length=32),
        },
    )
    write_json(out_dir / "stage_status.json", {"stage": "render", "complete": False, "reason": "rendered requests are not a completed run"})
    return {"requests": str(out_dir / "requests.jsonl"), "request_count": len(requests)}


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists() and not target.exists():
        if source.suffix == ".jsonl":
            write_jsonl(target, read_jsonl(source))
        else:
            write_json(target, read_json(source))


def _copy_dataset_config_snapshot(source: Path, target: Path) -> None:
    if source.exists():
        snapshot = read_json(source)
    elif target.exists():
        snapshot = read_json(target)
    else:
        return
    snapshot.pop("provider", None)
    write_json(target, snapshot)


def _copy_run_context(requests_path: Path, out_dir: Path) -> dict[str, str]:
    context: dict[str, str] = {}
    source_dir = requests_path.parent
    dataset_config_source = source_dir / "dataset_config_snapshot.json"
    if not dataset_config_source.exists():
        dataset_config_source = source_dir / "config_snapshot.json"
    dataset_config_target = out_dir / "dataset_config_snapshot.json"
    _copy_dataset_config_snapshot(dataset_config_source, dataset_config_target)
    if dataset_config_target.exists():
        context["dataset_config_snapshot.json"] = str(dataset_config_target)
    for name in [
        "resource_audit.json",
        "feasibility.jsonl",
        "resource_snapshot.json",
        "samples.jsonl",
        "sample_manifest.json",
        "render_manifest.json",
    ]:
        source = source_dir / name
        target = out_dir / name
        _copy_if_exists(source, target)
        if target.exists():
            context[name] = str(target)
    sample_manifest = out_dir / "sample_manifest.json"
    if sample_manifest.exists():
        manifest = read_json(sample_manifest)
        manifest["source_feasibility"] = "feasibility.jsonl"
        write_json(sample_manifest, manifest)
    render_manifest = out_dir / "render_manifest.json"
    if render_manifest.exists():
        manifest = read_json(render_manifest)
        manifest["samples_path"] = "samples.jsonl"
        manifest["resource_snapshot_path"] = "resource_snapshot.json"
        write_json(render_manifest, manifest)
    return context


def run_requests(config: ExperimentConfig, requests_path: Path, out_dir: Path, provider_name: str, limit: int | None = None) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    requests = read_jsonl(requests_path)
    context = _copy_run_context(requests_path, out_dir)
    run_requests_path = out_dir / "requests.jsonl"
    source_requests_path = str(requests_path)
    original_source_requests_path = str(requests_path)
    previous_manifest = read_json(out_dir / "run_manifest.json") if (out_dir / "run_manifest.json").exists() else {}
    if previous_manifest:
        source_requests_path = previous_manifest.get("source_requests_path", source_requests_path)
        original_source_requests_path = previous_manifest.get(
            "original_source_requests_path", original_source_requests_path
        )
    if requests_path.resolve() != run_requests_path.resolve():
        write_jsonl(run_requests_path, requests)
    responses_path = out_dir / "responses.jsonl"
    write_config_snapshot(config, out_dir)
    response_state = read_response_run_state(responses_path)
    completed_request_ids = response_state.completed_request_ids
    pending = [row for row in requests if row["request_id"] not in completed_request_ids]
    if limit is not None:
        pending = pending[:limit]
    run_provider_config = provider_config_for_run(config, provider_name)
    provider = get_provider(provider_name, run_provider_config)
    new_response_rows, response_count_after, success_count_after, run_status = run_provider_requests(
        provider,
        pending,
        responses_path,
        out_dir,
        provider_name,
        run_provider_config,
        "run",
        len(requests),
        response_state.counts["terminal_count"],
        response_state.response_count,
        response_state,
    )
    _write_run_manifest(
        out_dir,
        provider_name,
        config,
        run_requests_path,
        source_requests_path,
        len(requests),
        len(completed_request_ids),
        len(new_response_rows),
        response_count_after,
        success_count_after,
        run_status,
        "run",
        context,
        original_source_requests_path,
        run_provider_config,
    )
    return {"responses": str(responses_path), "attempted_count": len(new_response_rows)}


def resume_requests(config: ExperimentConfig, run_dir: Path, provider_name: str, limit: int | None = None) -> dict[str, Any]:
    requests_path = run_dir / "requests.jsonl"
    return run_requests_with_mode(config, requests_path, run_dir, provider_name, "resume", limit=limit)


def run_requests_with_mode(config: ExperimentConfig, requests_path: Path, out_dir: Path, provider_name: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    requests = read_jsonl(requests_path)
    context = _copy_run_context(requests_path, out_dir)
    run_requests_path = out_dir / "requests.jsonl"
    source_requests_path = str(requests_path)
    original_source_requests_path = str(requests_path)
    previous_manifest = read_json(out_dir / "run_manifest.json") if (out_dir / "run_manifest.json").exists() else {}
    if previous_manifest:
        source_requests_path = previous_manifest.get("source_requests_path", source_requests_path)
        original_source_requests_path = previous_manifest.get(
            "original_source_requests_path", original_source_requests_path
        )
    if requests_path.resolve() != run_requests_path.resolve():
        write_jsonl(run_requests_path, requests)
    responses_path = out_dir / "responses.jsonl"
    write_config_snapshot(config, out_dir)
    response_state = read_response_run_state(responses_path)
    completed_request_ids = response_state.completed_request_ids
    pending = [row for row in requests if row["request_id"] not in completed_request_ids]
    if limit is not None:
        pending = pending[:limit]
    run_provider_config = provider_config_for_run(config, provider_name)
    provider = get_provider(provider_name, run_provider_config)
    new_response_rows, response_count_after, success_count_after, run_status = run_provider_requests(
        provider,
        pending,
        responses_path,
        out_dir,
        provider_name,
        run_provider_config,
        mode,
        len(requests),
        response_state.counts["terminal_count"],
        response_state.response_count,
        response_state,
    )
    _write_run_manifest(
        out_dir,
        provider_name,
        config,
        run_requests_path,
        source_requests_path,
        len(requests),
        len(completed_request_ids),
        len(new_response_rows),
        response_count_after,
        success_count_after,
        run_status,
        mode,
        context,
        original_source_requests_path,
        run_provider_config,
    )
    return {"responses": str(responses_path), "attempted_count": len(new_response_rows)}


def _write_run_manifest(
    out_dir: Path,
    provider_name: str,
    config: ExperimentConfig,
    run_requests_path: Path,
    source_requests_path: str,
    request_count: int,
    previous_terminal_count: int,
    attempted_count: int,
    response_count: int,
    success_count: int,
    run_status: str,
    mode: str,
    context: dict[str, str],
    original_source_requests_path: str,
    run_provider_config: Any | None = None,
) -> None:
    path = out_dir / "run_manifest.json"
    previous = read_json(path) if path.exists() else {}
    events = list(previous.get("run_events", []))
    provider_config = run_provider_config or config.provider
    response_counts = read_response_run_state(out_dir / "responses.jsonl").counts
    terminal_count = response_counts["terminal_count"]
    permanent_error_count = response_counts["permanent_error_count"]
    retryable_failure_count = response_counts["retryable_failure_count"]
    events.append(
        {
            "mode": mode,
            "provider": provider_name,
            "model": provider_config.model,
            "thinking": provider_config.thinking,
            "reasoning_effort": provider_config.reasoning_effort,
            "status": run_status,
            "max_concurrency": provider_config.max_concurrency,
            "concurrency_mode": provider_config.concurrency_mode,
            "previous_terminal_count": previous_terminal_count,
            "attempted_count": attempted_count,
            "response_count_after": response_count,
            "success_count_after": success_count,
            "terminal_count_after": terminal_count,
            "permanent_error_count_after": permanent_error_count,
            "retryable_failure_count_after": retryable_failure_count,
        }
    )
    first_mode = previous.get("first_run_mode", mode)
    complete = terminal_count >= request_count
    write_json(
        path,
        {
            "provider": provider_name,
            "model": provider_config.model,
            "thinking": provider_config.thinking,
            "reasoning_effort": provider_config.reasoning_effort,
            "requests_path": str(run_requests_path),
            "source_requests_path": source_requests_path,
            "original_source_requests_path": original_source_requests_path,
            "request_count": request_count,
            "previous_terminal_count": previous_terminal_count,
            "attempted_count": attempted_count,
            "response_count": response_count,
            "success_count": success_count,
            "terminal_count": terminal_count,
            "permanent_error_count": permanent_error_count,
            "retryable_failure_count": retryable_failure_count,
            "run_mode": mode,
            "first_run_mode": first_mode,
            "latest_run_mode": mode,
            "run_events": events,
            "context_artifacts": context or previous.get("context_artifacts", {}),
            "max_concurrency": provider_config.max_concurrency,
            "concurrency_mode": provider_config.concurrency_mode,
            "provider_capability": {
                "protocol": provider_config.protocol,
                "base_url": provider_config.base_url,
                "api_key_env": provider_config.api_key_env,
                "supports_streaming": provider_config.supports_streaming,
                "supports_reasoning_channel": provider_config.supports_reasoning_channel,
                "thinking_mapping": provider_config.thinking_mapping,
                "request_adapter": provider_config.request_adapter,
                "recommended_max_concurrency": provider_config.recommended_max_concurrency,
            },
            "generation_config": {
                "temperature": provider_config.temperature,
                "top_p": provider_config.top_p,
                "max_answer_tokens": provider_config.max_answer_tokens,
                "request_interval_seconds": provider_config.request_interval_seconds,
                "retry_backoff_seconds": provider_config.retry_backoff_seconds,
                "retryable_cooldown_seconds": provider_config.retryable_cooldown_seconds,
                "reasoning_effort": provider_config.reasoning_effort,
                "save_reasoning": provider_config.save_reasoning,
                "reasoning_char_limit": provider_config.reasoning_char_limit,
            },
            "status": run_status,
            "complete": complete,
        },
    )
    write_json(out_dir / "stage_status.json", {"stage": "run", "complete": complete, "provider": provider_name})


def run_reparse(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    requests = read_jsonl(run_dir / "requests.jsonl")
    responses = read_jsonl(run_dir / "responses.jsonl")
    predictions, scores, metrics = build_predictions_and_scores(requests, responses, config.answer_parse_char_limit)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "scores.jsonl", scores)
    write_json(
        run_dir / "reparse_manifest.json",
        {
            "requests": len(requests),
            "responses": len(responses),
            "predictions": len(predictions),
            "scores": len(scores),
            "mode": "reparse",
            "no_api_requests_sent": True,
            "summary": metrics,
        },
    )
    return {"predictions": len(predictions), "scores": len(scores), "reparse_manifest": str(run_dir / "reparse_manifest.json")}


def run_report(run_dir: Path) -> dict[str, Any]:
    paths = generate_evaluation_report(run_dir)
    return {
        "evaluation_rows": str(paths.rows_path),
        "evaluation_metrics": str(paths.metrics_path),
        "evaluation_report": str(paths.notebook_path),
    }
