from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_provider_capabilities, provider_config_for_run
from .io_utils import read_json, read_jsonl
from .run_control import response_terminal_counts, terminal_request_ids


VALID_REQUEST_ADAPTERS = {
    "zhipu_thinking_body",
    "deepseek_reasoning_effort",
    "aliyun_enable_thinking",
    "openai_reasoning_effort",
    "nvidia_gemma_chat_template",
    "vllm_gemma_chat_template",
    "plain_openai",
    "mock",
}


def _iter_request_rows(requests_path: Path):
    paths = sorted(requests_path.glob("*.jsonl")) if requests_path.is_dir() else [requests_path]
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _audit_request_rows(requests_path: Path, renderer_version: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    request_ids: set[str] = set()
    request_hashes: set[str] = set()
    row_count = 0
    first_row: dict[str, Any] | None = None
    required = {"request_id", "sample_id", "base_sample_id", "task", "dataset_family", "information", "request_hash", "provider_payload", "sample"}
    for index, row in enumerate(_iter_request_rows(requests_path), 1):
        row_count += 1
        if first_row is None:
            first_row = row
        missing = sorted(field for field in required if field not in row)
        if missing:
            errors.append(f"request row {index} missing fields: {missing}")
            continue
        request_id = row["request_id"]
        request_hash = row["request_hash"]
        if request_id in request_ids:
            errors.append(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)
        if request_hash in request_hashes:
            errors.append(f"duplicate request_hash: {request_hash}")
        request_hashes.add(request_hash)
        if not str(request_id).endswith(str(request_hash)[:16]):
            errors.append(f"request_id does not end with request_hash prefix: {request_id}")
        nested_hash = row.get("sample", {}).get("request_hash")
        if nested_hash != request_hash:
            errors.append(f"nested sample request_hash mismatch: {request_id}")
        messages = row.get("provider_payload", {}).get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"provider_payload.messages is empty or invalid: {request_id}")
        if renderer_version and row.get("renderer_version") != renderer_version:
            errors.append(f"render_manifest renderer_version {renderer_version!r} does not match request row {index}")
    return {"errors": errors, "request_count": row_count, "first_row": first_row}


def _dataset_context_paths(requests_path: Path) -> dict[str, Path]:
    source_dir = requests_path.parent
    return {
        "dataset_config_snapshot": source_dir / "dataset_config_snapshot.json",
        "sample_manifest": source_dir / "sample_manifest.json",
        "render_manifest": source_dir / "render_manifest.json",
    }


def preflight_run(
    config: ExperimentConfig,
    requests_path: Path,
    out_dir: Path,
    provider_name: str,
    limit: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    requests_path = Path(requests_path)
    out_dir = Path(out_dir)
    if not requests_path.exists():
        return {
            "passed": False,
            "errors": [f"requests_path does not exist: {requests_path}"],
            "warnings": [],
        }
    context_paths = _dataset_context_paths(requests_path)
    render_manifest = {}
    render_manifest_path = context_paths["render_manifest"]
    if render_manifest_path.exists():
        render_manifest = read_json(render_manifest_path)
    request_audit = _audit_request_rows(requests_path, render_manifest.get("renderer_version"))
    request_count = request_audit["request_count"]
    if not request_count:
        errors.append(f"requests_path has no rows: {requests_path}")
    errors.extend(request_audit["errors"])

    dataset_snapshot = {}
    dataset_snapshot_path = context_paths["dataset_config_snapshot"]
    if dataset_snapshot_path.exists():
        dataset_snapshot = read_json(dataset_snapshot_path)
        if "provider" in dataset_snapshot:
            errors.append(f"dataset_config_snapshot must not contain provider: {dataset_snapshot_path}")
    else:
        warnings.append(f"dataset_config_snapshot not found beside requests: {dataset_snapshot_path}")

    sample_manifest = {}
    sample_manifest_path = context_paths["sample_manifest"]
    if sample_manifest_path.exists():
        sample_manifest = read_json(sample_manifest_path)
        product_id = sample_manifest.get("product_id")
        if product_id and product_id != config.product.product_id:
            errors.append(f"sample_manifest product_id {product_id!r} does not match config {config.product.product_id!r}")
        product_tier = sample_manifest.get("product_tier")
        if product_tier and product_tier != config.product.tier:
            errors.append(f"sample_manifest product_tier {product_tier!r} does not match config {config.product.tier!r}")
    else:
        warnings.append(f"sample_manifest not found beside requests: {sample_manifest_path}")

    if render_manifest_path.exists():
        request_count = render_manifest.get("request_count")
        if request_count is not None and request_count != request_audit["request_count"]:
            errors.append(
                f"render_manifest request_count {request_count} does not match requests rows {request_audit['request_count']}"
            )
    else:
        warnings.append(f"render_manifest not found beside requests: {render_manifest_path}")

    configured_provider = config.provider.name
    if provider_name != configured_provider and provider_name != "mock":
        errors.append(
            f"provider override {provider_name!r} must match config provider {configured_provider!r}; only mock is allowed"
        )
    capabilities = load_provider_capabilities()
    if provider_name != "mock" and provider_name not in capabilities:
        errors.append(f"provider {provider_name!r} is not defined in project/configs/providers.toml")
    run_provider = provider_config_for_run(config, provider_name)
    if run_provider.thinking not in {"on", "off"}:
        errors.append(f"provider.thinking must be 'on' or 'off', got {run_provider.thinking!r}")
    if run_provider.thinking == "on" and not run_provider.supports_reasoning_channel:
        errors.append(f"provider {provider_name!r} does not support reasoning channel for thinking=on")
    if run_provider.reasoning_effort and run_provider.reasoning_effort not in {"low", "medium", "high", "max"}:
        errors.append(f"provider.reasoning_effort must be low, medium, high, or max, got {run_provider.reasoning_effort!r}")
    if (
        provider_name == "nvidia"
        and run_provider.model == "openai/gpt-oss-120b"
        and run_provider.reasoning_effort
        and run_provider.reasoning_effort not in {"low", "medium", "high"}
    ):
        errors.append("NVIDIA openai/gpt-oss-120b reasoning_effort must be low, medium, or high")
    if run_provider.reasoning_effort and run_provider.thinking != "on":
        errors.append("provider.reasoning_effort requires provider.thinking = 'on'")
    if provider_name != "mock" and run_provider.request_adapter not in VALID_REQUEST_ADAPTERS:
        errors.append(f"provider.request_adapter must be one of {sorted(VALID_REQUEST_ADAPTERS)}, got {run_provider.request_adapter!r}")
    if run_provider.max_concurrency < 1:
        errors.append("max_concurrency must be >= 1")
    if run_provider.concurrency_mode not in {"fixed", "adaptive"}:
        errors.append(f"concurrency_mode must be fixed or adaptive, got {run_provider.concurrency_mode!r}")
    if run_provider.max_concurrency > run_provider.recommended_max_concurrency:
        warnings.append(
            f"max_concurrency {run_provider.max_concurrency} exceeds provider recommended_max_concurrency "
            f"{run_provider.recommended_max_concurrency}"
        )

    existing_rows = read_jsonl(out_dir / "responses.jsonl")
    terminal_ids = terminal_request_ids(existing_rows)
    response_counts = response_terminal_counts(existing_rows)
    pending_count = max(request_audit["request_count"] - len(terminal_ids), 0)
    planned_attempt_count = pending_count if limit is None else min(limit, pending_count)
    if out_dir.exists() and existing_rows:
        warnings.append(
            f"out_dir has existing responses; run will resume {planned_attempt_count} pending requests "
            f"({response_counts['terminal_count']} terminal)"
        )

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": {
            "requests_path": str(requests_path),
            "request_count": request_audit["request_count"],
            "product_id": sample_manifest.get("product_id") or config.product.product_id,
            "product_tier": sample_manifest.get("product_tier") or config.product.tier,
            "renderer_version": render_manifest.get("renderer_version"),
        },
        "provider_run": {
            "provider": provider_name,
            "configured_provider": configured_provider,
            "model": run_provider.model,
            "thinking": run_provider.thinking,
            "reasoning_effort": run_provider.reasoning_effort,
            "protocol": run_provider.protocol,
            "api_key_env": run_provider.api_key_env,
            "supports_streaming": run_provider.supports_streaming,
            "supports_reasoning_channel": run_provider.supports_reasoning_channel,
            "request_adapter": run_provider.request_adapter,
            "concurrency_mode": run_provider.concurrency_mode,
            "max_concurrency": run_provider.max_concurrency,
            "max_retries": run_provider.max_retries,
        },
        "execution_plan": {
            "out_dir": str(out_dir),
            "existing_response_rows": len(existing_rows),
            "terminal_count": response_counts["terminal_count"],
            "pending_count": pending_count,
            "planned_attempt_count": planned_attempt_count,
            "limit": limit,
        },
    }
