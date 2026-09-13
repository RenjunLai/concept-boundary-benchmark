from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from .config import ProviderConfig
from .io_utils import ensure_dir, read_jsonl, to_plain, write_json


RETRYABLE_ERROR_MARKERS = (
    "429",
    "rate limit",
    "server overloaded",
    "timeout",
    "temporarily unavailable",
    "connection",
)

PAUSE_REQUEST_FILE = "pause.requested"
PROGRESS_WRITE_INTERVAL_SECONDS = 5.0


def is_retryable_response(row: dict[str, Any]) -> bool:
    if row.get("status") == "success":
        return False
    text = str(row.get("error_state") or "").lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def _response_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": row.get("request_id"),
        "status": row.get("status"),
        "error_state": row.get("error_state"),
    }


@dataclass
class ResponseRunState:
    completed_request_ids: set[str]
    latest_by_request: dict[str, dict[str, Any]]
    response_count: int
    counts: dict[str, int]


def terminal_request_ids(responses: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in responses:
        request_id = row.get("request_id")
        if not request_id:
            continue
        if row.get("status") == "success" or not is_retryable_response(row):
            ids.add(request_id)
    return ids


def response_terminal_counts(responses: list[dict[str, Any]]) -> dict[str, int]:
    latest: dict[str, dict[str, Any]] = {}
    for row in responses:
        request_id = row.get("request_id")
        if request_id:
            latest[request_id] = row
    success = 0
    retryable = 0
    permanent = 0
    for row in latest.values():
        if row.get("status") == "success":
            success += 1
        elif is_retryable_response(row):
            retryable += 1
        else:
            permanent += 1
    return {
        "success_count": success,
        "retryable_failure_count": retryable,
        "permanent_error_count": permanent,
        "terminal_count": success + permanent,
    }


def _response_count_bucket(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    if row.get("status") == "success":
        return "success_count"
    if is_retryable_response(row):
        return "retryable_failure_count"
    return "permanent_error_count"


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_plain(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()


def read_response_run_state(path: Path) -> ResponseRunState:
    completed_request_ids: set[str] = set()
    latest_by_request: dict[str, dict[str, Any]] = {}
    response_count = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                response_count += 1
                row = json.loads(line)
                request_id = row.get("request_id")
                if not request_id:
                    continue
                state = _response_state(row)
                if state.get("status") == "success" or not is_retryable_response(state):
                    completed_request_ids.add(request_id)
                latest_by_request[request_id] = state
    counts = response_terminal_counts(list(latest_by_request.values()))
    return ResponseRunState(
        completed_request_ids=completed_request_ids,
        latest_by_request=latest_by_request,
        response_count=response_count,
        counts=counts,
    )


class ResponseReceiver:
    def __init__(
        self,
        out_dir: Path,
        responses_path: Path,
        provider_name: str,
        config: ProviderConfig,
        mode: str,
        request_count: int,
        previous_terminal_count: int,
        existing_response_count: int,
        existing_state: ResponseRunState | None = None,
    ):
        self.out_dir = ensure_dir(out_dir)
        self.responses_path = responses_path
        self.provider_name = provider_name
        self.config = config
        self.mode = mode
        self.request_count = request_count
        self.previous_terminal_count = previous_terminal_count
        self.attempted_count = 0
        self.response_count = existing_response_count
        if existing_state is None:
            existing_state = read_response_run_state(self.responses_path)
        self.latest_by_request = dict(existing_state.latest_by_request)
        counts = existing_state.counts
        self.success_count = counts["success_count"]
        self.retryable_failure_count = counts["retryable_failure_count"]
        self.permanent_error_count = counts["permanent_error_count"]
        self.terminal_count = counts["terminal_count"]
        self.rows: list[dict[str, Any]] = []
        self.retry_attempt_count = 0
        self.status = "running"
        self._last_progress_write_at = 0.0
        self.write_progress(current_concurrency=0, force=True)

    def receive(self, row: Any, current_concurrency: int, request_index: int) -> dict[str, Any]:
        plain = self.append_response(row, request_index=request_index)
        self.rows.append(plain)
        self.attempted_count += 1
        self.response_count += 1
        request_id = plain.get("request_id")
        if request_id:
            old_state = self.latest_by_request.get(request_id)
            self._apply_count_delta(old_state, -1)
            new_state = _response_state(plain)
            self.latest_by_request[request_id] = new_state
            self._apply_count_delta(new_state, 1)
        self.write_progress(current_concurrency=current_concurrency)
        return plain

    def _apply_count_delta(self, row: dict[str, Any] | None, delta: int) -> None:
        bucket = _response_count_bucket(row)
        if bucket is None:
            return
        if bucket == "success_count":
            self.success_count += delta
            self.terminal_count += delta
        elif bucket == "permanent_error_count":
            self.permanent_error_count += delta
            self.terminal_count += delta
        elif bucket == "retryable_failure_count":
            self.retryable_failure_count += delta

    def append_response(self, row: Any, request_index: int) -> dict[str, Any]:
        plain = to_plain(row)
        plain["run_request_index"] = request_index
        plain["run_receive_sequence"] = self.response_count + 1
        ensure_dir(self.responses_path.parent)
        with self.responses_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(plain, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
        return plain

    def record_event(self, **event: Any) -> None:
        _append_jsonl(self.out_dir / "concurrency_events.jsonl", [event])

    def record_attempts(self, attempts: list[dict[str, Any]]) -> None:
        _append_jsonl(self.out_dir / "attempts.jsonl", attempts)

    def write_progress(self, current_concurrency: int, status: str | None = None, force: bool = False) -> None:
        if status is not None:
            self.status = status
        elif self.terminal_count >= self.request_count:
            self.status = "complete"
            force = True
        now = time.monotonic()
        if not force and status is None and now - self._last_progress_write_at < PROGRESS_WRITE_INTERVAL_SECONDS:
            return
        write_json(
            self.out_dir / "run_progress.json",
            {
                "provider": self.provider_name,
                "model": self.config.model,
                "thinking": self.config.thinking,
                "mode": self.mode,
                "max_concurrency": self.config.max_concurrency,
                "concurrency_mode": self.config.concurrency_mode,
                "request_interval_seconds": self.config.request_interval_seconds,
                "retry_backoff_seconds": self.config.retry_backoff_seconds,
                "retryable_cooldown_seconds": self.config.retryable_cooldown_seconds,
                "current_concurrency": current_concurrency,
                "status": self.status,
                "request_count": self.request_count,
                "previous_terminal_count": self.previous_terminal_count,
                "attempted_count": self.attempted_count,
                "response_count": self.response_count,
                "success_count": self.success_count,
                "terminal_count": self.terminal_count,
                "permanent_error_count": self.permanent_error_count,
                "retryable_failure_count": self.retryable_failure_count,
                "retry_attempt_count": self.retry_attempt_count,
                "pending_count": max(self.request_count - self.terminal_count, 0),
                "complete": self.terminal_count >= self.request_count,
            },
        )
        self._last_progress_write_at = now


def _sleep_seconds(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _pause_requested(out_dir: Path) -> bool:
    return (out_dir / PAUSE_REQUEST_FILE).exists()


def _clear_pause_request(out_dir: Path) -> None:
    (out_dir / PAUSE_REQUEST_FILE).unlink(missing_ok=True)


def _invoke_with_retries(
    provider,
    request: dict[str, Any],
    attempt_index: int,
    max_retries: int,
    retry_backoff_seconds: float = 0.0,
):
    last_response = None
    retry_attempt_count = 0
    attempts: list[dict[str, Any]] = []
    for retry_index in range(max_retries + 1):
        if retry_index > 0:
            _sleep_seconds(retry_backoff_seconds)
        response = provider.call(request, attempt_index=attempt_index + retry_index)
        last_response = response
        plain = to_plain(response)
        retryable = is_retryable_response(plain)
        attempts.append(
            {
                "request_id": request.get("request_id"),
                "attempt_id": plain.get("attempt_id"),
                "attempt_index": attempt_index + retry_index,
                "retry_index": retry_index,
                "status": plain.get("status"),
                "error_state": plain.get("error_state"),
                "retryable": retryable,
            }
        )
        if not retryable:
            return response, retry_attempt_count, attempts
        retry_attempt_count += 1
    return last_response, retry_attempt_count, attempts


def _run_fixed_single_worker(
    provider,
    pending: list[dict[str, Any]],
    receiver: ResponseReceiver,
    max_retries: int,
    request_interval_seconds: float,
    retry_backoff_seconds: float,
    retryable_cooldown_seconds: float,
) -> str:
    previous_result_retryable = False
    for index, request in enumerate(pending, 1):
        if _pause_requested(receiver.out_dir):
            return "paused"
        if index > 1:
            if previous_result_retryable:
                _sleep_seconds(retryable_cooldown_seconds)
            else:
                _sleep_seconds(request_interval_seconds)
            if _pause_requested(receiver.out_dir):
                return "paused"
        response, retry_count, attempts = _invoke_with_retries(
            provider, request, index, max_retries, retry_backoff_seconds
        )
        receiver.retry_attempt_count += retry_count
        receiver.record_attempts(attempts)
        row = receiver.receive(response, current_concurrency=1, request_index=index)
        previous_result_retryable = is_retryable_response(row)
    return "complete"


def _run_fixed_thread_pool(
    provider,
    pending: list[dict[str, Any]],
    receiver: ResponseReceiver,
    max_workers: int,
    max_retries: int,
    retry_backoff_seconds: float,
    request_interval_seconds: float,
    retryable_cooldown_seconds: float,
) -> str:
    paused = False
    next_index = 0
    attempt_index = 1
    next_submit_at = 0.0
    previous_result_retryable = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active: dict[Any, int] = {}

        def submit_until_capacity() -> None:
            nonlocal next_index, attempt_index, next_submit_at, previous_result_retryable, paused
            while next_index < len(pending) and len(active) < max_workers:
                if _pause_requested(receiver.out_dir):
                    paused = True
                    return
                now = time.monotonic()
                if next_submit_at > now:
                    _sleep_seconds(next_submit_at - now)
                    if _pause_requested(receiver.out_dir):
                        paused = True
                        return
                request = pending[next_index]
                active[
                    executor.submit(_invoke_with_retries, provider, request, attempt_index, max_retries, retry_backoff_seconds)
                ] = next_index + 1
                next_index += 1
                attempt_index += max_retries + 1
                delay = retryable_cooldown_seconds if previous_result_retryable else request_interval_seconds
                next_submit_at = time.monotonic() + delay
                previous_result_retryable = False

        submit_until_capacity()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                response, retry_count, attempts = future.result()
                receiver.retry_attempt_count += retry_count
                receiver.record_attempts(attempts)
                request_index = active.pop(future)
                row = receiver.receive(response, current_concurrency=len(active), request_index=request_index)
                previous_result_retryable = is_retryable_response(row) or any(attempt.get("retryable") for attempt in attempts)
            if _pause_requested(receiver.out_dir):
                paused = True
            else:
                submit_until_capacity()
    return "paused" if paused else "complete"


def _run_adaptive_thread_pool(
    provider,
    pending: list[dict[str, Any]],
    receiver: ResponseReceiver,
    max_workers: int,
    max_retries: int,
    retry_backoff_seconds: float,
    request_interval_seconds: float,
    retryable_cooldown_seconds: float,
    min_workers: int = 1,
) -> str:
    current_workers = max(min_workers, 1)
    next_index = 0
    attempt_index = 1
    recent_failures = 0
    next_submit_at = 0.0
    previous_result_retryable = False
    paused = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active: dict[Any, int] = {}

        def submit_until_capacity() -> None:
            nonlocal next_index, attempt_index, next_submit_at, previous_result_retryable, paused
            while next_index < len(pending) and len(active) < current_workers:
                if _pause_requested(receiver.out_dir):
                    paused = True
                    return
                now = time.monotonic()
                if next_submit_at > now:
                    _sleep_seconds(next_submit_at - now)
                    if _pause_requested(receiver.out_dir):
                        paused = True
                        return
                request = pending[next_index]
                active[
                    executor.submit(
                        _invoke_with_retries, provider, request, attempt_index, max_retries, retry_backoff_seconds
                    )
                ] = next_index + 1
                next_index += 1
                attempt_index += max_retries + 1
                delay = retryable_cooldown_seconds if previous_result_retryable else request_interval_seconds
                next_submit_at = time.monotonic() + delay
                previous_result_retryable = False

        submit_until_capacity()
        receiver.record_event(event="adaptive_start", current_concurrency=current_workers)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                response, retry_count, attempts = future.result()
                receiver.retry_attempt_count += retry_count
                receiver.record_attempts(attempts)
                request_index = active.pop(future)
                row = receiver.receive(response, current_concurrency=len(active), request_index=request_index)
                saw_retryable_attempt = any(attempt.get("retryable") for attempt in attempts)
                previous_result_retryable = is_retryable_response(row) or saw_retryable_attempt
                if is_retryable_response(row) or saw_retryable_attempt:
                    recent_failures += 1
                    if current_workers > min_workers:
                        current_workers -= 1
                        receiver.record_event(
                            event="adaptive_decrease",
                            current_concurrency=current_workers,
                            reason="retryable_failure",
                        )
                else:
                    recent_failures = 0
                    if _pause_requested(receiver.out_dir):
                        paused = True
                    elif current_workers < max_workers:
                        current_workers += 1
                        receiver.record_event(
                            event="adaptive_increase",
                            current_concurrency=current_workers,
                            reason="successful_response",
                        )
            if _pause_requested(receiver.out_dir):
                paused = True
            else:
                submit_until_capacity()
    return "paused" if paused else "complete"


def run_provider_requests(
    provider,
    pending: list[dict[str, Any]],
    responses_path: Path,
    out_dir: Path,
    provider_name: str,
    config: ProviderConfig,
    mode: str,
    request_count: int,
    previous_terminal_count: int,
    existing_response_count: int,
    existing_state: ResponseRunState | None = None,
) -> tuple[list[dict[str, Any]], int, int, str]:
    receiver = ResponseReceiver(
        out_dir=out_dir,
        responses_path=responses_path,
        provider_name=provider_name,
        config=config,
        mode=mode,
        request_count=request_count,
        previous_terminal_count=previous_terminal_count,
        existing_response_count=existing_response_count,
        existing_state=existing_state,
    )
    if not pending:
        receiver.write_progress(current_concurrency=0, status="complete")
        return receiver.rows, receiver.response_count, receiver.success_count, "complete"

    max_retries = max(0, int(config.max_retries))
    max_workers = max(1, int(config.max_concurrency))
    request_interval_seconds = max(0.0, float(config.request_interval_seconds))
    retry_backoff_seconds = max(0.0, float(config.retry_backoff_seconds))
    retryable_cooldown_seconds = max(0.0, float(config.retryable_cooldown_seconds))
    mode_name = config.concurrency_mode
    final_status = "complete"
    if mode_name == "fixed":
        if max_workers == 1:
            final_status = _run_fixed_single_worker(
                provider,
                pending,
                receiver,
                max_retries,
                request_interval_seconds,
                retry_backoff_seconds,
                retryable_cooldown_seconds,
            )
        else:
            final_status = _run_fixed_thread_pool(
                provider,
                pending,
                receiver,
                max_workers,
                max_retries,
                retry_backoff_seconds,
                request_interval_seconds,
                retryable_cooldown_seconds,
            )
    elif mode_name == "adaptive":
        final_status = _run_adaptive_thread_pool(
            provider,
            pending,
            receiver,
            max_workers,
            max_retries,
            retry_backoff_seconds,
            request_interval_seconds,
            retryable_cooldown_seconds,
        )
    else:
        raise ValueError(f"unknown provider concurrency_mode: {mode_name}")
    if receiver.terminal_count >= receiver.request_count:
        final_status = "complete"
    receiver.write_progress(current_concurrency=0, status=final_status)
    if final_status == "paused":
        _clear_pause_request(out_dir)
    return receiver.rows, receiver.response_count, receiver.success_count, final_status
