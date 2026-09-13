import time
import unittest
from pathlib import Path
import threading

from typer.testing import CliRunner

from bool_logic.config import load_config
from bool_logic.io_utils import read_json, read_jsonl, write_jsonl
from bool_logic.run_control import (
    PAUSE_REQUEST_FILE,
    PROGRESS_WRITE_INTERVAL_SECONDS,
    read_response_run_state,
    response_terminal_counts,
    run_provider_requests,
    terminal_request_ids,
)
from bool_logic.schemas import ProviderResponse


REPO_ROOT = Path(__file__).resolve().parents[2]


class RunControlTests(unittest.TestCase):
    def test_response_run_state_matches_existing_response_semantics(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            responses_path = Path("responses.jsonl")
            responses = [
                {
                    "request_id": "r1",
                    "status": "success",
                    "error_state": None,
                    "raw_response": {"payload": "x" * 1000},
                },
                {"request_id": "r2", "status": "error", "error_state": "429 rate limit"},
                {"request_id": "r3", "status": "error", "error_state": "BadRequestError"},
                {"request_id": "r1", "status": "error", "error_state": "429 rate limit"},
            ]
            write_jsonl(responses_path, responses)

            state = read_response_run_state(responses_path)

            self.assertEqual(state.response_count, len(responses))
            self.assertEqual(state.completed_request_ids, terminal_request_ids(responses))
            self.assertEqual(state.counts, response_terminal_counts(responses))
            self.assertEqual(set(state.latest_by_request["r1"]), {"request_id", "status", "error_state"})

    def test_fixed_receiver_writes_completed_results_incrementally(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            provider = RecordingProvider(out_dir)
            pending = [_request("r1"), _request("r2")]

            rows, response_count, success_count, status = run_provider_requests(
                provider,
                pending,
                responses_path,
                out_dir,
                "deepseek",
                config.provider,
                "run",
                request_count=2,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            self.assertEqual(provider.seen_line_counts, [0, 1])
            self.assertEqual(len(rows), 2)
            self.assertEqual(response_count, 2)
            self.assertEqual(success_count, 2)
            self.assertEqual(status, "complete")
            written = read_jsonl(responses_path)
            self.assertEqual(len(written), 2)
            self.assertEqual([row["run_request_index"] for row in written], [1, 2])
            self.assertEqual([row["run_receive_sequence"] for row in written], [1, 2])
            self.assertEqual(len(read_jsonl(out_dir / "attempts.jsonl")), 2)
            self.assertTrue(read_json(out_dir / "run_progress.json")["complete"])
            self.assertEqual(read_json(out_dir / "run_progress.json")["status"], "complete")

    def test_fixed_concurrency_uses_configured_worker_count(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(base_config, max_concurrency=2, concurrency_mode="fixed")
            provider = SlowProvider(delay=0.2)

            start = time.perf_counter()
            run_provider_requests(
                provider,
                [_request("r1"), _request("r2")],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=2,
                previous_terminal_count=0,
                existing_response_count=0,
            )
            elapsed = time.perf_counter() - start

            self.assertLess(elapsed, 0.35)
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["concurrency_mode"], "fixed")
            self.assertEqual(progress["max_concurrency"], 2)
            self.assertEqual(progress["request_interval_seconds"], 0.0)
            self.assertEqual(progress["retry_backoff_seconds"], 0.0)
            self.assertEqual(progress["retryable_cooldown_seconds"], 0.0)

    def test_fixed_single_worker_respects_request_interval(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(
                base_config,
                max_concurrency=1,
                concurrency_mode="fixed",
                request_interval_seconds=0.05,
            )
            provider = RecordingProvider(out_dir)

            start = time.perf_counter()
            run_provider_requests(
                provider,
                [_request("r1"), _request("r2")],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=2,
                previous_terminal_count=0,
                existing_response_count=0,
            )
            elapsed = time.perf_counter() - start

            self.assertGreaterEqual(elapsed, 0.05)
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["request_interval_seconds"], 0.05)

    def test_retryable_response_respects_retry_backoff(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(
                base_config,
                max_concurrency=1,
                concurrency_mode="fixed",
                max_retries=1,
                retry_backoff_seconds=0.05,
            )
            provider = RetryableThenSuccessfulProvider()

            start = time.perf_counter()
            rows, _, success_count, _ = run_provider_requests(
                provider,
                [_request("r1")],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=1,
                previous_terminal_count=0,
                existing_response_count=0,
            )
            elapsed = time.perf_counter() - start

            self.assertGreaterEqual(elapsed, 0.05)
            self.assertEqual(len(rows), 1)
            self.assertEqual(success_count, 1)
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["retry_backoff_seconds"], 0.05)
            self.assertEqual(progress["retry_attempt_count"], 1)
            attempts = read_jsonl(out_dir / "attempts.jsonl")
            self.assertEqual([row["retry_index"] for row in attempts], [0, 1])

    def test_retryable_terminal_response_cools_down_before_next_request(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(
                base_config,
                max_concurrency=1,
                concurrency_mode="fixed",
                max_retries=0,
                request_interval_seconds=0.0,
                retryable_cooldown_seconds=0.05,
            )
            provider = RetryableThenSuccessfulProvider()

            start = time.perf_counter()
            rows, _, success_count, _ = run_provider_requests(
                provider,
                [_request("r1"), _request("r2")],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=2,
                previous_terminal_count=0,
                existing_response_count=0,
            )
            elapsed = time.perf_counter() - start

            self.assertGreaterEqual(elapsed, 0.05)
            self.assertEqual(len(rows), 2)
            self.assertEqual(success_count, 1)
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["retryable_cooldown_seconds"], 0.05)

    def test_adaptive_concurrency_records_adjustment_events(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(base_config, max_concurrency=2, concurrency_mode="adaptive")
            provider = FailingThenSuccessfulProvider()

            rows, _, success_count, _ = run_provider_requests(
                provider,
                [_request("r1"), _request("r2"), _request("r3")],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=3,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(success_count, 3)
            events = read_jsonl(out_dir / "concurrency_events.jsonl")
            self.assertTrue(any(row["event"] == "adaptive_increase" for row in events))
            self.assertTrue(any(row["event"] == "adaptive_decrease" for row in events))
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["retry_attempt_count"], 1)
            attempts = read_jsonl(out_dir / "attempts.jsonl")
            self.assertEqual(len(attempts), 4)
            self.assertTrue(any(row["retryable"] for row in attempts))

    def test_fixed_thread_pool_keeps_bounded_active_window(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(base_config, max_concurrency=2, concurrency_mode="fixed")
            provider = ActiveCountingProvider(delay=0.05)

            rows, _, success_count, status = run_provider_requests(
                provider,
                [_request(f"r{i}") for i in range(6)],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=6,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            self.assertEqual(len(rows), 6)
            self.assertEqual(success_count, 6)
            self.assertEqual(status, "complete")
            self.assertLessEqual(provider.max_active, 2)

    def test_cooperative_pause_waits_for_active_and_stops_submitting(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(base_config, max_concurrency=2, concurrency_mode="fixed")
            provider = PauseAfterFirstProvider(out_dir)

            rows, _, _, status = run_provider_requests(
                provider,
                [_request(f"r{i}") for i in range(5)],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=5,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            self.assertEqual(status, "paused")
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(read_jsonl(responses_path)), 2)
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["status"], "paused")
            self.assertFalse((out_dir / PAUSE_REQUEST_FILE).exists())

    def test_adaptive_pause_stops_submitting_without_increasing_window(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = _with_provider(base_config, max_concurrency=4, concurrency_mode="adaptive")
            provider = PauseAfterFirstProvider(out_dir)

            rows, _, _, status = run_provider_requests(
                provider,
                [_request(f"r{i}") for i in range(5)],
                responses_path,
                out_dir,
                "deepseek",
                config,
                "run",
                request_count=5,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            self.assertEqual(status, "paused")
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(read_jsonl(responses_path)), 1)
            events = read_jsonl(out_dir / "concurrency_events.jsonl")
            self.assertFalse(any(row["event"] == "adaptive_increase" for row in events))
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(progress["status"], "paused")
            self.assertEqual(progress["current_concurrency"], 0)
            self.assertFalse((out_dir / PAUSE_REQUEST_FILE).exists())

    def test_resume_preserves_existing_attempt_and_event_logs(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            write_jsonl(out_dir / "attempts.jsonl", [{"request_id": "old", "attempt_index": 1}])
            write_jsonl(out_dir / "concurrency_events.jsonl", [{"event": "old_event"}])
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

            run_provider_requests(
                RecordingProvider(out_dir),
                [_request("r1")],
                responses_path,
                out_dir,
                "deepseek",
                config.provider,
                "resume",
                request_count=2,
                previous_terminal_count=1,
                existing_response_count=1,
            )

            attempts = read_jsonl(out_dir / "attempts.jsonl")
            events = read_jsonl(out_dir / "concurrency_events.jsonl")
            self.assertEqual(attempts[0]["request_id"], "old")
            self.assertEqual(attempts[-1]["request_id"], "r1")
            self.assertEqual(events, [{"event": "old_event"}])

    def test_incremental_counts_match_full_recount_for_repeated_request_ids(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            write_jsonl(
                responses_path,
                [
                    {"request_id": "r1", "status": "success", "error_state": None},
                    {"request_id": "r2", "status": "error", "error_state": "429 rate limit"},
                    {"request_id": "r3", "status": "error", "error_state": "BadRequestError"},
                ],
            )
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            existing_state = read_response_run_state(responses_path)
            provider = SequenceProvider(
                [
                    _response(_request("r2"), 1, "success", None),
                    _response(_request("r1"), 2, "error", "429 rate limit"),
                    _response(_request("r3"), 3, "success", None),
                ]
            )

            rows, response_count, success_count, status = run_provider_requests(
                provider,
                [_request("r2"), _request("r1"), _request("r3")],
                responses_path,
                out_dir,
                "deepseek",
                _with_provider(config, max_retries=0),
                "resume",
                request_count=4,
                previous_terminal_count=existing_state.counts["terminal_count"],
                existing_response_count=existing_state.response_count,
                existing_state=existing_state,
            )

            all_rows = read_jsonl(responses_path)
            expected = response_terminal_counts(all_rows)
            progress = read_json(out_dir / "run_progress.json")
            self.assertEqual(len(rows), 3)
            self.assertEqual(response_count, len(all_rows))
            self.assertEqual(success_count, expected["success_count"])
            self.assertEqual(progress["success_count"], expected["success_count"])
            self.assertEqual(progress["retryable_failure_count"], expected["retryable_failure_count"])
            self.assertEqual(progress["permanent_error_count"], expected["permanent_error_count"])
            self.assertEqual(progress["terminal_count"], expected["terminal_count"])

    def test_attempt_and_event_logs_append_without_rewriting_history(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            attempts_path = out_dir / "attempts.jsonl"
            events_path = out_dir / "concurrency_events.jsonl"
            write_jsonl(attempts_path, [{"request_id": "old", "attempt_index": 1}])
            write_jsonl(events_path, [{"event": "old_event"}])
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

            run_provider_requests(
                FailingThenSuccessfulProvider(),
                [_request("r1"), _request("r2")],
                responses_path,
                out_dir,
                "deepseek",
                _with_provider(config, max_concurrency=2, concurrency_mode="adaptive"),
                "resume",
                request_count=2,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            attempts = read_jsonl(attempts_path)
            events = read_jsonl(events_path)
            self.assertEqual(attempts[0], {"request_id": "old", "attempt_index": 1})
            self.assertGreater(len(attempts), 1)
            self.assertEqual(events[0], {"event": "old_event"})
            self.assertTrue(any(row.get("event") == "adaptive_start" for row in events))

    def test_progress_is_throttled_but_completion_is_forced(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            out_dir = Path("run")
            responses_path = out_dir / "responses.jsonl"
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            provider = ProgressTimestampProvider(out_dir)

            run_provider_requests(
                provider,
                [_request(f"r{i}") for i in range(3)],
                responses_path,
                out_dir,
                "deepseek",
                config.provider,
                "run",
                request_count=3,
                previous_terminal_count=0,
                existing_response_count=0,
            )

            progress = read_json(out_dir / "run_progress.json")
            self.assertTrue(progress["complete"])
            self.assertEqual(progress["attempted_count"], 3)
            self.assertEqual(progress["terminal_count"], 3)
            self.assertEqual(provider.progress_attempted_counts, [0, 0, 0])

    def test_progress_interval_constant_is_five_seconds(self):
        self.assertEqual(PROGRESS_WRITE_INTERVAL_SECONDS, 5.0)


class RecordingProvider:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.seen_line_counts: list[int] = []

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        path = self.out_dir / "responses.jsonl"
        self.seen_line_counts.append(0 if not path.exists() else len(path.read_text(encoding="utf-8").splitlines()))
        return _response(request, attempt_index, "success", None)


class SlowProvider:
    def __init__(self, delay: float):
        self.delay = delay

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        time.sleep(self.delay)
        return _response(request, attempt_index, "success", None)


class FailingThenSuccessfulProvider:
    def __init__(self):
        self.count = 0

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        self.count += 1
        if self.count == 2:
            return _response(request, attempt_index, "error", "429 rate limit")
        return _response(request, attempt_index, "success", None)


class RetryableThenSuccessfulProvider:
    def __init__(self):
        self.count = 0

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        self.count += 1
        if self.count == 1:
            return _response(request, attempt_index, "error", "429 rate limit")
        return _response(request, attempt_index, "success", None)


class ActiveCountingProvider:
    def __init__(self, delay: float):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return _response(request, attempt_index, "success", None)
        finally:
            with self.lock:
                self.active -= 1


class PauseAfterFirstProvider:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.count = 0
        self.lock = threading.Lock()

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        with self.lock:
            self.count += 1
            if self.count == 1:
                (self.out_dir / PAUSE_REQUEST_FILE).write_text("pause requested\n", encoding="utf-8")
        time.sleep(0.05)
        return _response(request, attempt_index, "success", None)


class SequenceProvider:
    def __init__(self, responses: list[ProviderResponse]):
        self.responses = list(responses)

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        return self.responses.pop(0)


class ProgressTimestampProvider:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.progress_attempted_counts: list[int] = []

    def call(self, request: dict, attempt_index: int = 1) -> ProviderResponse:
        path = self.out_dir / "run_progress.json"
        self.progress_attempted_counts.append(read_json(path)["attempted_count"] if path.exists() else -1)
        return _response(request, attempt_index, "success", None)


def _request(request_id: str) -> dict:
    return {"request_id": request_id, "sample_id": f"s-{request_id}", "provider_payload": {"messages": []}}


def _response(request: dict, attempt_index: int, status: str, error_state: str | None) -> ProviderResponse:
    return ProviderResponse(
        request_id=request["request_id"],
        sample_id=request["sample_id"],
        attempt_id=f"attempt-{attempt_index}",
        status=status,
        provider="deepseek",
        model="deepseek-v4-flash",
        request_payload=request["provider_payload"],
        raw_response={"mock": True} if status == "success" else None,
        final_answer_text="True" if status == "success" else "",
        reasoning_text=None,
        answer_truncated=False,
        reasoning_truncated=False,
        reasoning_available=False,
        error_state=error_state,
        actual_request_parameters={"provider": "deepseek"},
    )


def _with_provider(config, **provider_values):
    return config.provider.__class__(**{**config.provider.__dict__, **provider_values})


if __name__ == "__main__":
    unittest.main()
