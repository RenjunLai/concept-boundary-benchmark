import time
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from bool_logic.cli import app
from bool_logic.config import load_config
from bool_logic.io_utils import read_json, write_json, write_jsonl
from bool_logic.pipeline import run_reparse, run_report, run_requests, run_requests_with_mode
from bool_logic.schemas import ProviderResponse


REPO_ROOT = Path(__file__).resolve().parents[2]


class CliProviderSelectionTests(unittest.TestCase):
    def test_run_uses_provider_from_config_when_not_overridden(self):
        runner = CliRunner()
        config = Path("project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
        requests = Path("project/runs/test/requests.jsonl")

        with patch("bool_logic.cli.run_requests", return_value={"ok": True}) as run_requests:
            result = runner.invoke(
                app,
                [
                    "run",
                    "--config",
                    str(config),
                    "--requests",
                    str(requests),
                    "--out",
                    "project/runs/deepseek_cli_probe",
                    "--limit",
                    "1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run_requests.call_args.args[3], "deepseek")

    def test_run_provider_override_is_still_respected(self):
        runner = CliRunner()
        config = Path("project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
        requests = Path("project/runs/test/requests.jsonl")

        with patch("bool_logic.cli.run_requests", return_value={"ok": True}) as run_requests:
            result = runner.invoke(
                app,
                [
                    "run",
                    "--config",
                    str(config),
                    "--requests",
                    str(requests),
                    "--out",
                    "project/runs/deepseek_cli_probe",
                    "--provider",
                    "mock",
                    "--limit",
                    "1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run_requests.call_args.args[3], "mock")

    def test_real_provider_override_must_match_config(self):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml",
                "--requests",
                "project/runs/test/requests.jsonl",
                "--out",
                "project/runs/provider_mismatch_probe",
                "--provider",
                "zhipu",
                "--limit",
                "1",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("provider override must match", result.output)

    def test_run_separates_dataset_and_provider_config_snapshots(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            run_dir = Path("run")
            source_dir = Path("source")
            source_dir.mkdir()
            write_json(source_dir / "config_snapshot.json", {"provider": {"name": "zhipu"}})
            write_json(run_dir / "dataset_config_snapshot.json", {"provider": {"name": "old"}})
            write_json(source_dir / "sample_manifest.json", {})
            request = {
                "request_id": "r1",
                "sample_id": "s1",
                "provider_payload": {"messages": []},
                "sample": {
                    "sample_id": "s1",
                    "task": "T1",
                    "dataset_family": "wordnet",
                    "constraint_cell": [1, 0],
                    "a_count": 1,
                    "b_count": 0,
                    "information": "I0",
                    "candidate_ids": ["x1"],
                    "gold_answer": True,
                },
            }
            write_jsonl(source_dir / "requests.jsonl", [request])
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

            run_requests(config, source_dir / "requests.jsonl", run_dir, "mock", limit=1)

            self.assertNotIn("provider", read_json(run_dir / "dataset_config_snapshot.json"))
            self.assertEqual(read_json(run_dir / "config_snapshot.json")["provider"]["name"], "deepseek")
            manifest = read_json(run_dir / "run_manifest.json")
            self.assertIn("dataset_config_snapshot.json", manifest["context_artifacts"])
            self.assertEqual(manifest["provider"], "mock")
            self.assertEqual(manifest["model"], "mock")
            self.assertEqual(manifest["provider_capability"]["protocol"], "local_mock")
            self.assertEqual(manifest["provider_capability"]["request_adapter"], "mock")
            self.assertFalse(manifest["provider_capability"]["supports_streaming"])

    def test_run_writes_each_response_and_progress_incrementally(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            run_dir = Path("run")
            source_dir = Path("source")
            source_dir.mkdir()
            requests = [_mock_request("r1", True), _mock_request("r2", False)]
            write_jsonl(source_dir / "requests.jsonl", requests)
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            seen_counts: list[int] = []

            def fake_call(request, attempt_index=1):
                response_path = run_dir / "responses.jsonl"
                seen_counts.append(0 if not response_path.exists() else len(response_path.read_text().splitlines()))
                return _provider_response(request, attempt_index, "True")

            with patch("bool_logic.pipeline.get_provider") as get_provider:
                get_provider.return_value.call.side_effect = fake_call
                run_requests(config, source_dir / "requests.jsonl", run_dir, "deepseek")

            self.assertEqual(seen_counts, [0, 1])
            self.assertEqual(len((run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            progress = read_json(run_dir / "run_progress.json")
            self.assertEqual(progress["attempted_count"], 2)
            self.assertEqual(progress["response_count"], 2)
            self.assertEqual(progress["terminal_count"], 2)
            self.assertTrue(progress["complete"])

    def test_fixed_concurrency_mode_is_run_specific(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            run_dir = Path("run")
            source_dir = Path("source")
            source_dir.mkdir()
            write_jsonl(source_dir / "requests.jsonl", [_mock_request("r1", True), _mock_request("r2", False)])
            base_config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
            config = base_config.__class__(
                name=base_config.name,
                tasks=base_config.tasks,
                matrix=base_config.matrix,
                resources=base_config.resources,
                budgets=base_config.budgets,
                product=base_config.product,
                provider=base_config.provider.__class__(
                    **{**base_config.provider.__dict__, "max_concurrency": 2, "concurrency_mode": "fixed"}
                ),
                answer_parse_char_limit=base_config.answer_parse_char_limit,
            )

            def fake_call(request, attempt_index=1):
                time.sleep(0.2)
                return _provider_response(request, attempt_index, "True")

            start = time.perf_counter()
            with patch("bool_logic.pipeline.get_provider") as get_provider:
                get_provider.return_value.call.side_effect = fake_call
                run_requests_with_mode(config, source_dir / "requests.jsonl", run_dir, "deepseek", "run")
            elapsed = time.perf_counter() - start

            manifest = read_json(run_dir / "run_manifest.json")
            self.assertEqual(manifest["thinking"], "off")
            self.assertEqual(manifest["concurrency_mode"], "fixed")
            self.assertEqual(manifest["max_concurrency"], 2)
            self.assertEqual(manifest["provider_capability"]["request_adapter"], "deepseek_reasoning_effort")
            self.assertEqual(manifest["provider_capability"]["recommended_max_concurrency"], 10)
            self.assertLess(elapsed, 0.35)

    def test_resume_preserves_original_source_requests_path(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            run_dir = Path("run")
            source_dir = Path("source")
            source_dir.mkdir()
            requests = [_mock_request("r1", True), _mock_request("r2", False)]
            source_requests = source_dir / "requests.jsonl"
            write_jsonl(source_requests, requests)
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

            with patch("bool_logic.pipeline.get_provider") as get_provider:
                get_provider.return_value.call.side_effect = lambda request, attempt_index=1: _provider_response(
                    request, attempt_index, "True"
                )
                run_requests(config, source_requests, run_dir, "deepseek", limit=1)
                run_requests_with_mode(config, run_dir / "requests.jsonl", run_dir, "deepseek", "resume", limit=1)

            manifest = read_json(run_dir / "run_manifest.json")
            self.assertEqual(manifest["source_requests_path"], str(source_requests))
            self.assertEqual(manifest["original_source_requests_path"], str(source_requests))

    def test_resume_does_not_retry_permanent_failure(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            run_dir = Path("run")
            run_dir.mkdir()
            requests = [_mock_request("r1", True)]
            write_jsonl(run_dir / "requests.jsonl", requests)
            write_jsonl(
                run_dir / "responses.jsonl",
                [
                    {
                        "request_id": "r1",
                        "sample_id": "s-r1",
                        "status": "error",
                        "error_state": "BadRequestError: unsupported parameter",
                    }
                ],
            )
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

            with patch("bool_logic.pipeline.get_provider") as get_provider:
                result = run_requests_with_mode(config, run_dir / "requests.jsonl", run_dir, "deepseek", "resume")

            self.assertEqual(result["attempted_count"], 0)
            get_provider.return_value.call.assert_not_called()
            manifest = read_json(run_dir / "run_manifest.json")
            self.assertTrue(manifest["complete"])
            progress = read_json(run_dir / "run_progress.json")
            self.assertTrue(progress["complete"])
            self.assertEqual(progress["pending_count"], 0)
            self.assertEqual(progress["terminal_count"], 1)
            self.assertEqual(manifest["permanent_error_count"], 1)

    def test_reparse_and_report_do_not_require_legacy_nvidia_provider_registry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            run_dir = Path("run")
            run_dir.mkdir()
            request = _mock_request("r1", True)
            write_jsonl(run_dir / "requests.jsonl", [request])
            write_jsonl(
                run_dir / "responses.jsonl",
                [
                    {
                        "request_id": "r1",
                        "sample_id": "s-r1",
                        "attempt_id": "attempt-old",
                        "status": "error",
                        "provider": "nvidia_deepseek",
                        "model": "deepseek-ai/deepseek-v4-pro",
                        "request_payload": {},
                        "raw_response": None,
                        "final_answer_text": "",
                        "reasoning_text": None,
                        "answer_truncated": False,
                        "reasoning_truncated": False,
                        "reasoning_available": False,
                        "error_state": "429 rate limit",
                        "actual_request_parameters": {"provider": "nvidia_deepseek"},
                    },
                    {
                        "request_id": "r1",
                        "sample_id": "s-r1",
                        "attempt_id": "attempt-new",
                        "status": "success",
                        "provider": "nvidia",
                        "model": "openai/gpt-oss-120b",
                        "request_payload": {},
                        "raw_response": {"mock": True},
                        "final_answer_text": "True",
                        "reasoning_text": "reasoning",
                        "answer_truncated": False,
                        "reasoning_truncated": False,
                        "reasoning_available": True,
                        "error_state": None,
                        "actual_request_parameters": {
                            "provider": "nvidia",
                            "request_adapter": "openai_reasoning_effort",
                        },
                    },
                ],
            )
            write_json(run_dir / "sample_manifest.json", {"base_sample_count": 1, "sample_count": 1})
            write_json(run_dir / "run_manifest.json", {"provider": "nvidia_deepseek", "model": "legacy", "complete": True, "request_count": 1})
            config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

            with patch("bool_logic.pipeline.get_provider") as get_provider:
                run_reparse(config, run_dir)
                run_report(run_dir)

            get_provider.assert_not_called()
            metrics = read_json(run_dir / "evaluation_metrics.json")
            self.assertEqual(metrics["overall"]["selected_response_count"], 1)
            self.assertEqual(metrics["overall"]["correct"], 1)


def _mock_request(request_id: str, gold_answer: bool) -> dict:
    return {
        "request_id": request_id,
        "sample_id": f"s-{request_id}",
        "provider_payload": {"messages": []},
        "sample": {
            "base_sample_id": f"b-{request_id}",
            "sample_id": f"s-{request_id}",
            "task": "T1",
            "dataset_family": "wordnet",
            "constraint_cell": [1, 0],
            "a_count": 1,
            "b_count": 0,
            "information": "I0",
            "candidate_ids": ["x1"],
            "gold_answer": gold_answer,
        },
    }


def _provider_response(request: dict, attempt_index: int, answer: str) -> ProviderResponse:
    return ProviderResponse(
        request_id=request["request_id"],
        sample_id=request["sample_id"],
        attempt_id=f"attempt-{attempt_index}",
        status="success",
        provider="deepseek",
        model="deepseek-v4-flash",
        request_payload=request["provider_payload"],
        raw_response={"mock": True},
        final_answer_text=answer,
        reasoning_text=None,
        answer_truncated=False,
        reasoning_truncated=False,
        reasoning_available=False,
        error_state=None,
        actual_request_parameters={"provider": "deepseek"},
    )


if __name__ == "__main__":
    unittest.main()
