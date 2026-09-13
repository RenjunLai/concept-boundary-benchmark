import unittest
from pathlib import Path

from typer.testing import CliRunner

from bool_logic.cli import app
from bool_logic.config import load_config
from bool_logic.io_utils import read_json, write_json, write_jsonl
from bool_logic.preflight import preflight_run


REPO_ROOT = Path(__file__).resolve().parents[2]


class PreflightTests(unittest.TestCase):
    def test_preflight_accepts_current_test_dataset_and_mock_override(self):
        config = load_config(REPO_ROOT / "project/configs/experiments/test.toml")

        result = preflight_run(
            config,
            REPO_ROOT / "project/runs/test/requests.jsonl",
            REPO_ROOT / "project/runs/preflight_probe",
            "mock",
        )

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["dataset"]["request_count"], 1080)
        self.assertEqual(result["dataset"]["product_id"], "test")
        self.assertEqual(result["provider_run"]["provider"], "mock")
        self.assertEqual(result["provider_run"]["model"], "mock")
        self.assertEqual(result["provider_run"]["protocol"], "local_mock")
        self.assertEqual(result["provider_run"]["request_adapter"], "mock")
        self.assertFalse(result["provider_run"]["supports_streaming"])

    def test_preflight_rejects_real_provider_override_mismatch(self):
        config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

        result = preflight_run(
            config,
            REPO_ROOT / "project/runs/test/requests.jsonl",
            REPO_ROOT / "project/runs/preflight_probe",
            "zhipu",
        )

        self.assertFalse(result["passed"])
        self.assertTrue(any("provider override" in error for error in result["errors"]))

    def test_preflight_rejects_unknown_request_adapter(self):
        config = load_config(REPO_ROOT / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")
        bad_provider = config.provider.__class__(**{**config.provider.__dict__, "request_adapter": "unknown_adapter"})
        bad_config = config.__class__(
            name=config.name,
            tasks=config.tasks,
            matrix=config.matrix,
            resources=config.resources,
            budgets=config.budgets,
            product=config.product,
            provider=bad_provider,
            answer_parse_char_limit=config.answer_parse_char_limit,
        )

        result = preflight_run(
            bad_config,
            REPO_ROOT / "project/runs/test/requests.jsonl",
            REPO_ROOT / "project/runs/preflight_probe",
            "deepseek",
        )

        self.assertFalse(result["passed"])
        self.assertTrue(any("request_adapter" in error for error in result["errors"]))

    def test_preflight_rejects_provider_in_dataset_snapshot(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            source_dir = Path("source")
            source_dir.mkdir()
            write_json(source_dir / "dataset_config_snapshot.json", {"provider": {"name": "zhipu"}})
            write_json(source_dir / "sample_manifest.json", {"product_id": "test", "product_tier": "test"})
            write_json(source_dir / "render_manifest.json", {"request_count": 1, "renderer_version": "separated_gloss_v2"})
            write_jsonl(source_dir / "requests.jsonl", [_request("abcd1234abcd1234abcd1234abcd1234")])
            config = load_config(REPO_ROOT / "project/configs/experiments/test.toml")

            result = preflight_run(config, source_dir / "requests.jsonl", Path("run"), "mock")

        self.assertFalse(result["passed"])
        self.assertTrue(any("dataset_config_snapshot" in error for error in result["errors"]))

    def test_preflight_accepts_sharded_request_directory(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            source_dir = Path("source")
            requests_dir = source_dir / "requests"
            requests_dir.mkdir(parents=True)
            write_json(source_dir / "dataset_config_snapshot.json", {"name": "full"})
            write_json(source_dir / "sample_manifest.json", {"product_id": "full", "product_tier": "full"})
            write_json(source_dir / "render_manifest.json", {"request_count": 2, "renderer_version": "separated_gloss_v2"})
            write_jsonl(requests_dir / "part-00000.jsonl", [_request("abcd1234abcd1234abcd1234abcd1234")])
            write_jsonl(requests_dir / "part-00001.jsonl", [_request("efef5678efef5678efef5678efef5678")])
            config = load_config(REPO_ROOT / "project/configs/experiments/full.toml")

            result = preflight_run(config, requests_dir, Path("run"), "mock")

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["dataset"]["request_count"], 2)
        self.assertEqual(result["dataset"]["product_tier"], "full")

    def test_cli_preflight_run_is_read_only_and_returns_success(self):
        runner = CliRunner()
        out_dir = REPO_ROOT / "project/runs/preflight_cli_probe"
        result = runner.invoke(
            app,
            [
                "preflight-run",
                "--config",
                str(REPO_ROOT / "project/configs/experiments/test.toml"),
                "--requests",
                str(REPO_ROOT / "project/runs/test/requests.jsonl"),
                "--out",
                str(out_dir),
                "--provider",
                "mock",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse((out_dir / "run_manifest.json").exists())


def _request(request_hash: str) -> dict:
    request_id = f"request_{request_hash[:16]}"
    return {
        "request_id": request_id,
        "sample_id": "sample1",
        "base_sample_id": "base1",
        "task": "T1",
        "dataset_family": "wordnet",
        "information": "I0",
        "renderer_version": "separated_gloss_v2",
        "request_hash": request_hash,
        "provider_payload": {"messages": [{"role": "user", "content": "x"}]},
        "sample": {"sample_id": "sample1", "request_hash": request_hash},
    }


if __name__ == "__main__":
    unittest.main()
