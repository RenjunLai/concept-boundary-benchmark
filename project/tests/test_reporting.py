import json
import tempfile
import unittest
from pathlib import Path

from bool_logic.pipeline import run_report


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


class ReportingTests(unittest.TestCase):
    def test_report_writes_evaluation_json_and_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            sample = {
                "base_sample_id": "b1",
                "sample_id": "s1",
                "sample_hash": "sh1",
                "request_hash": "rh1",
                "product_id": "test",
                "dataset_family": "wordnet",
                "task": "T1",
                "information": "I0",
                "constraint_cell": [1, 0],
                "a_count": 1,
                "b_count": 0,
                "seed_id": "x1",
                "A1_id": "a1",
                "A_branch_ids": [],
                "B_ids": [],
                "paired_A_ids": [],
                "B_sources": [],
                "gold_answer": True,
                "candidate_ids": ["x1"],
                "metadata": {"final_positive_count": 4, "negative_count": 4, "hard_negative_count": 0},
            }
            write_jsonl(run_dir / "requests.jsonl", [{"request_id": "r1", "sample": sample}])
            write_jsonl(run_dir / "responses.jsonl", [{"request_id": "r1", "status": "success", "final_answer_text": "True"}])
            write_json(run_dir / "sample_manifest.json", {"base_sample_count": 1, "sample_count": 1, "reuse_cap_unit": "base_sample", "paired_information_levels": ["I0"]})
            write_json(run_dir / "run_manifest.json", {"provider": "mock", "model": "mock", "complete": True, "request_count": 1, "response_count": 1, "success_count": 1})

            result = run_report(run_dir)

            self.assertTrue((run_dir / "evaluation_rows.jsonl").exists())
            self.assertTrue((run_dir / "evaluation_metrics.json").exists())
            self.assertTrue((run_dir / "evaluation_report.ipynb").exists())
            self.assertFalse((run_dir / "analysis_report.md").exists())
            self.assertIn("evaluation_report", result)
            metrics = json.loads((run_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["overall"]["accuracy"], 1.0)
            self.assertEqual(metrics["overall"]["selected_response_count"], 1)
            self.assertEqual(metrics["task_accuracy"][0]["task"], "T1")
            self.assertIn("overall_accuracy_matrix", metrics)
            notebook_text = (run_dir / "evaluation_report.ipynb").read_text(encoding="utf-8")
            self.assertIn("evaluation_metrics.json", notebook_text)


if __name__ == "__main__":
    unittest.main()
