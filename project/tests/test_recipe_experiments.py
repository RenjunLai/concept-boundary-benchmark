import json
import tempfile
import unittest
from pathlib import Path

from bool_logic.recipe_candidate_scheduler_probe import _quantiles
from bool_logic.recipe_b_depth_audit import _b_options_by_depth
from bool_logic.recipe_scheduler_audit import build_recipe_scheduler_audit
from bool_logic.resources import Inventory
from bool_logic.schemas import ResourceObject
from bool_logic.taxonomy_bitsets import BitsetHelper


def obj(object_id, surface, children=(), parents=()):
    return ResourceObject(
        dataset_family="wordnet",
        object_id=object_id,
        surface=surface,
        gloss=f"{surface} gloss",
        synonyms=(surface,),
        children=tuple(children),
        parents=tuple(parents),
        usable_flags={"noun": True, "clean_surface": True, "has_gloss": True},
        metadata={},
    )


class RecipeExperimentTests(unittest.TestCase):
    def test_b_depth_expands_candidates_without_changing_direct_boundary(self):
        inventory = Inventory(
            "wordnet",
            {
                "a1": obj("a1", "a1", children=("branch", "direct_b")),
                "branch": obj("branch", "branch", children=("seed", "branch_b"), parents=("a1",)),
                "seed": obj("seed", "seed", parents=("branch",)),
                "branch_b": obj("branch_b", "branch_b", children=("deep_b",), parents=("branch",)),
                "deep_b": obj("deep_b", "deep_b", children=("deep_leaf",), parents=("branch_b",)),
                "deep_leaf": obj("deep_leaf", "deep_leaf", parents=("deep_b",)),
                "direct_b": obj("direct_b", "direct_b", children=("direct_leaf",), parents=("a1",)),
                "direct_leaf": obj("direct_leaf", "direct_leaf", parents=("direct_b",)),
            },
        )
        bitsets = BitsetHelper(inventory)

        direct = _b_options_by_depth(bitsets, "a1", ("branch",), "seed", max_depth=1)
        deep = _b_options_by_depth(bitsets, "a1", ("branch",), "seed", max_depth=None)

        self.assertEqual({row["B_id"] for row in direct}, {"branch_b", "direct_b"})
        self.assertEqual({row["B_id"] for row in deep}, {"branch_b", "deep_b", "direct_b"})

    def test_scheduler_balanced_scenario_has_no_shortage_on_synthetic_capacity(self):
        payload = {
            "cell_rows": [
                {
                    "dataset_family": "wordnet",
                    "constraint_cell": [1, 0],
                    "counts_by_min_final_positive": {
                        "5": {"T1": 700, "T2_any": 700, "T2_all_gold_sizes": 700, "T3": 700}
                    },
                },
                {
                    "dataset_family": "babelnet",
                    "constraint_cell": [1, 0],
                    "counts_by_min_final_positive": {
                        "5": {"T1": 120, "T2_any": 120, "T2_all_gold_sizes": 120, "T3": 120}
                    },
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            threshold_path = root / "threshold.json"
            threshold_path.write_text(json.dumps(payload), encoding="utf-8")
            summary = build_recipe_scheduler_audit(threshold_path, root / "out", threshold=5)

        recommended = next(row for row in summary["scenarios"] if row["scenario_id"] == "A_balanced_simple")
        self.assertEqual(recommended["shortage_count"], 0)
        self.assertEqual(summary["recommendation"]["scenario_id"], "A_balanced_simple")

    def test_scheduler_probe_quantiles_are_stable(self):
        self.assertEqual(_quantiles([3, 1, 2])["p50"], 2)
        self.assertEqual(_quantiles([])["count"], 0)

    def test_scheduler_probe_writes_all_task_views_when_available(self):
        from bool_logic.recipe_candidate_scheduler_probe import _record_sample
        from collections import Counter

        state = {
            "base_samples": 0,
            "task_counts": Counter(),
            "answer_counts": Counter(),
            "cell_answer_counts": Counter(),
            "candidate_usage": Counter(),
            "pool_usage": Counter(),
            "negative_sources": Counter(),
            "fallback_count": 0,
        }
        for task in ("T1", "T2", "T3"):
            _record_sample(state, task, "wordnet", (1, 0), ("x", "y") if task != "T1" else ("x",), "true", Counter(), False)
        self.assertEqual(state["task_counts"], Counter({"T1": 1, "T2": 1, "T3": 1}))

    def test_scheduler_probe_can_record_t1_false(self):
        from bool_logic.recipe_candidate_scheduler_probe import _record_sample
        from collections import Counter

        state = {
            "base_samples": 0,
            "task_counts": Counter(),
            "answer_counts": Counter(),
            "cell_answer_counts": Counter(),
            "candidate_usage": Counter(),
            "pool_usage": Counter(),
            "negative_sources": Counter(),
            "fallback_count": 0,
        }
        _record_sample(state, "T1", "wordnet", (1, 0), ("n",), "false", Counter({"hard": 1}), False)
        self.assertEqual(state["answer_counts"][("T1", "false")], 1)
        self.assertEqual(state["negative_sources"]["hard"], 1)


if __name__ == "__main__":
    unittest.main()
