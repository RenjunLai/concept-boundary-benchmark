import unittest

from bool_logic.audit import FeasibilityRow, build_feasibility
from bool_logic.config import BudgetConfig, ExperimentConfig, MatrixConfig, ProductConfig, ResourceConfig
from bool_logic.generator import generate_frozen_samples
from bool_logic.generator import _select_branches
from bool_logic.pipeline import _finalize_full_stream_validation, _update_full_stream_validation
from bool_logic.product_scale import stratified_product_size_target
from bool_logic.taxonomy_bitsets import BitsetHelper
from bool_logic.resources import Inventory, clean_zh_synonyms, preferred_chinese_surface
from bool_logic.schemas import ResourceObject


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


class GenerationContractTests(unittest.TestCase):
    def test_config_matrix_is_complete(self):
        config = ExperimentConfig()
        self.assertEqual(config.matrix.a_count, (1, 2, 3, 4, 5))
        self.assertEqual(config.matrix.b_count, (0, 1, 2, 3, 4, 5))
        self.assertEqual(config.matrix.information, ("I0", "I1", "I2"))

    def test_babelnet_mi_surface_filter(self):
        item = {"bn": "bn:00054288n", "zh_synonyms": ["弥", "迷因", "模因"]}
        self.assertEqual(preferred_chinese_surface(item), "迷因")
        self.assertNotIn("弥", clean_zh_synonyms(item))

    def test_generation_pairs_information_levels_with_seed_fields(self):
        inventory = Inventory(
            "wordnet",
            {
                "root": obj("root", "root", children=("positive", "excluded", "outside")),
                "positive": obj("positive", "positive", children=("seed", "p1", "p2", "p3", "p4", "b1"), parents=("root",)),
                "excluded": obj("excluded", "excluded", children=("n1", "n2", "n3", "n4"), parents=("root",)),
                "outside": obj("outside", "outside", children=("o1", "o2", "o3", "o4"), parents=("root",)),
                "seed": obj("seed", "seed", parents=("positive",)),
                "p1": obj("p1", "p1", parents=("positive",)),
                "p2": obj("p2", "p2", parents=("positive",)),
                "p3": obj("p3", "p3", parents=("positive",)),
                "p4": obj("p4", "p4", parents=("positive",)),
                "b1": obj("b1", "b1", children=("b_leaf1", "b_leaf2", "b_leaf3", "b_leaf4"), parents=("positive",)),
                "b_leaf1": obj("b_leaf1", "b_leaf1", parents=("b1",)),
                "b_leaf2": obj("b_leaf2", "b_leaf2", parents=("b1",)),
                "b_leaf3": obj("b_leaf3", "b_leaf3", parents=("b1",)),
                "b_leaf4": obj("b_leaf4", "b_leaf4", parents=("b1",)),
                "n1": obj("n1", "n1", parents=("excluded",)),
                "n2": obj("n2", "n2", parents=("excluded",)),
                "n3": obj("n3", "n3", parents=("excluded",)),
                "n4": obj("n4", "n4", parents=("excluded",)),
                "o1": obj("o1", "o1", parents=("outside",)),
                "o2": obj("o2", "o2", parents=("outside",)),
                "o3": obj("o3", "o3", parents=("outside",)),
                "o4": obj("o4", "o4", parents=("outside",)),
            },
        )
        row = FeasibilityRow(
            dataset_family="wordnet",
            task="T2",
            constraint_cell=(1, 1),
            a_count=1,
            b_count=1,
            information_policy="paired_I0_I1_I2_per_base_sample_id",
            closure_size_distribution={},
            usable_seed_count=1,
            feasible_seed_cell_count=1,
            usable_candidate_count=8,
            direct_usable_child_count=3,
            B_sources=("inside_A1",),
            available_primary_negative_source=("hits_B", "outside_A1"),
            hard_negative_available=True,
            easy_negative_available=True,
            t1_capacity=8,
            t2_pool_space_estimate=1,
            t3_pool_space_estimate=1,
            status="available",
            reason="ok",
        )
        config = ExperimentConfig(
            tasks=("T2",),
            matrix=MatrixConfig(a_count=(1,), b_count=(1,), information=("I0", "I1", "I2")),
            resources=ResourceConfig(dataset_families=("wordnet",), min_final_positive_count=1, max_seed_per_family=20),
            budgets=BudgetConfig(max_seed_reuse=0, max_a1_reuse=0, max_candidate_reuse=0, max_constraint_reuse=0, max_pool_reuse=0),
            product=ProductConfig(product_id="test", tier="test", wordnet_quota_per_stratum=1, babelnet_quota_per_stratum=0),
        )
        samples, manifest = generate_frozen_samples(config, {"wordnet": inventory}, [row])
        self.assertEqual(manifest["product_id"], "test")
        self.assertEqual(manifest["sampling_policy"]["type"], "post_gate_stratified_without_replacement")
        self.assertEqual(manifest["base_sample_count"], 1)
        self.assertEqual(sorted(sample.information for sample in samples), ["I0", "I1", "I2"])
        self.assertEqual(len({sample.base_sample_id for sample in samples}), 1)
        self.assertEqual(len({sample.sample_hash for sample in samples}), 3)
        self.assertTrue(all(sample.seed_id for sample in samples))
        self.assertTrue(all(sample.A1_id for sample in samples))

    def test_standard_product_uses_post_gate_weights_and_product_id(self):
        inventory = Inventory(
            "wordnet",
            {
                "root": obj("root", "root", children=("positive", "excluded", "outside")),
                "positive": obj("positive", "positive", children=("seed", "seed2", "p1", "p2", "p3", "p4", "b1"), parents=("root",)),
                "excluded": obj("excluded", "excluded", children=("n1", "n2", "n3", "n4"), parents=("root",)),
                "outside": obj("outside", "outside", children=("o1", "o2", "o3", "o4"), parents=("root",)),
                "seed": obj("seed", "seed", parents=("positive",)),
                "seed2": obj("seed2", "seed2", parents=("positive",)),
                "p1": obj("p1", "p1", parents=("positive",)),
                "p2": obj("p2", "p2", parents=("positive",)),
                "p3": obj("p3", "p3", parents=("positive",)),
                "p4": obj("p4", "p4", parents=("positive",)),
                "b1": obj("b1", "b1", children=("b_leaf1", "b_leaf2", "b_leaf3", "b_leaf4"), parents=("positive",)),
                "b_leaf1": obj("b_leaf1", "b_leaf1", parents=("b1",)),
                "b_leaf2": obj("b_leaf2", "b_leaf2", parents=("b1",)),
                "b_leaf3": obj("b_leaf3", "b_leaf3", parents=("b1",)),
                "b_leaf4": obj("b_leaf4", "b_leaf4", parents=("b1",)),
                "n1": obj("n1", "n1", parents=("excluded",)),
                "n2": obj("n2", "n2", parents=("excluded",)),
                "n3": obj("n3", "n3", parents=("excluded",)),
                "n4": obj("n4", "n4", parents=("excluded",)),
                "o1": obj("o1", "o1", parents=("outside",)),
                "o2": obj("o2", "o2", parents=("outside",)),
                "o3": obj("o3", "o3", parents=("outside",)),
                "o4": obj("o4", "o4", parents=("outside",)),
            },
        )
        config = ExperimentConfig(
            tasks=("T1",),
            matrix=MatrixConfig(a_count=(1,), b_count=(1,), information=("I0", "I1", "I2")),
            resources=ResourceConfig(dataset_families=("wordnet",), min_final_positive_count=1, max_seed_per_family=20),
            budgets=BudgetConfig(max_seed_reuse=0, max_a1_reuse=0, max_candidate_reuse=0, max_constraint_reuse=0, max_pool_reuse=0),
            product=ProductConfig(product_id="main", tier="main", wordnet_quota_per_stratum=1),
        )
        samples, manifest = generate_frozen_samples(config, {"wordnet": inventory}, [])
        self.assertEqual(manifest["product_id"], "main")
        self.assertEqual(manifest["sampling_policy"]["weight_population"], "post_gate_eligible_population")
        self.assertEqual(manifest["base_sample_count"], 1)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample.product_id == "main" for sample in samples))
        stratum = manifest["strata"][0]
        self.assertEqual(stratum["post_gate_population"], 20)
        self.assertTrue(all(sample.inclusion_probability == 0.05 for sample in samples))
        self.assertTrue(all(sample.sampling_weight == 20.0 for sample in samples))

    def test_test_product_uses_precise_small_stratified_quota(self):
        config = ExperimentConfig(product=ProductConfig())
        self.assertEqual(config.product.product_id, "test")
        self.assertEqual(config.product.tier, "test")
        self.assertEqual(config.product.wordnet_quota_per_stratum, 3)
        self.assertEqual(config.product.babelnet_quota_per_stratum, 1)
        target = stratified_product_size_target(config)
        self.assertEqual(target["family_quotas"], {"wordnet": 3, "babelnet": 1})
        self.assertTrue(target["dataset_family_count_is_not_multiplier"])
        self.assertEqual(target["target_base_samples"], 360)
        self.assertEqual(target["target_rendered_requests"], 1080)

    def test_main_product_size_uses_family_quota_sum_not_family_count(self):
        config = ExperimentConfig(
            product=ProductConfig(product_id="main", tier="main", wordnet_quota_per_stratum=650, babelnet_quota_per_stratum=100)
        )
        target = stratified_product_size_target(config)
        self.assertEqual(target["family_quota_sum_per_task_cell"], 750)
        self.assertEqual(target["target_base_samples"], 67500)
        self.assertEqual(target["target_rendered_requests"], 202500)

    def test_branch_selection_keeps_seed_branch_for_single_branch_constraint(self):
        inventory = Inventory(
            "wordnet",
            {
                "a1": obj("a1", "a1", children=("seed_branch", "other_branch")),
                "seed_branch": obj("seed_branch", "seed_branch", children=("seed",), parents=("a1",)),
                "other_branch": obj("other_branch", "other_branch", children=("other",), parents=("a1",)),
                "seed": obj("seed", "seed", parents=("seed_branch",)),
                "other": obj("other", "other", parents=("other_branch",)),
            },
        )
        bitsets = BitsetHelper(inventory)
        self.assertEqual(_select_branches(bitsets, "a1", "seed", 1), ("seed_branch",))

    def test_feasibility_population_applies_positive_pool_gate(self):
        inventory = Inventory(
            "wordnet",
            {
                "root": obj("root", "root", children=("wide", "small", "outside")),
                "wide": obj("wide", "wide", children=("seed1", "p1", "p2", "p3", "p4", "p5"), parents=("root",)),
                "small": obj("small", "small", children=("seed2",), parents=("root",)),
                "outside": obj("outside", "outside", children=("o1", "o2", "o3", "o4"), parents=("root",)),
                "seed1": obj("seed1", "seed1", parents=("wide",)),
                "p1": obj("p1", "p1", parents=("wide",)),
                "p2": obj("p2", "p2", parents=("wide",)),
                "p3": obj("p3", "p3", parents=("wide",)),
                "p4": obj("p4", "p4", parents=("wide",)),
                "p5": obj("p5", "p5", parents=("wide",)),
                "seed2": obj("seed2", "seed2", parents=("small",)),
                "o1": obj("o1", "o1", parents=("outside",)),
                "o2": obj("o2", "o2", parents=("outside",)),
                "o3": obj("o3", "o3", parents=("outside",)),
                "o4": obj("o4", "o4", parents=("outside",)),
            },
        )
        config = ExperimentConfig(
            tasks=("T1",),
            matrix=MatrixConfig(a_count=(1,), b_count=(0,), information=("I0", "I1", "I2")),
            resources=ResourceConfig(dataset_families=("wordnet",), min_final_positive_count=5, max_seed_per_family=20),
        )

        rows = build_feasibility(config, {"wordnet": inventory})

        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0].feasible_seed_cell_count, len(inventory.objects))
        self.assertGreaterEqual(rows[0].final_positive_min, 5)

    def test_full_stream_validation_accepts_paired_samples(self):
        inventory = Inventory(
            "wordnet",
            {
                "root": obj("root", "root", children=("positive", "excluded", "outside")),
                "positive": obj("positive", "positive", children=("seed", "p1", "p2", "p3", "p4", "b1"), parents=("root",)),
                "excluded": obj("excluded", "excluded", children=("n1", "n2", "n3", "n4"), parents=("root",)),
                "outside": obj("outside", "outside", children=("o1", "o2", "o3", "o4"), parents=("root",)),
                "seed": obj("seed", "seed", parents=("positive",)),
                "p1": obj("p1", "p1", parents=("positive",)),
                "p2": obj("p2", "p2", parents=("positive",)),
                "p3": obj("p3", "p3", parents=("positive",)),
                "p4": obj("p4", "p4", parents=("positive",)),
                "b1": obj("b1", "b1", children=("b_leaf1", "b_leaf2", "b_leaf3", "b_leaf4"), parents=("positive",)),
                "b_leaf1": obj("b_leaf1", "b_leaf1", parents=("b1",)),
                "b_leaf2": obj("b_leaf2", "b_leaf2", parents=("b1",)),
                "b_leaf3": obj("b_leaf3", "b_leaf3", parents=("b1",)),
                "b_leaf4": obj("b_leaf4", "b_leaf4", parents=("b1",)),
                "n1": obj("n1", "n1", parents=("excluded",)),
                "n2": obj("n2", "n2", parents=("excluded",)),
                "n3": obj("n3", "n3", parents=("excluded",)),
                "n4": obj("n4", "n4", parents=("excluded",)),
                "o1": obj("o1", "o1", parents=("outside",)),
                "o2": obj("o2", "o2", parents=("outside",)),
                "o3": obj("o3", "o3", parents=("outside",)),
                "o4": obj("o4", "o4", parents=("outside",)),
            },
        )
        row = FeasibilityRow(
            dataset_family="wordnet",
            task="T2",
            constraint_cell=(1, 1),
            a_count=1,
            b_count=1,
            information_policy="paired_I0_I1_I2_per_base_sample_id",
            closure_size_distribution={},
            usable_seed_count=1,
            feasible_seed_cell_count=1,
            usable_candidate_count=8,
            direct_usable_child_count=3,
            B_sources=("inside_A1",),
            available_primary_negative_source=("hits_B", "outside_A1"),
            hard_negative_available=True,
            easy_negative_available=True,
            t1_capacity=8,
            t2_pool_space_estimate=1,
            t3_pool_space_estimate=1,
            status="available",
            reason="ok",
        )
        config = ExperimentConfig(
            tasks=("T2",),
            matrix=MatrixConfig(a_count=(1,), b_count=(1,), information=("I0", "I1", "I2")),
            resources=ResourceConfig(dataset_families=("wordnet",), min_final_positive_count=1, max_seed_per_family=20),
            budgets=BudgetConfig(max_seed_reuse=0, max_a1_reuse=0, max_candidate_reuse=0, max_constraint_reuse=0, max_pool_reuse=0),
            product=ProductConfig(product_id="test", tier="test", wordnet_quota_per_stratum=1, babelnet_quota_per_stratum=0),
        )
        samples, _ = generate_frozen_samples(config, {"wordnet": inventory}, [row])
        validation = {
            "sample_hashes": set(),
            "request_hashes": set(),
            "sample_ids": set(),
            "base_samples": {},
            "duplicate_sample_hashes": 0,
            "duplicate_request_hashes": 0,
            "duplicate_sample_ids": 0,
            "oracle_mismatch_count": 0,
        }

        for sample in samples:
            _update_full_stream_validation(validation, sample, {"wordnet": inventory})
        result = _finalize_full_stream_validation(validation)

        self.assertTrue(result["passed"])
        self.assertEqual(result["base_sample_count"], 1)
        self.assertEqual(result["unpaired_base_sample_count"], 0)
        self.assertEqual(result["oracle_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
