import unittest

from bool_logic.constants import RENDERING_TEMPLATE_VERSION
from bool_logic.rendering import render_sample


RESOURCES = {
    "wordnet": {
        "a1": {"surface": "ability", "gloss": "the quality of being able to perform"},
        "a2": {"surface": "skill", "gloss": "an ability acquired by training"},
        "b1": {"surface": "weapon", "gloss": "an instrument used in fighting"},
        "c1": {"surface": "aptitude", "gloss": "inherent ability"},
        "c2": {"surface": "aba", "gloss": "a loose sleeveless outer garment"},
        "c3": {"surface": "abacus", "gloss": "a calculator that performs arithmetic functions"},
        "c4": {"surface": "abandon", "gloss": "lack of restraint or control"},
    },
    "babelnet": {
        "a1": {"surface": "能力", "gloss": "能够完成某事的性质。"},
        "a2": {"surface": "技能", "gloss": "通过训练获得的能力。"},
        "b1": {"surface": "武器", "gloss": "用于战斗的工具。"},
        "c1": {"surface": "适应性", "gloss": "适应变化环境的能力。"},
        "c2": {"surface": "外衣", "gloss": "穿在外面的衣服。"},
        "c3": {"surface": "算盘", "gloss": "用于计算的工具。"},
        "c4": {"surface": "放纵", "gloss": "缺少约束或控制。"},
    },
}


def sample(family="wordnet", task="T2", information="I0"):
    candidate_ids = ["c1"] if task == "T1" else ["c1", "c2", "c3", "c4"]
    return {
        "base_sample_id": "base1",
        "sample_id": f"{family}_{task}_{information}",
        "renderer_version": RENDERING_TEMPLATE_VERSION,
        "task": task,
        "dataset_family": family,
        "information": information,
        "A1_id": "a1",
        "A_branch_ids": ["a2"],
        "B_ids": ["b1"],
        "candidate_ids": candidate_ids,
    }


class RenderingTests(unittest.TestCase):
    def test_wordnet_i0_has_surfaces_without_gloss(self):
        request = render_sample(sample("wordnet", "T2", "I0"), RESOURCES)
        self.assertEqual(request.renderer_version, RENDERING_TEMPLATE_VERSION)
        self.assertEqual(request.request_hash, request.sample["request_hash"])
        self.assertTrue(request.request_id.endswith(request.request_hash[:16]))
        self.assertEqual(request.sample["renderer_version"], RENDERING_TEMPLATE_VERSION)
        self.assertIn("- A1 ability is the outer required concept.", request.prompt)
        self.assertIn("at least one concept below", request.prompt)
        self.assertIn("- A2 skill", request.prompt)
        self.assertIn("- B1 weapon", request.prompt)
        self.assertIn("1. aptitude", request.prompt)
        self.assertNotIn("Concept notes", request.prompt)
        self.assertNotIn("the quality of being able to perform", request.prompt)
        self.assertNotIn("inherent ability", request.prompt)

    def test_wordnet_i1_has_concept_gloss_but_not_candidate_gloss(self):
        request = render_sample(sample("wordnet", "T2", "I1"), RESOURCES)
        self.assertIn("Concept notes:", request.prompt)
        self.assertIn("- A1 ability: the quality of being able to perform", request.prompt)
        self.assertIn("- A2 skill: an ability acquired by training", request.prompt)
        self.assertIn("- B1 weapon: an instrument used in fighting", request.prompt)
        self.assertNotIn("Candidate notes:", request.prompt)
        self.assertNotIn("inherent ability", request.prompt)

    def test_wordnet_i2_has_candidate_gloss(self):
        request = render_sample(sample("wordnet", "T2", "I2"), RESOURCES)
        self.assertIn("Candidate notes:", request.prompt)
        self.assertIn("- 1. aptitude: inherent ability", request.prompt)

    def test_babelnet_uses_chinese_template_and_separated_gloss(self):
        request = render_sample(sample("babelnet", "T3", "I1"), RESOURCES)
        self.assertIn("请判断下面的概念边界题。", request.prompt)
        self.assertIn("是外层必须满足的概念", request.prompt)
        self.assertIn("- A1 能力", request.prompt)
        self.assertIn("至少属于下面一个概念", request.prompt)
        self.assertIn("- A2 技能", request.prompt)
        self.assertIn("概念解释：", request.prompt)
        self.assertIn("- A1 能力: 能够完成某事的性质。", request.prompt)
        self.assertNotIn("You are given", request.prompt)
        self.assertNotIn("Candidate notes", request.prompt)
        self.assertNotIn("候选解释：", request.prompt)
        self.assertNotIn("适应变化环境的能力。", request.prompt)
        self.assertIn("第一个词必须是 True 或 False", request.prompt)


if __name__ == "__main__":
    unittest.main()
