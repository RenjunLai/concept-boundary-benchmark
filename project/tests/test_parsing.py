import unittest

from bool_logic.parsing import build_predictions_and_scores, parse_answer


class ParsingTests(unittest.TestCase):
    def test_boolean_first_word(self):
        self.assertEqual(parse_answer("T1", "True, because x", 0)[0], True)
        self.assertEqual(parse_answer("T3", "False.", 0)[0], False)
        parsed, error, _ = parse_answer("T1", "The answer is true.", 0)
        self.assertIsNone(parsed)
        self.assertEqual(error, "Format Error")

    def test_t2_candidate_list(self):
        self.assertEqual(parse_answer("T2", "[1, 3]", 4)[0], (1, 3))
        self.assertEqual(parse_answer("T2", "[]", 4)[0], ())
        self.assertEqual(parse_answer("T2", "[1, 1]", 4)[1], "Format Error")
        self.assertEqual(parse_answer("T2", "[5]", 4)[1], "Format Error")

    def test_t2_scores_as_exact_set_and_deduplicates_responses(self):
        request = {
            "request_id": "r1",
            "sample": {
                "sample_id": "s1",
                "task": "T2",
                "dataset_family": "wordnet",
                "constraint_cell": [1, 0],
                "a_count": 1,
                "b_count": 0,
                "base_sample_id": "b1",
                "information": "I0",
                "candidate_ids": ["a", "b", "c", "d"],
                "gold_answer": [1, 3],
                "sampling_weight": 2.0,
            },
        }
        responses = [
            {"request_id": "r1", "status": "error", "error_state": "Timeout", "final_answer_text": ""},
            {"request_id": "r1", "status": "success", "final_answer_text": "[3, 1]"},
        ]
        predictions, scores, metrics = build_predictions_and_scores([request], responses)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(scores), 1)
        self.assertTrue(scores[0].correct)
        self.assertEqual(scores[0].base_sample_id, "b1")
        self.assertEqual(scores[0].sampling_weight, 2.0)
        self.assertEqual(metrics["sampling_weighted"]["weighted_accuracy"], 1.0)
        self.assertEqual(metrics["response_rows_seen"], 2)
        self.assertEqual(metrics["selected_response_count"], 1)

    def test_reasoning_text_is_audit_only_and_not_scored(self):
        request = {
            "request_id": "r1",
            "sample": {
                "sample_id": "s1",
                "task": "T1",
                "dataset_family": "wordnet",
                "constraint_cell": [1, 0],
                "a_count": 1,
                "b_count": 0,
                "base_sample_id": "b1",
                "information": "I0",
                "candidate_ids": ["x"],
                "gold_answer": True,
            },
        }
        response = {
            "request_id": "r1",
            "status": "success",
            "final_answer_text": "True",
            "reasoning_text": "False would be tempting in the trace.",
            "reasoning_available": True,
            "reasoning_truncated": False,
        }
        predictions, scores, _ = build_predictions_and_scores([request], [response])

        self.assertEqual(predictions[0].parsed_answer, True)
        self.assertTrue(predictions[0].reasoning_available)
        self.assertTrue(scores[0].correct)

    def test_reasoning_without_final_answer_is_null_output(self):
        request = {
            "request_id": "r1",
            "sample": {
                "sample_id": "s1",
                "task": "T1",
                "dataset_family": "wordnet",
                "constraint_cell": [1, 0],
                "a_count": 1,
                "b_count": 0,
                "base_sample_id": "b1",
                "information": "I0",
                "candidate_ids": ["x"],
                "gold_answer": True,
            },
        }
        response = {
            "request_id": "r1",
            "status": "success",
            "final_answer_text": "",
            "reasoning_text": "True appears only in reasoning.",
            "reasoning_available": True,
            "reasoning_truncated": False,
        }
        predictions, scores, _ = build_predictions_and_scores([request], [response])

        self.assertIsNone(predictions[0].parsed_answer)
        self.assertEqual(predictions[0].parse_error_type, "Null Output")
        self.assertFalse(scores[0].correct)

    def test_refusal_is_separate_parse_error(self):
        request = {
            "request_id": "r1",
            "sample": {
                "base_sample_id": "b1",
                "sample_id": "s1",
                "task": "T1",
                "dataset_family": "wordnet",
                "constraint_cell": [1, 0],
                "a_count": 1,
                "b_count": 0,
                "information": "I0",
                "candidate_ids": ["x"],
                "gold_answer": True,
            },
        }
        response = {"request_id": "r1", "status": "success", "final_answer_text": "I cannot answer this question."}
        predictions, scores, _ = build_predictions_and_scores([request], [response])

        self.assertEqual(predictions[0].parse_error_type, "Refusal")
        self.assertFalse(scores[0].correct)


if __name__ == "__main__":
    unittest.main()
