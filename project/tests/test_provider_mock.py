from pathlib import Path
import unittest

from bool_logic.config import ProviderConfig, load_config
from bool_logic.parsing import build_predictions_and_scores
from bool_logic.providers import (
    AliyunProvider,
    DeepSeekProvider,
    NvidiaProvider,
    OpenBayesGemmaProvider,
    MockProvider,
    ZhipuProvider,
    get_provider,
    ProviderError,
)


class ProviderMockTests(unittest.TestCase):
    def test_provider_capability_profile_is_loaded(self):
        config = load_config(Path(__file__).resolve().parents[2] / "project/configs/experiments/test_deepseek_v4_flash_nothinking.toml")

        self.assertEqual(config.provider.name, "deepseek")
        self.assertEqual(config.provider.api_key_env, "DEEPSEEK_API_KEY")
        self.assertFalse(config.provider.supports_streaming)
        self.assertTrue(config.provider.supports_reasoning_channel)
        self.assertEqual(config.provider.request_adapter, "deepseek_reasoning_effort")
        self.assertEqual(config.provider.recommended_max_concurrency, 10)

    def test_mock_provider_roundtrip(self):
        request = {
            "request_id": "r1",
            "sample_id": "s1",
            "provider_payload": {"messages": []},
            "sample": {
                "base_sample_id": "b1",
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
        response = MockProvider(ProviderConfig()).call(request)
        predictions, scores, metrics = build_predictions_and_scores([request], [response.__dict__])
        self.assertEqual(predictions[0].parsed_answer, True)
        self.assertTrue(scores[0].correct)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_zhipu_thinking_parameter_mapping(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        off_provider = object.__new__(ZhipuProvider)
        off_provider.config = ProviderConfig(thinking="off")
        off_kwargs = off_provider._request_kwargs(request)
        self.assertEqual(off_kwargs["extra_body"]["thinking"]["type"], "disabled")
        self.assertTrue(off_kwargs["extra_body"]["clear_thinking"])

        on_provider = object.__new__(ZhipuProvider)
        on_provider.config = ProviderConfig(thinking="on")
        on_kwargs = on_provider._request_kwargs(request)
        self.assertEqual(on_kwargs["extra_body"]["thinking"]["type"], "enabled")
        self.assertTrue(on_kwargs["extra_body"]["clear_thinking"])
        self.assertNotIn("stream", on_kwargs)

    def test_deepseek_no_thinking_parameter_mapping(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(DeepSeekProvider)
        provider.config = ProviderConfig(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            model="deepseek-v4-flash",
            thinking="off",
            temperature=1.0,
            top_p=1.0,
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "disabled")
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("clear_thinking", kwargs["extra_body"])
        self.assertNotIn("stream", kwargs)

    def test_deepseek_thinking_reasoning_effort_mapping(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(DeepSeekProvider)
        provider.config = ProviderConfig(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            model="deepseek-v4-pro",
            thinking="on",
            reasoning_effort="high",
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "enabled")
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_aliyun_thinking_parameter_mapping(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        off_provider = object.__new__(AliyunProvider)
        off_provider.config = ProviderConfig(name="aliyun", model="glm-5.1", thinking="off")
        off_kwargs = off_provider._request_kwargs(request)
        self.assertIs(off_kwargs["extra_body"]["enable_thinking"], False)
        self.assertNotIn("thinking", off_kwargs["extra_body"])

        on_provider = object.__new__(AliyunProvider)
        on_provider.config = ProviderConfig(name="aliyun", model="glm-5.1", thinking="on")
        on_kwargs = on_provider._request_kwargs(request)
        self.assertIs(on_kwargs["extra_body"]["enable_thinking"], True)
        self.assertNotIn("thinking", on_kwargs["extra_body"])
        self.assertNotIn("stream", on_kwargs)

    def test_nvidia_gpt_oss_mapping_uses_openai_reasoning_effort(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(NvidiaProvider)
        provider.config = ProviderConfig(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="nv_key",
            model="openai/gpt-oss-120b",
            request_adapter="openai_reasoning_effort",
            thinking="on",
            reasoning_effort="high",
            temperature=1.0,
            top_p=1.0,
            max_answer_tokens=32768,
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["model"], "openai/gpt-oss-120b")
        self.assertEqual(kwargs["extra_body"], {})
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertNotIn("stream", kwargs)

    def test_nvidia_gemma_mapping_uses_chat_template_thinking(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(NvidiaProvider)
        provider.config = ProviderConfig(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="nv_key",
            model="google/gemma-4-31b-it",
            request_adapter="nvidia_gemma_chat_template",
            thinking="on",
            temperature=1.0,
            top_p=0.95,
            max_answer_tokens=32768,
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["model"], "google/gemma-4-31b-it")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertEqual(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertTrue(kwargs["extra_body"]["include_reasoning"])
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("stream", kwargs)

    def test_nvidia_minimax_mapping_uses_plain_openai_without_thinking_controls(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(NvidiaProvider)
        provider.config = ProviderConfig(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="nv_key",
            model="minimaxai/minimax-m2.7",
            request_adapter="plain_openai",
            thinking="on",
            temperature=1.0,
            top_p=0.95,
            max_answer_tokens=32768,
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["model"], "minimaxai/minimax-m2.7")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertEqual(kwargs["extra_body"], {})
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("stream", kwargs)

    def test_openbayes_gemma_mapping_uses_plain_openai(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(OpenBayesGemmaProvider)
        provider.config = ProviderConfig(
            name="openbayes_gemma",
            base_url="https://hdu-ipg-67y8s8yw5zkl.gear-c1.openbayes.net/v1",
            api_key_env="OPENBAYES_GEMMA_API_KEY",
            model="google/gemma-4-31b-it",
            request_adapter="plain_openai",
            thinking="off",
            temperature=1.0,
            top_p=0.95,
            max_answer_tokens=128,
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["model"], "google/gemma-4-31b-it")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_tokens"], 128)
        self.assertEqual(kwargs["extra_body"], {})
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("stream", kwargs)

    def test_openbayes_gemma_thinking_mapping_uses_vllm_chat_template(self):
        request = {"provider_payload": {"messages": [{"role": "user", "content": "x"}]}}

        provider = object.__new__(OpenBayesGemmaProvider)
        provider.config = ProviderConfig(
            name="openbayes_gemma_thinking",
            base_url="https://hdu-ipg-67y8s8yw5zkl.gear-c1.openbayes.net/v1",
            api_key_env="OPENBAYES_GEMMA_API_KEY",
            model="google/gemma-4-31b-it",
            request_adapter="vllm_gemma_chat_template",
            thinking="on",
            temperature=1.0,
            top_p=0.95,
            max_answer_tokens=32768,
        )
        kwargs = provider._request_kwargs(request)

        self.assertEqual(kwargs["model"], "google/gemma-4-31b-it")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertEqual(kwargs["extra_body"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertNotIn("include_reasoning", kwargs["extra_body"])
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("stream", kwargs)

    def test_removed_nvidia_legacy_providers_are_not_new_request_sources(self):
        with self.assertRaises(ProviderError):
            get_provider("nvidia_deepseek", ProviderConfig(name="nvidia_deepseek"))
        with self.assertRaises(ProviderError):
            get_provider("nvidia_glm", ProviderConfig(name="nvidia_glm"))


if __name__ == "__main__":
    unittest.main()
