from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from .config import DEFAULT_REQUEST_ADAPTERS, ProviderConfig
from .io_utils import stable_hash
from .schemas import ProviderResponse


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ProviderError(f"missing required environment variable: {config.api_key_env}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("openai SDK is not installed; run `uv pip install --system openai` in the active environment") from exc
        self.client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)

    def _thinking_type(self) -> str:
        return "enabled" if self.config.thinking == "on" else "disabled"

    def _request_adapter(self) -> str:
        return self.config.request_adapter or DEFAULT_REQUEST_ADAPTERS.get(self.config.name, "zhipu_thinking_body")

    def _extra_body(self) -> dict[str, Any]:
        adapter = self._request_adapter()
        if adapter == "zhipu_thinking_body":
            return {
                "thinking": {"type": self._thinking_type()},
                "clear_thinking": True,
            }
        if adapter == "deepseek_reasoning_effort":
            return {"thinking": {"type": self._thinking_type()}}
        if adapter == "aliyun_enable_thinking":
            return {"enable_thinking": self.config.thinking == "on"}
        if adapter == "nvidia_gemma_chat_template":
            return {
                "chat_template_kwargs": {"enable_thinking": self.config.thinking == "on"},
                "include_reasoning": self.config.save_reasoning and self.config.thinking == "on",
            }
        if adapter == "vllm_gemma_chat_template":
            return {"chat_template_kwargs": {"enable_thinking": self.config.thinking == "on"}}
        if adapter in {"openai_reasoning_effort", "plain_openai"}:
            return {}
        raise ProviderError(f"unknown request_adapter: {adapter}")

    def _request_kwargs(self, request: dict[str, Any]) -> dict[str, Any]:
        adapter = self._request_adapter()
        kwargs = {
            "model": self.config.model,
            "messages": request["provider_payload"]["messages"],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_answer_tokens,
            "extra_body": self._extra_body(),
        }
        if adapter in {"deepseek_reasoning_effort", "openai_reasoning_effort"}:
            if self.config.thinking == "on" and self.config.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
        return kwargs

    def _message_text_fields(self, message: Any) -> tuple[str, str | None]:
        final_answer = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning", None)
        if hasattr(message, "model_dump"):
            message_dump = message.model_dump()
            if reasoning is None:
                reasoning = message_dump.get("reasoning_content") or message_dump.get("reasoning")
        reasoning_text = str(reasoning) if reasoning is not None else None
        return final_answer, reasoning_text

    def call(self, request: dict[str, Any], attempt_index: int = 1) -> ProviderResponse:
        kwargs = self._request_kwargs(request)
        request_adapter = self._request_adapter()
        attempt_id = f"attempt_{stable_hash({'request_id': request['request_id'], 'attempt': attempt_index, 'time': time.time_ns()})}"
        try:
            response = self.client.chat.completions.create(**kwargs)
            raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
            message = response.choices[0].message
            final_answer, reasoning_text = self._message_text_fields(message)
            reasoning_truncated = False
            if reasoning_text and len(reasoning_text) > self.config.reasoning_char_limit:
                reasoning_text = reasoning_text[: self.config.reasoning_char_limit]
                reasoning_truncated = True
            return ProviderResponse(
                request_id=request["request_id"],
                sample_id=request["sample_id"],
                attempt_id=attempt_id,
                status="success",
                provider=self.config.name,
                model=self.config.model,
                request_payload=kwargs,
                raw_response=raw,
                final_answer_text=final_answer,
                reasoning_text=reasoning_text if self.config.save_reasoning else None,
                answer_truncated=False,
                reasoning_truncated=reasoning_truncated,
                reasoning_available=bool(reasoning_text),
                error_state=None,
                actual_request_parameters={
                    "protocol": self.config.protocol,
                    "base_url": self.config.base_url,
                    "model": self.config.model,
                    "request_adapter": request_adapter,
                    "thinking": self.config.thinking,
                    "reasoning_effort": self.config.reasoning_effort,
                    "save_reasoning": self.config.save_reasoning,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_answer_tokens": self.config.max_answer_tokens,
                },
            )
        except Exception as exc:  # provider errors must be persisted as run artifacts
            return ProviderResponse(
                request_id=request["request_id"],
                sample_id=request["sample_id"],
                attempt_id=attempt_id,
                status="error",
                provider=self.config.name,
                model=self.config.model,
                request_payload=kwargs,
                raw_response=None,
                final_answer_text="",
                reasoning_text=None,
                answer_truncated=False,
                reasoning_truncated=False,
                reasoning_available=False,
                error_state=type(exc).__name__ + ": " + str(exc),
                actual_request_parameters={
                    "protocol": self.config.protocol,
                    "base_url": self.config.base_url,
                    "model": self.config.model,
                    "request_adapter": request_adapter,
                    "thinking": self.config.thinking,
                    "reasoning_effort": self.config.reasoning_effort,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                },
            )


class ZhipuProvider(OpenAICompatibleProvider):
    pass


class DeepSeekProvider(OpenAICompatibleProvider):
    pass


class AliyunProvider(OpenAICompatibleProvider):
    pass


class NvidiaProvider(OpenAICompatibleProvider):
    pass


class OpenBayesGemmaProvider(OpenAICompatibleProvider):
    pass


class MockProvider:
    def __init__(self, config: ProviderConfig, mode: str = "gold"):
        self.config = config
        self.mode = mode

    def call(self, request: dict[str, Any], attempt_index: int = 1) -> ProviderResponse:
        sample = request["sample"]
        if self.mode == "empty":
            answer = ""
        elif sample["task"] == "T2":
            answer = str(sample["gold_answer"]).replace("(", "[").replace(")", "]")
        else:
            answer = "True" if sample["gold_answer"] else "False"
        return ProviderResponse(
            request_id=request["request_id"],
            sample_id=request["sample_id"],
            attempt_id=f"attempt_{stable_hash({'request_id': request['request_id'], 'attempt': attempt_index})}",
            status="success",
            provider="mock",
            model="mock",
            request_payload=request["provider_payload"],
            raw_response={"mock": True, "content": answer, "reasoning_content": "mock reasoning"},
            final_answer_text=answer,
            reasoning_text="mock reasoning",
            answer_truncated=False,
            reasoning_truncated=False,
            reasoning_available=True,
            error_state=None,
            actual_request_parameters={"provider": "mock", "mode": self.mode},
        )


def get_provider(name: str, config: ProviderConfig):
    if name == "zhipu":
        return ZhipuProvider(config)
    if name == "deepseek":
        return DeepSeekProvider(config)
    if name == "aliyun":
        return AliyunProvider(config)
    if name == "nvidia":
        return NvidiaProvider(config)
    if name == "openbayes_gemma":
        return OpenBayesGemmaProvider(config)
    if name == "openbayes_gemma_thinking":
        return OpenBayesGemmaProvider(config)
    if name == "mock":
        return MockProvider(config)
    raise ProviderError(f"unknown provider: {name}")
