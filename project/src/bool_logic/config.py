from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .constants import A_COUNTS, B_COUNTS, DATASET_FAMILIES, INFORMATION_LEVELS, TASKS
from .io_utils import repo_root


DEFAULT_REQUEST_ADAPTERS = {
    "zhipu": "zhipu_thinking_body",
    "deepseek": "deepseek_reasoning_effort",
    "aliyun": "aliyun_enable_thinking",
    "nvidia": "openai_reasoning_effort",
}


@dataclass(frozen=True)
class MatrixConfig:
    a_count: tuple[int, ...] = A_COUNTS
    b_count: tuple[int, ...] = B_COUNTS
    information: tuple[str, ...] = INFORMATION_LEVELS


@dataclass(frozen=True)
class BudgetConfig:
    max_base_samples: int = 12
    max_base_samples_per_task_family: int = 2
    max_cells_per_task_family: int = 2
    max_seed_reuse: int = 2
    max_a1_reuse: int = 2
    max_candidate_reuse: int = 2
    max_constraint_reuse: int = 2
    max_pool_reuse: int = 1
    cell_order: str = "balanced"
    seed: int = 13


@dataclass(frozen=True)
class ResourceConfig:
    dataset_families: tuple[str, ...] = DATASET_FAMILIES
    wordnet_enabled: bool = True
    babelnet_enabled: bool = True
    babelnet_path: Path = repo_root() / "external" / "openhownet_resources" / "unzipped" / "babel_data"
    min_final_positive_count: int = 5
    max_seed_per_family: int = 0


@dataclass(frozen=True)
class ProductConfig:
    product_id: str = "test"
    tier: str = "test"
    wordnet_quota_per_stratum: int = 3
    babelnet_quota_per_stratum: int = 1
    candidate_scheduler: str = "reuse_aware_mixed_negatives"
    candidate_window_limit: int = 64
    shard_size: int = 100_000


@dataclass(frozen=True)
class ProviderCapability:
    provider_name: str = "zhipu"
    protocol: str = "openai_compatible"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env: str = "ZAI_API_KEY"
    default_model: str = "glm-5.1"
    supports_streaming: bool = False
    supports_reasoning_channel: bool = True
    request_adapter: str = ""
    thinking_mapping: str = "thinking.type enabled/disabled"
    recommended_max_concurrency: int = 1


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "zhipu"
    protocol: str = "openai_compatible"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env: str = "ZAI_API_KEY"
    model: str = "glm-5.1"
    timeout_seconds: float = 60.0
    max_concurrency: int = 1
    concurrency_mode: str = "fixed"
    max_retries: int = 1
    request_interval_seconds: float = 0.0
    retry_backoff_seconds: float = 0.0
    retryable_cooldown_seconds: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_answer_tokens: int = 128
    thinking: str = "off"
    reasoning_effort: str | None = None
    save_reasoning: bool = True
    reasoning_char_limit: int = 4000
    supports_streaming: bool = False
    supports_reasoning_channel: bool = True
    request_adapter: str = ""
    thinking_mapping: str = "thinking.type enabled/disabled"
    recommended_max_concurrency: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "smoke"
    tasks: tuple[str, ...] = TASKS
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    product: ProductConfig = field(default_factory=ProductConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    answer_parse_char_limit: int = 2000


def _tuple(value, default):
    if value is None:
        return tuple(default)
    return tuple(value)


def load_provider_capabilities(path: Path | None = None) -> dict[str, ProviderCapability]:
    path = path or (repo_root() / "project" / "configs" / "providers.toml")
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    capabilities: dict[str, ProviderCapability] = {}
    for name, values in raw.items():
        capabilities[name] = ProviderCapability(
            provider_name=name,
            protocol=str(values.get("protocol", ProviderCapability().protocol)),
            base_url=str(values.get("base_url", ProviderCapability().base_url)),
            api_key_env=str(values.get("api_key_env", ProviderCapability().api_key_env)),
            default_model=str(values.get("default_model", ProviderCapability().default_model)),
            supports_streaming=bool(values.get("supports_streaming", ProviderCapability().supports_streaming)),
            supports_reasoning_channel=bool(
                values.get("supports_reasoning_channel", ProviderCapability().supports_reasoning_channel)
            ),
            request_adapter=str(values.get("request_adapter", "")),
            thinking_mapping=str(values.get("thinking_mapping", ProviderCapability().thinking_mapping)),
            recommended_max_concurrency=int(
                values.get("recommended_max_concurrency", ProviderCapability().recommended_max_concurrency)
            ),
        )
    return capabilities


def load_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    matrix_raw = raw.get("matrix", {})
    resources_raw = raw.get("resources", {})
    budgets_raw = raw.get("budgets", {})
    product_raw = raw.get("product", {})
    provider_raw = raw.get("provider", {})
    provider_name = str(provider_raw.get("name", "zhipu"))
    provider_capability = load_provider_capabilities().get(provider_name, ProviderCapability(provider_name=provider_name))
    request_adapter = provider_raw.get("request_adapter", provider_capability.request_adapter)
    if not request_adapter:
        request_adapter = DEFAULT_REQUEST_ADAPTERS.get(provider_name, "")
    thinking_mapping = str(provider_raw.get("thinking_mapping", provider_capability.thinking_mapping))
    babelnet_path = Path(resources_raw.get("babelnet_path", ResourceConfig().babelnet_path))
    if not babelnet_path.is_absolute():
        babelnet_path = (path.parent / babelnet_path).resolve()
    return ExperimentConfig(
        name=raw.get("name", path.stem),
        tasks=_tuple(raw.get("tasks"), TASKS),
        matrix=MatrixConfig(
            a_count=_tuple(matrix_raw.get("a_count"), A_COUNTS),
            b_count=_tuple(matrix_raw.get("b_count"), B_COUNTS),
            information=_tuple(matrix_raw.get("information"), INFORMATION_LEVELS),
        ),
        resources=ResourceConfig(
            dataset_families=_tuple(resources_raw.get("dataset_families"), DATASET_FAMILIES),
            wordnet_enabled=bool(resources_raw.get("wordnet_enabled", True)),
            babelnet_enabled=bool(resources_raw.get("babelnet_enabled", True)),
            babelnet_path=babelnet_path,
            min_final_positive_count=int(resources_raw.get("min_final_positive_count", 5)),
            max_seed_per_family=int(resources_raw.get("max_seed_per_family", 0)),
        ),
        budgets=BudgetConfig(
            max_base_samples=int(budgets_raw.get("max_base_samples", 12)),
            max_base_samples_per_task_family=int(budgets_raw.get("max_base_samples_per_task_family", 2)),
            max_cells_per_task_family=int(budgets_raw.get("max_cells_per_task_family", 2)),
            max_seed_reuse=int(budgets_raw.get("max_seed_reuse", 2)),
            max_a1_reuse=int(budgets_raw.get("max_a1_reuse", 2)),
            max_candidate_reuse=int(budgets_raw.get("max_candidate_reuse", 2)),
            max_constraint_reuse=int(budgets_raw.get("max_constraint_reuse", 2)),
            max_pool_reuse=int(budgets_raw.get("max_pool_reuse", 1)),
            cell_order=str(budgets_raw.get("cell_order", "balanced")),
            seed=int(budgets_raw.get("seed", 13)),
        ),
        product=ProductConfig(
            product_id=str(product_raw.get("product_id", ProductConfig().product_id)),
            tier=str(product_raw.get("tier", ProductConfig().tier)),
            wordnet_quota_per_stratum=int(
                product_raw.get("wordnet_quota_per_stratum", ProductConfig().wordnet_quota_per_stratum)
            ),
            babelnet_quota_per_stratum=int(
                product_raw.get("babelnet_quota_per_stratum", ProductConfig().babelnet_quota_per_stratum)
            ),
            candidate_scheduler=str(product_raw.get("candidate_scheduler", ProductConfig().candidate_scheduler)),
            candidate_window_limit=int(product_raw.get("candidate_window_limit", ProductConfig().candidate_window_limit)),
            shard_size=int(product_raw.get("shard_size", ProductConfig().shard_size)),
        ),
        provider=ProviderConfig(
            name=provider_name,
            protocol=provider_raw.get("protocol", provider_capability.protocol),
            base_url=provider_raw.get("base_url", provider_capability.base_url),
            api_key_env=provider_raw.get("api_key_env", provider_capability.api_key_env),
            model=provider_raw.get("model", provider_capability.default_model),
            timeout_seconds=float(provider_raw.get("timeout_seconds", 60.0)),
            max_concurrency=int(provider_raw.get("max_concurrency", 1)),
            concurrency_mode=str(provider_raw.get("concurrency_mode", "fixed")),
            max_retries=int(provider_raw.get("max_retries", 1)),
            request_interval_seconds=float(provider_raw.get("request_interval_seconds", 0.0)),
            retry_backoff_seconds=float(provider_raw.get("retry_backoff_seconds", 0.0)),
            retryable_cooldown_seconds=float(provider_raw.get("retryable_cooldown_seconds", 0.0)),
            temperature=float(provider_raw.get("temperature", 0.0)),
            top_p=float(provider_raw.get("top_p", 1.0)),
            max_answer_tokens=int(provider_raw.get("max_answer_tokens", 128)),
            thinking=provider_raw.get("thinking", "off"),
            reasoning_effort=provider_raw.get("reasoning_effort"),
            save_reasoning=bool(provider_raw.get("save_reasoning", True)),
            reasoning_char_limit=int(provider_raw.get("reasoning_char_limit", 4000)),
            supports_streaming=provider_capability.supports_streaming,
            supports_reasoning_channel=provider_capability.supports_reasoning_channel,
            request_adapter=request_adapter,
            thinking_mapping=thinking_mapping,
            recommended_max_concurrency=provider_capability.recommended_max_concurrency,
        ),
        answer_parse_char_limit=int(raw.get("answer_parse_char_limit", 2000)),
    )


def provider_config_for_run(config: ExperimentConfig, provider_name: str) -> ProviderConfig:
    if provider_name == config.provider.name:
        return config.provider
    capabilities = load_provider_capabilities()
    capability = capabilities.get(provider_name)
    if capability is None:
        if provider_name == "mock":
            capability = ProviderCapability(
                provider_name="mock",
                protocol="local_mock",
                base_url="",
                api_key_env="",
                default_model="mock",
                supports_streaming=False,
                supports_reasoning_channel=True,
                request_adapter="mock",
                thinking_mapping="mock fixed gold-answer provider",
                recommended_max_concurrency=config.provider.max_concurrency,
            )
        else:
            capability = ProviderCapability(provider_name=provider_name)
    request_adapter = capability.request_adapter
    if not request_adapter:
        request_adapter = DEFAULT_REQUEST_ADAPTERS.get(provider_name, "")
    return ProviderConfig(
        name=provider_name,
        protocol=capability.protocol,
        base_url=capability.base_url,
        api_key_env=capability.api_key_env,
        model=capability.default_model,
        timeout_seconds=config.provider.timeout_seconds,
        max_concurrency=config.provider.max_concurrency,
        concurrency_mode=config.provider.concurrency_mode,
        max_retries=config.provider.max_retries,
        request_interval_seconds=config.provider.request_interval_seconds,
        retry_backoff_seconds=config.provider.retry_backoff_seconds,
        retryable_cooldown_seconds=config.provider.retryable_cooldown_seconds,
        temperature=config.provider.temperature,
        top_p=config.provider.top_p,
        max_answer_tokens=config.provider.max_answer_tokens,
        thinking=config.provider.thinking,
        reasoning_effort=config.provider.reasoning_effort,
        save_reasoning=config.provider.save_reasoning,
        reasoning_char_limit=config.provider.reasoning_char_limit,
        supports_streaming=capability.supports_streaming,
        supports_reasoning_channel=capability.supports_reasoning_channel,
        request_adapter=request_adapter,
        thinking_mapping=capability.thinking_mapping,
        recommended_max_concurrency=capability.recommended_max_concurrency,
    )
