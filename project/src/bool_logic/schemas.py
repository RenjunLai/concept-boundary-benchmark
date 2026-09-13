from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Task = Literal["T1", "T2", "T3"]
DatasetFamily = Literal["wordnet", "babelnet"]
InformationLevel = Literal["I0", "I1", "I2"]


@dataclass(frozen=True)
class ResourceObject:
    dataset_family: str
    object_id: str
    surface: str
    gloss: str
    synonyms: tuple[str, ...]
    children: tuple[str, ...]
    parents: tuple[str, ...]
    usable_flags: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenSample:
    base_sample_id: str
    sample_id: str
    product_id: str
    task: str
    dataset_family: str
    constraint_cell: tuple[int, int]
    a_count: int
    b_count: int
    condition_count: int
    information: str
    seed_id: str
    A1_id: str
    A_branch_ids: tuple[str, ...]
    B_ids: tuple[str, ...]
    paired_A_ids: tuple[str, ...]
    B_sources: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    gold_answer: bool | tuple[int, ...]
    oracle_version: str
    renderer_version: str
    frame_version: str
    stratum_id: str
    inclusion_probability: float
    sampling_weight: float
    sample_hash: str
    request_hash: str
    reuse_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedRequest:
    request_id: str
    sample_id: str
    base_sample_id: str
    task: str
    dataset_family: str
    information: str
    renderer_version: str
    prompt: str
    system_prompt: str
    expected_protocol: str
    request_hash: str
    provider_payload: dict[str, Any]
    sample: dict[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    request_id: str
    sample_id: str
    attempt_id: str
    status: str
    provider: str
    model: str
    request_payload: dict[str, Any]
    raw_response: dict[str, Any] | None
    final_answer_text: str
    reasoning_text: str | None
    answer_truncated: bool
    reasoning_truncated: bool
    reasoning_available: bool
    error_state: str | None
    actual_request_parameters: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    request_id: str
    task: str
    parsed_answer: bool | tuple[int, ...] | None
    parse_error_type: str | None
    answer_truncated: bool
    reasoning_available: bool
    reasoning_truncated: bool


@dataclass(frozen=True)
class ScoreRecord:
    base_sample_id: str
    sample_id: str
    request_id: str
    task: str
    dataset_family: str
    constraint_cell: tuple[int, int]
    a_count: int
    b_count: int
    information: str
    sampling_weight: float
    gold_answer: bool | tuple[int, ...]
    parsed_answer: bool | tuple[int, ...] | None
    correct: bool
    format_valid: bool
    parse_error_type: str | None
