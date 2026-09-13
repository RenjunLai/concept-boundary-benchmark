from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import Any

from .schemas import Prediction, ScoreRecord

WORD_RE = re.compile(r"^\s*([A-Za-z]+)")
LIST_RE = re.compile(r"\[[^\]]*\]")


def parse_answer(task: str, text: str, option_count: int = 0, char_limit: int = 2000) -> tuple[bool | tuple[int, ...] | None, str | None, bool]:
    text = text or ""
    truncated = len(text) > char_limit
    limited = text[:char_limit].strip()
    if not limited:
        return None, "Null Output", truncated
    if task in ("T1", "T3"):
        match = WORD_RE.match(limited)
        if not match:
            return None, "Format Error", truncated
        first = match.group(1).lower()
        if first == "true":
            return True, None, truncated
        if first == "false":
            return False, None, truncated
        return None, "Format Error", truncated
    match = LIST_RE.search(limited)
    if not match:
        return None, "Format Error", truncated
    try:
        parsed = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None, "Format Error", truncated
    if not isinstance(parsed, list):
        return None, "Format Error", truncated
    if any(not isinstance(item, int) for item in parsed):
        return None, "Format Error", truncated
    if len(set(parsed)) != len(parsed):
        return None, "Format Error", truncated
    if any(item < 1 or item > option_count for item in parsed):
        return None, "Format Error", truncated
    return tuple(parsed), None, truncated


def answer_matches(task: str, parsed: bool | tuple[int, ...] | None, gold: bool | tuple[int, ...]) -> bool:
    if parsed is None:
        return False
    if task == "T2":
        return isinstance(parsed, tuple) and isinstance(gold, tuple) and set(parsed) == set(gold)
    return parsed == gold


def is_refusal(text: str) -> bool:
    normalized = (text or "").lower()
    markers = (
        "i can't",
        "i cannot",
        "i am unable",
        "unable to answer",
        "cannot answer",
        "can't answer",
        "无法",
        "不能回答",
        "无法回答",
        "不能提供",
    )
    return any(marker in normalized for marker in markers)


def select_responses_for_scoring(requests: list[dict[str, Any]], responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    request_ids = {request["request_id"] for request in requests}
    selected: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
    for index, response in enumerate(responses):
        request_id = response.get("request_id")
        if request_id not in request_ids:
            continue
        rank = (1 if response.get("status") == "success" else 0, index)
        if request_id not in selected or rank > selected[request_id][0]:
            selected[request_id] = (rank, response)
    return [selected[request["request_id"]][1] for request in requests if request["request_id"] in selected]


def build_predictions_and_scores(
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    answer_parse_char_limit: int = 2000,
) -> tuple[list[Prediction], list[ScoreRecord], dict[str, Any]]:
    request_by_id = {row["request_id"]: row for row in requests}
    selected_responses = select_responses_for_scoring(requests, responses)
    predictions: list[Prediction] = []
    scores: list[ScoreRecord] = []
    for response in selected_responses:
        request = request_by_id.get(response.get("request_id"))
        if not request:
            continue
        sample = request["sample"]
        task = sample["task"]
        option_count = len(sample.get("candidate_ids") or []) if task == "T2" else 0
        if response.get("status") != "success":
            parsed, parse_error, truncated = None, response.get("error_state") or "Provider Error", bool(response.get("answer_truncated"))
        else:
            parsed, parse_error, truncated = parse_answer(
                task,
                response.get("final_answer_text") or "",
                option_count=option_count,
                char_limit=answer_parse_char_limit,
            )
            if parse_error is not None and is_refusal(response.get("final_answer_text") or ""):
                parse_error = "Refusal"
        prediction = Prediction(
            sample_id=sample["sample_id"],
            request_id=request["request_id"],
            task=task,
            parsed_answer=parsed,
            parse_error_type=parse_error,
            answer_truncated=truncated or bool(response.get("answer_truncated")),
            reasoning_available=bool(response.get("reasoning_available")),
            reasoning_truncated=bool(response.get("reasoning_truncated")),
        )
        predictions.append(prediction)
        gold = sample["gold_answer"]
        if isinstance(gold, list):
            gold = tuple(gold)
        correct = parse_error is None and answer_matches(task, parsed, gold)
        scores.append(
            ScoreRecord(
                base_sample_id=sample.get("base_sample_id", sample["sample_id"]),
                sample_id=sample["sample_id"],
                request_id=request["request_id"],
                task=task,
                dataset_family=sample["dataset_family"],
                constraint_cell=tuple(sample["constraint_cell"]),
                a_count=sample["a_count"],
                b_count=sample["b_count"],
                information=sample["information"],
                sampling_weight=float(sample.get("sampling_weight", 1.0)),
                gold_answer=gold,
                parsed_answer=parsed,
                correct=correct,
                format_valid=parse_error is None,
                parse_error_type=parse_error,
            )
        )
    metrics = summarize_scores(scores)
    metrics["request_count"] = len(requests)
    metrics["response_rows_seen"] = len(responses)
    metrics["selected_response_count"] = len(selected_responses)
    return predictions, scores, metrics


def summarize_scores(scores: list[ScoreRecord]) -> dict[str, Any]:
    def bucket_key(score: ScoreRecord, fields: tuple[str, ...]) -> tuple:
        return tuple(getattr(score, field) for field in fields)

    total = len(scores)
    correct = sum(1 for score in scores if score.correct)
    weight_total = sum(score.sampling_weight for score in scores)
    weighted_correct = sum(score.sampling_weight for score in scores if score.correct)
    metrics: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "accuracy": None if total == 0 else correct / total,
        "sampling_weighted": {
            "weight_total": weight_total,
            "weighted_correct": weighted_correct,
            "weighted_accuracy": None if weight_total == 0 else weighted_correct / weight_total,
        },
        "slices": {},
    }
    for fields in [
        ("dataset_family",),
        ("task",),
        ("information",),
        ("constraint_cell",),
        ("a_count",),
        ("b_count",),
        ("dataset_family", "constraint_cell"),
        ("dataset_family", "task"),
        ("dataset_family", "task", "information"),
    ]:
        grouped: dict[tuple, list[ScoreRecord]] = defaultdict(list)
        for score in scores:
            grouped[bucket_key(score, fields)].append(score)
        metrics["slices"]["/".join(fields)] = {
            "|".join(str(item) for item in key): {
                "total": len(group),
                "correct": sum(1 for row in group if row.correct),
                "accuracy": sum(1 for row in group if row.correct) / len(group),
                "weight_total": sum(row.sampling_weight for row in group),
                "weighted_correct": sum(row.sampling_weight for row in group if row.correct),
                "weighted_accuracy": (
                    None
                    if sum(row.sampling_weight for row in group) == 0
                    else sum(row.sampling_weight for row in group if row.correct)
                    / sum(row.sampling_weight for row in group)
                ),
            }
            for key, group in grouped.items()
        }
    by_base: dict[str, list[ScoreRecord]] = defaultdict(list)
    for score in scores:
        by_base[score.base_sample_id].append(score)
    complete_pairs = 0
    pair_rows: list[dict[str, Any]] = []
    for base_sample_id, group in by_base.items():
        by_information = {row.information: row for row in group}
        if {"I0", "I1", "I2"}.issubset(by_information):
            complete_pairs += 1
            pair_rows.append(
                {
                    "base_sample_id": base_sample_id,
                    "task": group[0].task,
                    "dataset_family": group[0].dataset_family,
                    "constraint_cell": group[0].constraint_cell,
                    "I0_correct": by_information["I0"].correct,
                    "I1_correct": by_information["I1"].correct,
                    "I2_correct": by_information["I2"].correct,
                }
            )
    metrics["information_effect"] = {
        "analysis_unit": "base_sample_id",
        "complete_pair_count": complete_pairs,
        "I0_correct": sum(1 for row in pair_rows if row["I0_correct"]),
        "I1_correct": sum(1 for row in pair_rows if row["I1_correct"]),
        "I2_correct": sum(1 for row in pair_rows if row["I2_correct"]),
    }
    return metrics
