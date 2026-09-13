from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import nbformat as nbf
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from .io_utils import ensure_dir, write_json, write_jsonl
from .parsing import answer_matches, parse_answer


@dataclass(frozen=True)
class EvaluationReportPaths:
    output_dir: Path
    rows_path: Path
    metrics_path: Path
    notebook_path: Path


class SnapshotInventory:
    def __init__(self, objects: dict[str, dict[str, Any]]):
        self.objects = objects
        self.object_ids = set(objects)
        self.children = {key: tuple(value.get("children") or ()) for key, value in objects.items()}
        self._closure_cache: dict[str, frozenset[str]] = {}

    def usable_closure(self, object_id: str | None) -> frozenset[str]:
        if not object_id:
            return frozenset()
        cached = self._closure_cache.get(object_id)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = list(self.children.get(object_id, ()))
        while stack:
            current = stack.pop()
            if current in seen or current not in self.object_ids:
                continue
            seen.add(current)
            stack.extend(self.children.get(current, ()))
        cached = frozenset(seen & self.object_ids)
        self._closure_cache[object_id] = cached
        return cached


def normalize_gold(task: str, gold: Any) -> bool | tuple[int, ...]:
    if task == "T2":
        return tuple(gold or ())
    return bool(gold)


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _pct(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{100 * float(value):.2f}%"


def _number(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.4f}"
    return f"{int(value):,}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        return str(value)

    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(out)


def load_requests(run_dir: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    request_order: list[str] = []
    request_meta: dict[str, dict[str, Any]] = {}
    with (run_dir / "requests.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = row.get("sample") or {}
            task = sample.get("task")
            request_id = row["request_id"]
            a_count = int(sample.get("a_count", 0))
            b_count = int(sample.get("b_count", 0))
            request_order.append(request_id)
            request_meta[request_id] = {
                "sample": sample,
                "task": task,
                "dataset_family": sample.get("dataset_family"),
                "information": sample.get("information"),
                "a_count": a_count,
                "b_count": b_count,
                "constraint_cell": f"A{a_count}:B{b_count}",
                "gold": normalize_gold(task, sample.get("gold_answer")),
                "option_count": len(sample.get("candidate_ids") or ()) if task == "T2" else 0,
            }
    return request_order, request_meta


def select_responses(run_dir: Path, request_meta: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    selected: dict[str, dict[str, Any]] = {}
    selected_rank: dict[str, tuple[int, int]] = {}
    response_rows_seen = 0
    responses_path = run_dir / "responses.jsonl"
    if not responses_path.exists():
        return selected, response_rows_seen
    with responses_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            response_rows_seen += 1
            row = json.loads(line)
            request_id = row.get("request_id")
            if request_id not in request_meta:
                continue
            rank = (1 if row.get("status") == "success" else 0, index)
            if request_id in selected_rank and rank <= selected_rank[request_id]:
                continue
            raw_response = row.get("raw_response") or {}
            usage = raw_response.get("usage") or {}
            choices = raw_response.get("choices") or []
            finish_reason = "unknown"
            if choices:
                finish_reason = choices[0].get("finish_reason") or choices[0].get("stop_reason") or "unknown"
            selected_rank[request_id] = rank
            selected[request_id] = {
                "status": row.get("status") or "unknown",
                "final_answer_text": row.get("final_answer_text") or "",
                "error_state": row.get("error_state"),
                "answer_truncated": bool(row.get("answer_truncated")),
                "reasoning_available": bool(row.get("reasoning_available")),
                "reasoning_truncated": bool(row.get("reasoning_truncated")),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": finish_reason,
            }
    return selected, response_rows_seen


def build_evaluation_rows(
    request_order: list[str],
    request_meta: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    answer_parse_char_limit: int = 2000,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parse_errors: Counter[tuple[str, str]] = Counter()
    t2_rows: list[dict[str, Any]] = []
    for request_id in request_order:
        meta = request_meta[request_id]
        sample = meta["sample"]
        response = selected.get(request_id)
        provider_error = False
        answer_truncated = False
        reasoning_available = False
        reasoning_truncated = False
        if response is None:
            parsed = None
            parse_error = "Missing Response"
            status = "missing"
            finish_reason = "unknown"
        elif response["status"] != "success":
            parsed = None
            parse_error = response.get("error_state") or "Provider Error"
            status = response["status"]
            finish_reason = response.get("finish_reason") or "unknown"
            provider_error = True
            answer_truncated = bool(response.get("answer_truncated"))
            reasoning_available = bool(response.get("reasoning_available"))
            reasoning_truncated = bool(response.get("reasoning_truncated"))
        else:
            parsed, parse_error, parse_truncated = parse_answer(
                meta["task"],
                response.get("final_answer_text") or "",
                option_count=meta["option_count"],
                char_limit=answer_parse_char_limit,
            )
            status = response["status"]
            finish_reason = response.get("finish_reason") or "unknown"
            answer_truncated = parse_truncated or bool(response.get("answer_truncated"))
            reasoning_available = bool(response.get("reasoning_available"))
            reasoning_truncated = bool(response.get("reasoning_truncated"))
        if parse_error is not None:
            parse_errors[(str(meta["task"]), str(parse_error))] += 1
        correct = answer_matches(str(meta["task"]), parsed, meta["gold"]) if parse_error is None else False
        gold_set_size = len(meta["gold"]) if meta["task"] == "T2" else None
        predicted_set_size = len(parsed) if meta["task"] == "T2" and isinstance(parsed, tuple) else None
        row = {
            "request_id": request_id,
            "base_sample_id": sample.get("base_sample_id", sample.get("sample_id")),
            "sample_id": sample.get("sample_id"),
            "task": meta["task"],
            "dataset_family": meta["dataset_family"],
            "information": meta["information"],
            "a_count": meta["a_count"],
            "b_count": meta["b_count"],
            "constraint_cell": meta["constraint_cell"],
            "sampling_weight": float(sample.get("sampling_weight", 1.0)),
            "gold_answer": meta["gold"],
            "parsed_answer": parsed,
            "correct": bool(correct),
            "format_valid": parse_error is None,
            "parse_error_type": parse_error,
            "provider_error": provider_error,
            "status": status,
            "answer_truncated": answer_truncated,
            "reasoning_available": reasoning_available,
            "reasoning_truncated": reasoning_truncated,
            "gold_set_size": gold_set_size,
            "predicted_set_size": predicted_set_size,
            "predicted_size_label": str(predicted_set_size) if predicted_set_size is not None else "invalid",
            "pred_true": parsed is True if meta["task"] in {"T1", "T3"} else None,
            "gold_true": bool(meta["gold"]) if meta["task"] in {"T1", "T3"} else None,
            "finish_reason": finish_reason,
        }
        rows.append(row)
        if meta["task"] == "T2":
            t2_rows.append(row | {"sample": sample})
    return rows, parse_errors, t2_rows


def load_snapshot_inventories(run_dir: Path) -> dict[str, SnapshotInventory]:
    path = run_dir / "resource_snapshot.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {family: SnapshotInventory(objects) for family, objects in data.items()}


def candidate_category(inventory: SnapshotInventory, sample: dict[str, Any], candidate_id: str, index: int) -> str:
    gold_set = set(sample.get("gold_answer") or ())
    if index in gold_set:
        return "positive"
    a1_id = sample.get("A1_id")
    a_branch_ids = tuple(sample.get("A_branch_ids") or ())
    b_ids = tuple(sample.get("B_ids") or ())
    inside_a1 = candidate_id in inventory.usable_closure(a1_id)
    branch_hits = tuple(branch_id for branch_id in a_branch_ids if candidate_id in inventory.usable_closure(branch_id))
    hits_positive_branch = bool(branch_hits) if a_branch_ids else inside_a1
    hits_any_b = any(candidate_id in inventory.usable_closure(b_id) for b_id in b_ids)
    if inside_a1 and hits_positive_branch and hits_any_b:
        return "B-excluded"
    return "other-negative"


def build_t2_candidate_frame(run_dir: Path, t2_rows: list[dict[str, Any]]) -> pd.DataFrame:
    inventories = load_snapshot_inventories(run_dir)
    if not inventories:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in t2_rows:
        sample = row["sample"]
        inventory = inventories.get(sample.get("dataset_family"))
        if inventory is None:
            continue
        parsed = row.get("parsed_answer")
        selected_positions = set(parsed) if isinstance(parsed, tuple) else set()
        for index, candidate_id in enumerate(sample.get("candidate_ids") or (), 1):
            category = candidate_category(inventory, sample, candidate_id, index)
            rows.append(
                {
                    "request_id": row["request_id"],
                    "dataset_family": row["dataset_family"],
                    "information": row["information"],
                    "a_count": row["a_count"],
                    "b_count": row["b_count"],
                    "gold_set_size": row["gold_set_size"],
                    "category": category,
                    "selected": index in selected_positions,
                }
            )
    return pd.DataFrame(rows)


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_bool_dtype(clean[col]):
            clean[col] = clean[col].astype(bool)
    return json.loads(clean.replace({np.nan: None}).to_json(orient="records", force_ascii=False))


def accuracy_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        columns = [*group_cols, "total", "correct", "accuracy"]
        return pd.DataFrame(columns=columns)
    if not group_cols:
        return pd.DataFrame(
            [
                {
                    "total": len(df),
                    "correct": int(df["correct"].sum()),
                    "accuracy": float(df["correct"].mean()) if len(df) else math.nan,
                }
            ]
        )
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(total=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
    )
    out["correct"] = out["correct"].astype(int)
    return out


def binary_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    valid = df[df["task"].isin(["T1", "T3"])] if not df.empty else df
    if valid.empty:
        return {
            "n": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
            "accuracy": None,
            "positive_precision": None,
            "positive_recall": None,
            "negative_precision": None,
            "negative_recall": None,
            "predicted_true_rate": None,
        }
    tp = int(((valid["gold_true"] == True) & (valid["pred_true"] == True)).sum())
    fp = int(((valid["gold_true"] == False) & (valid["pred_true"] == True)).sum())
    fn = int(((valid["gold_true"] == True) & (valid["pred_true"] != True)).sum())
    tn = int(((valid["gold_true"] == False) & (valid["pred_true"] != True)).sum())
    return {
        "n": len(valid),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": _safe_rate(tp + tn, len(valid)),
        "positive_precision": _safe_rate(tp, tp + fp),
        "positive_recall": _safe_rate(tp, tp + fn),
        "negative_precision": _safe_rate(tn, tn + fn),
        "negative_recall": _safe_rate(tn, tn + fp),
        "predicted_true_rate": _safe_rate(tp + fp, len(valid)),
    }


def accuracy_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = accuracy_table(df, ["a_count", "b_count"])
    matrix = grouped.pivot(index="a_count", columns="b_count", values="accuracy")
    return matrix.reindex(index=sorted(df["a_count"].dropna().unique()), columns=sorted(df["b_count"].dropna().unique()))


def binary_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (a_count, b_count), group in df.groupby(["a_count", "b_count"], dropna=False):
        metrics = binary_metrics(group)
        records.append({"a_count": a_count, "b_count": b_count, metric: metrics[metric]})
    matrix = pd.DataFrame(records).pivot(index="a_count", columns="b_count", values=metric)
    return matrix.reindex(index=sorted(df["a_count"].dropna().unique()), columns=sorted(df["b_count"].dropna().unique()))


def matrix_payload(matrix: pd.DataFrame) -> dict[str, Any]:
    if matrix.empty:
        return {"index": [], "columns": [], "values": []}
    values: list[list[float | None]] = []
    for _, row in matrix.iterrows():
        value_row: list[float | None] = []
        for value in row:
            value_row.append(None if pd.isna(value) else float(value))
        values.append(value_row)
    return {
        "index": [int(value) for value in matrix.index],
        "columns": [int(value) for value in matrix.columns],
        "values": values,
    }


def t2_size_distribution_payload(t2_df: pd.DataFrame) -> dict[str, Any]:
    order_cols = ["0", "1", "2", "3", "4", "invalid"]
    if t2_df.empty:
        return {"gold_set_size": [0, 1, 2], "predicted_size_labels": order_cols, "counts": [], "shares": []}
    counts = pd.crosstab(t2_df["gold_set_size"], t2_df["predicted_size_label"])
    counts = counts.reindex(index=[0, 1, 2], columns=order_cols, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    return {
        "gold_set_size": [int(value) for value in counts.index],
        "predicted_size_labels": order_cols,
        "counts": counts.astype(int).values.tolist(),
        "shares": shares.fillna(0.0).astype(float).values.tolist(),
    }


def build_evaluation_metrics(
    run_dir: Path,
    run_name: str,
    rows: list[dict[str, Any]],
    candidate_df: pd.DataFrame,
    parse_errors: Counter[tuple[str, str]],
    response_rows_seen: int,
    selected_response_count: int,
) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("evaluation report requires at least one request row")
    task_accuracy = accuracy_table(df, ["task"])
    family_accuracy = accuracy_table(df, ["dataset_family"])
    information_accuracy = accuracy_table(df, ["information"])
    slice_tables = {
        "dataset_family": frame_to_records(family_accuracy),
        "task": frame_to_records(task_accuracy),
        "information": frame_to_records(information_accuracy),
        "a_count": frame_to_records(accuracy_table(df, ["a_count"])),
        "b_count": frame_to_records(accuracy_table(df, ["b_count"])),
        "constraint_cell": frame_to_records(accuracy_table(df, ["a_count", "b_count"])),
        "dataset_family/task": frame_to_records(accuracy_table(df, ["dataset_family", "task"])),
        "task/information": frame_to_records(accuracy_table(df, ["task", "information"])),
        "task/constraint_cell": frame_to_records(accuracy_table(df, ["task", "a_count", "b_count"])),
    }
    t2 = df[df["task"] == "T2"]
    t2_gold = accuracy_table(t2, ["gold_set_size"]) if len(t2) else pd.DataFrame()
    binary_matrices: dict[str, dict[str, Any]] = {}
    for task in ["T1", "T3"]:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        binary_matrices[task] = {
            metric: matrix_payload(binary_matrix(task_df, metric))
            for metric in ["positive_precision", "positive_recall", "negative_precision", "negative_recall"]
        }
    candidate_summary = pd.DataFrame()
    candidate_by_b = pd.DataFrame()
    if not candidate_df.empty:
        candidate_summary = (
            candidate_df.groupby("category")
            .agg(total=("selected", "size"), selected=("selected", "sum"), selected_rate=("selected", "mean"))
            .reset_index()
        )
        candidate_by_b = (
            candidate_df.groupby(["b_count", "category"])
            .agg(total=("selected", "size"), selected=("selected", "sum"), selected_rate=("selected", "mean"))
            .reset_index()
        )
    total = int(len(df))
    correct = int(df["correct"].sum())
    weight_total = float(df["sampling_weight"].sum())
    weighted_correct = float(df.loc[df["correct"], "sampling_weight"].sum())
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": _safe_rate(correct, total),
            "weight_total": weight_total,
            "weighted_correct": weighted_correct,
            "weighted_accuracy": _safe_rate(weighted_correct, weight_total),
            "response_rows_seen": int(response_rows_seen),
            "selected_response_count": int(selected_response_count),
            "valid_answer_count": int(df["format_valid"].sum()),
            "invalid_or_error_count": int((~df["format_valid"]).sum()),
            "provider_error_count": int(df["provider_error"].sum()),
        },
        "parse_errors": [{"task": task, "error": error, "count": count} for (task, error), count in sorted(parse_errors.items())],
        "task_accuracy": frame_to_records(task_accuracy),
        "dataset_family_accuracy": frame_to_records(family_accuracy),
        "information_accuracy": frame_to_records(information_accuracy),
        "accuracy_slices": slice_tables,
        "overall_accuracy_matrix": matrix_payload(accuracy_matrix(df)),
        "t1_binary_metrics": binary_metrics(df[df["task"] == "T1"]),
        "t3_binary_metrics": binary_metrics(df[df["task"] == "T3"]),
        "binary_matrices": binary_matrices,
        "t2_exact_set_accuracy": _safe_rate(int(t2["correct"].sum()), len(t2)) if len(t2) else None,
        "t2_accuracy_by_gold_set_size": frame_to_records(t2_gold),
        "t2_predicted_size_distribution": t2_size_distribution_payload(t2) if len(t2) else {},
        "t2_candidate_summary": frame_to_records(candidate_summary),
        "t2_candidate_by_b_count": frame_to_records(candidate_by_b),
    }


def _notebook_cells(run_name: str, run_dir: Path, metrics: dict[str, Any]) -> list[Any]:
    title = run_name.replace("_", " ")
    cells: list[Any] = [
        nbf.v4.new_markdown_cell(
            f"# Evaluation Report: {title}\n\n"
            f"- Generated: `{metrics['generated_at']}`\n"
            f"- Source run: `{run_dir}`\n"
            f"- This notebook is generated from saved `requests.jsonl` and `responses.jsonl`; it does not call any provider."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "METRICS_PATH = Path('evaluation_metrics.json')\n"
            "metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))\n\n"
            "def pct(value):\n"
            "    return '' if value is None else f'{100 * float(value):.2f}%'\n\n"
            "def matrix_from_payload(payload):\n"
            "    return pd.DataFrame(payload['values'], index=payload['index'], columns=payload['columns'], dtype=float)\n\n"
            "def display_pct_matrix(payload):\n"
            "    matrix = matrix_from_payload(payload)\n"
            "    display(matrix.map(pct))\n\n"
            "def plot_heatmap(payload, title, cmap='viridis', vmin=0.10, vmax=0.90):\n"
            "    matrix = matrix_from_payload(payload)\n"
            "    values = matrix.to_numpy(dtype=float)\n"
            "    fig, ax = plt.subplots(figsize=(8.5, 5.2))\n"
            "    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')\n"
            "    ax.set_title(title)\n"
            "    ax.set_xlabel('b_count')\n"
            "    ax.set_ylabel('a_count')\n"
            "    ax.set_xticks(range(len(matrix.columns)))\n"
            "    ax.set_xticklabels([f'B{int(col)}' for col in matrix.columns])\n"
            "    ax.set_yticks(range(len(matrix.index)))\n"
            "    ax.set_yticklabels([f'A{int(row)}' for row in matrix.index])\n"
            "    for i in range(values.shape[0]):\n"
            "        for j in range(values.shape[1]):\n"
            "            value = values[i, j]\n"
            "            if np.isnan(value):\n"
            "                text = ''\n"
            "                norm = 0\n"
            "            else:\n"
            "                text = f'{100 * value:.1f}'\n"
            "                norm = (value - vmin) / (vmax - vmin)\n"
            "            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if norm < 0.45 else 'black')\n"
            "    cbar = fig.colorbar(im, ax=ax)\n"
            "    cbar.ax.set_ylabel('Value (%)')\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n\n"
            "def plot_t2_predicted_size_distribution(payload, title):\n"
            "    labels = payload['predicted_size_labels']\n"
            "    shares = pd.DataFrame(payload['shares'], index=payload['gold_set_size'], columns=labels)\n"
            "    colors = {'0': '#8da0cb', '1': '#66c2a5', '2': '#fc8d62', '3': '#e78ac3', '4': '#a6d854', 'invalid': '#999999'}\n"
            "    row_order = [2, 1, 0]\n"
            "    fig, ax = plt.subplots(figsize=(12.5, 5.0))\n"
            "    y_pos = np.arange(len(row_order))\n"
            "    left = np.zeros(len(row_order))\n"
            "    for label in labels:\n"
            "        values = shares.loc[row_order, label].fillna(0).to_numpy(dtype=float) * 100\n"
            "        legend_label = f'Pred {label}' if label != 'invalid' else 'Invalid'\n"
            "        ax.barh(y_pos, values, left=left, color=colors[label], edgecolor='none', height=0.52, label=legend_label)\n"
            "        for i, value in enumerate(values):\n"
            "            if value >= 7:\n"
            "                color = 'black' if label in {'1', '4'} else 'white'\n"
            "                ax.text(left[i] + value / 2, y_pos[i], f'{value:.0f}%', ha='center', va='center', fontsize=10, color=color)\n"
            "        left += values\n"
            "    ax.set_title(title, fontsize=15)\n"
            "    ax.set_xlabel('Share of T2 responses (%)')\n"
            "    ax.set_xlim(0, 100)\n"
            "    ax.set_yticks(y_pos)\n"
            "    ax.set_yticklabels([f'Gold {value}' for value in row_order])\n"
            "    ax.grid(axis='x', color='#bbbbbb', linewidth=0.8, alpha=0.8)\n"
            "    ax.set_axisbelow(True)\n"
            "    for spine in ['top', 'right']:\n"
            "        ax.spines[spine].set_visible(False)\n"
            "    ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
        ),
    ]
    overall = metrics["overall"]
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Run Summary\n\n"
            + _markdown_table(
                ["field", "value"],
                [
                    ["total", _number(overall["total"])],
                    ["correct", _number(overall["correct"])],
                    ["overall_accuracy", _pct(overall["accuracy"])],
                    ["response_rows_seen", _number(overall["response_rows_seen"])],
                    ["selected_response_count", _number(overall["selected_response_count"])],
                    ["valid_answer_count", _number(overall["valid_answer_count"])],
                    ["invalid_or_error_count", _number(overall["invalid_or_error_count"])],
                    ["provider_error_count", _number(overall["provider_error_count"])],
                ],
            )
        )
    )
    task_accuracy = pd.DataFrame(metrics["task_accuracy"])
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Overall and Task-Level Accuracy\n\n"
            + _markdown_table(
                ["task", "total", "correct", "accuracy"],
                [[row["task"], _number(row["total"]), _number(row["correct"]), _pct(row["accuracy"])] for _, row in task_accuracy.iterrows()],
            )
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Overall Accuracy by A x B Cell"))
    cells.append(
        nbf.v4.new_code_cell(
            "display_pct_matrix(metrics['overall_accuracy_matrix'])\n"
            "plot_heatmap(metrics['overall_accuracy_matrix'], 'Overall Accuracy by A x B')"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## T1 / T3 Binary Diagnostics\n\n"
            "Positive class is `True`; negative class is `False`. The four matrices are positive precision, positive recall, "
            "negative precision, and negative recall."
        )
    )
    for task in ["T1", "T3"]:
        if task not in metrics["binary_matrices"]:
            continue
        cells.append(nbf.v4.new_markdown_cell(f"### {task} A x B Matrices"))
        for metric in ["positive_precision", "positive_recall", "negative_precision", "negative_recall"]:
            cells.append(
                nbf.v4.new_code_cell(
                    f"display_pct_matrix(metrics['binary_matrices']['{task}']['{metric}'])\n"
                    f"plot_heatmap(metrics['binary_matrices']['{task}']['{metric}'], '{task} {metric}')"
                )
            )
    if metrics.get("t2_accuracy_by_gold_set_size"):
        t2_gold = pd.DataFrame(metrics["t2_accuracy_by_gold_set_size"])
        cells.append(
            nbf.v4.new_markdown_cell(
                "## T2 Diagnostics\n\n"
                "T2 is evaluated by exact-set accuracy at question level.\n\n"
                "### Accuracy by Gold Set Size\n\n"
                + _markdown_table(
                    ["gold_set_size", "total", "correct", "accuracy"],
                    [
                        [int(row["gold_set_size"]), _number(row["total"]), _number(row["correct"]), _pct(row["accuracy"])]
                        for _, row in t2_gold.iterrows()
                    ],
                )
            )
        )
        cells.append(
            nbf.v4.new_code_cell(
                f"plot_t2_predicted_size_distribution(metrics['t2_predicted_size_distribution'], '{title.title()}: Predicted Set Size by Gold Set Size')"
            )
        )
    parse_rows = metrics["parse_errors"]
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Error / Invalid Output Summary\n\n"
            + (
                _markdown_table(["task", "error", "count"], [[row["task"], row["error"], _number(row["count"])] for row in parse_rows])
                if parse_rows
                else "No parse or provider errors were observed."
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "# Regenerate this report from the run directory:\n"
            "# boollogic report --run <run_dir>\n"
        )
    )
    return cells


def write_evaluation_notebook(
    paths: EvaluationReportPaths,
    run_name: str,
    run_dir: Path,
    metrics: dict[str, Any],
    execute: bool = True,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = _notebook_cells(run_name, run_dir, metrics)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    if execute:
        nb = NotebookClient(nb, timeout=600, kernel_name="python3").execute(cwd=str(paths.output_dir))
    paths.notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def generate_evaluation_report(
    run_dir: Path,
    run_name: str | None = None,
    output_dir: Path | None = None,
    answer_parse_char_limit: int = 2000,
    execute_notebook: bool = True,
) -> EvaluationReportPaths:
    run_dir = run_dir.resolve()
    output_dir = ensure_dir((output_dir or run_dir).resolve())
    paths = EvaluationReportPaths(
        output_dir=output_dir,
        rows_path=output_dir / "evaluation_rows.jsonl",
        metrics_path=output_dir / "evaluation_metrics.json",
        notebook_path=output_dir / "evaluation_report.ipynb",
    )
    request_order, request_meta = load_requests(run_dir)
    selected, response_rows_seen = select_responses(run_dir, request_meta)
    rows, parse_errors, t2_rows = build_evaluation_rows(
        request_order,
        request_meta,
        selected,
        answer_parse_char_limit=answer_parse_char_limit,
    )
    candidate_df = build_t2_candidate_frame(run_dir, t2_rows)
    metrics = build_evaluation_metrics(
        run_dir,
        run_name or run_dir.name,
        rows,
        candidate_df,
        parse_errors,
        response_rows_seen,
        len(selected),
    )
    write_jsonl(paths.rows_path, rows)
    write_json(paths.metrics_path, metrics)
    write_evaluation_notebook(paths, run_name or run_dir.name, run_dir, metrics, execute=execute_notebook)
    return paths
