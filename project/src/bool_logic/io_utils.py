from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(to_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(to_plain(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    last_error: OSError | None = None
    for attempt in range(20):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        temp_path.unlink(missing_ok=True)
    finally:
        if last_error is not None:
            raise last_error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_plain(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
