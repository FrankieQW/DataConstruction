from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObjectAsset:
    uid: str
    glb_path: Path
    relative_path: str
    annotation: dict[str, Any]


@dataclass(frozen=True)
class Classification:
    decision: str
    canonical_class: str | None
    placement_type: str | None
    support_classes: tuple[str, ...]
    target_dimension: str | None
    target_min_m: float | None
    target_max_m: float | None
    confidence: float
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

