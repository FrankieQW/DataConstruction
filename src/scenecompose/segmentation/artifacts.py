from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_npz(path: Path, required: dict[str, tuple[int | None, ...]]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    missing = sorted(set(required) - set(result))
    if missing:
        raise ValueError(f"Missing arrays in {path}: {', '.join(missing)}")
    for key, expected in required.items():
        value = result[key]
        if value.ndim != len(expected):
            raise ValueError(f"{path}:{key} expected {len(expected)} dimensions, got {value.ndim}")
        for actual_size, expected_size in zip(value.shape, expected):
            if expected_size is not None and actual_size != expected_size:
                raise ValueError(f"{path}:{key} has invalid shape {value.shape}")
    return result


def encode_mask_rle(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        bit = int(value != 0)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return counts


def decode_mask_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    values: list[np.ndarray] = []
    bit = 0
    for count in counts:
        if count < 0:
            raise ValueError("RLE counts must be non-negative")
        values.append(np.full(count, bit, dtype=np.uint8))
        bit = 1 - bit
    flat = np.concatenate(values) if values else np.empty(0, dtype=np.uint8)
    if flat.size != height * width:
        raise ValueError("RLE size does not match mask dimensions")
    return flat.reshape((height, width), order="F").astype(bool)


@dataclass(frozen=True)
class MaskObservation:
    mask_id: str
    view_id: str
    class_id: int
    prompt: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    height: int
    width: int
    rle: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id, "view_id": self.view_id, "class_id": self.class_id,
            "prompt": self.prompt, "score": self.score, "box_xyxy": list(self.box_xyxy),
            "height": self.height, "width": self.width, "rle": list(self.rle),
        }
