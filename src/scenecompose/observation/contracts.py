from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ObservationRegion:
    observation_id: str
    source_scene: str
    source_sha256: str
    anchor_floor: Vector3
    anchor_eye: Vector3
    reference_forward: Vector3
    reference_right: Vector3
    up: Vector3
    radius_m: float
    horizontal_angle_deg: float
    vertical_range_world: tuple[float, float]
    camera_motion_radius_m: float
    context_margin_m: float
    partition_geometry: str
    source_faces: str
    core_triangle_count: int
    context_triangle_count: int
    quality: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


def observation_id(
    source_sha256: str,
    anchor: Vector3,
    forward: Vector3,
    radius_m: float,
    horizontal_angle_deg: float,
    config_digest: str = "",
) -> str:
    payload = {
        "source": source_sha256,
        "anchor": [round(value, 5) for value in anchor],
        "forward": [round(value, 6) for value in forward],
        "radius_m": round(radius_m, 5),
        "horizontal_angle_deg": round(horizontal_angle_deg, 5),
        "config_digest": config_digest,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"observation_{digest}"


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
