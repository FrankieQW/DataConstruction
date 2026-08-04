from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObservationPartitionConfig:
    schema_version: int = 1
    observations_per_scene: int = 8
    seed: int = 20260804
    meters_per_blender_unit: float | None = None
    up_axis: str = "Z"
    floor_max_slope_deg: float = 15.0
    floor_height_band_m: float = 0.25
    floor_component_cell_m: float = 0.5
    min_floor_component_area_m2: float = 2.0
    min_anchor_spacing_m: float = 2.0
    anchor_attempts_per_output: int = 64
    observer_height_m: float = 1.6
    min_head_clearance_m: float = 1.8
    min_body_clearance_radius_m: float = 0.35
    body_clearance_ray_count: int = 16
    radius_m: float = 6.0
    horizontal_angle_deg: float = 100.0
    vertical_below_anchor_m: float = 0.25
    vertical_above_anchor_m: float = 4.0
    camera_motion_radius_m: float = 1.0
    context_margin_m: float = 0.5
    direction_trials: int = 8
    minimum_core_triangles: int = 500
    export_format: str = "glb"
    include_materials: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "ObservationPartitionConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown observation partition config keys: {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if self.up_axis != "Z":
            raise ValueError("The first implementation supports only Z-up scenes")
        positive = {
            "observations_per_scene": self.observations_per_scene,
            "floor_height_band_m": self.floor_height_band_m,
            "floor_component_cell_m": self.floor_component_cell_m,
            "min_floor_component_area_m2": self.min_floor_component_area_m2,
            "min_anchor_spacing_m": self.min_anchor_spacing_m,
            "anchor_attempts_per_output": self.anchor_attempts_per_output,
            "observer_height_m": self.observer_height_m,
            "min_head_clearance_m": self.min_head_clearance_m,
            "min_body_clearance_radius_m": self.min_body_clearance_radius_m,
            "body_clearance_ray_count": self.body_clearance_ray_count,
            "radius_m": self.radius_m,
            "horizontal_angle_deg": self.horizontal_angle_deg,
            "vertical_above_anchor_m": self.vertical_above_anchor_m,
            "direction_trials": self.direction_trials,
            "minimum_core_triangles": self.minimum_core_triangles,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if self.meters_per_blender_unit is not None and self.meters_per_blender_unit <= 0:
            raise ValueError("meters_per_blender_unit must be positive when set")
        if not 0.0 <= self.floor_max_slope_deg < 90.0:
            raise ValueError("floor_max_slope_deg must be in [0, 90)")
        if not 1.0 < self.horizontal_angle_deg < 179.0:
            raise ValueError("horizontal_angle_deg must be between 1 and 179")
        if self.vertical_below_anchor_m < 0 or self.camera_motion_radius_m < 0 or self.context_margin_m < 0:
            raise ValueError("vertical_below_anchor_m, camera_motion_radius_m and context_margin_m must be non-negative")
        if self.export_format != "glb":
            raise ValueError("Observation partitions currently support only export_format='glb'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
