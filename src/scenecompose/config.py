from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PartitionConfig:
    schema_version: int = 1
    max_triangles_per_region: int = 1_500_000
    max_estimated_points_per_region: int = 250_000
    max_xy_extent_m: float = 8.0
    halo_m: float = 0.75
    min_split_extent_m: float = 0.25
    point_density_per_m2: float = 2_500.0
    meters_per_blender_unit: float | None = None
    export_format: str = "blend"
    include_materials: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "PartitionConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown partition config keys: {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported config schema_version: {self.schema_version}")
        positive = {
            "max_triangles_per_region": self.max_triangles_per_region,
            "max_estimated_points_per_region": self.max_estimated_points_per_region,
            "max_xy_extent_m": self.max_xy_extent_m,
            "min_split_extent_m": self.min_split_extent_m,
            "point_density_per_m2": self.point_density_per_m2,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if self.halo_m < 0:
            raise ValueError("halo_m must be non-negative")
        if self.meters_per_blender_unit is not None and self.meters_per_blender_unit <= 0:
            raise ValueError("meters_per_blender_unit must be positive when set")
        if self.export_format not in {"glb", "blend"}:
            raise ValueError("export_format must be 'glb' or 'blend'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
