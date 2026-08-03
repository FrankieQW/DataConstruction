from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Bounds3D:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def xy_extent(self) -> tuple[float, float]:
        return (
            self.maximum[0] - self.minimum[0],
            self.maximum[1] - self.minimum[1],
        )

    def expanded_xy(self, margin_m: float, scene_bounds: "Bounds3D") -> "Bounds3D":
        return Bounds3D(
            minimum=(
                max(scene_bounds.minimum[0], self.minimum[0] - margin_m),
                max(scene_bounds.minimum[1], self.minimum[1] - margin_m),
                scene_bounds.minimum[2],
            ),
            maximum=(
                min(scene_bounds.maximum[0], self.maximum[0] + margin_m),
                min(scene_bounds.maximum[1], self.maximum[1] + margin_m),
                scene_bounds.maximum[2],
            ),
        )


@dataclass(frozen=True)
class SceneRegion:
    region_id: str
    is_identity: bool
    core_bounds: Bounds3D
    context_bounds: Bounds3D
    triangle_count: int
    context_triangle_count: int
    estimated_point_count: int
    surface_area_m2: float
    over_budget: bool
    world_from_region: tuple[float, ...]
    geometry_path: str | None = None
    source_faces_path: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IDENTITY_MATRIX_4X4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def uniform_scale_matrix(scale: float) -> tuple[float, ...]:
    return (
        scale, 0.0, 0.0, 0.0,
        0.0, scale, 0.0, 0.0,
        0.0, 0.0, scale, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
