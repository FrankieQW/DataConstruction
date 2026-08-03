from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import PartitionConfig
from .contracts import Bounds3D, IDENTITY_MATRIX_4X4, SceneRegion, uniform_scale_matrix

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


class _Node:
    __slots__ = ("indices", "bounds")

    def __init__(self, indices: IntArray, bounds: Bounds3D) -> None:
        self.indices = indices
        self.bounds = bounds


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _estimated_points(areas_m2: FloatArray, density: float) -> int:
    return int(np.ceil(float(np.sum(areas_m2, dtype=np.float64)) * density))


def _over_budget(
    triangle_count: int,
    estimated_points: int,
    bounds: Bounds3D,
    config: PartitionConfig,
) -> bool:
    extent_x, extent_y = bounds.xy_extent
    return (
        triangle_count > config.max_triangles_per_region
        or estimated_points > config.max_estimated_points_per_region
        or max(extent_x, extent_y) > config.max_xy_extent_m
    )


def _content_region_id(
    scene_fingerprint: str,
    bounds: Bounds3D,
    triangle_count: int,
) -> str:
    payload = {
        "scene": scene_fingerprint,
        "minimum": [round(value, 6) for value in bounds.minimum],
        "maximum": [round(value, 6) for value in bounds.maximum],
        "triangles": triangle_count,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"region_{digest}"


def _split_node(
    node: _Node,
    centroids_xy: FloatArray,
    config: PartitionConfig,
) -> tuple[_Node, _Node] | None:
    extent = np.asarray(node.bounds.xy_extent, dtype=np.float64)
    for axis_value in np.argsort(-extent, kind="stable"):
        axis = int(axis_value)
        if extent[axis] < 2.0 * config.min_split_extent_m:
            continue

        values = centroids_xy[node.indices, axis]
        unique_values, counts = np.unique(values, return_counts=True)
        if unique_values.size < 2:
            continue

        cumulative = np.cumsum(counts)
        split_unique_index = int(np.searchsorted(cumulative, values.size // 2, side="left"))
        split_unique_index = min(split_unique_index, unique_values.size - 2)
        left_value = float(unique_values[split_unique_index])
        right_value = float(unique_values[split_unique_index + 1])
        split_value = 0.5 * (left_value + right_value)

        lower = node.bounds.minimum[axis]
        upper = node.bounds.maximum[axis]
        if (
            split_value - lower < config.min_split_extent_m
            or upper - split_value < config.min_split_extent_m
        ):
            continue

        belongs_left = values < split_value
        if not np.any(belongs_left) or np.all(belongs_left):
            continue
        left_indices = node.indices[belongs_left]
        right_indices = node.indices[~belongs_left]

        left_min = list(node.bounds.minimum)
        left_max = list(node.bounds.maximum)
        right_min = list(node.bounds.minimum)
        right_max = list(node.bounds.maximum)
        left_max[axis] = split_value
        right_min[axis] = split_value
        return (
            _Node(left_indices, Bounds3D(tuple(left_min), tuple(left_max))),
            _Node(right_indices, Bounds3D(tuple(right_min), tuple(right_max))),
        )
    return None


def _inside_xy(centroids_xy: FloatArray, bounds: Bounds3D) -> NDArray[np.bool_]:
    return (
        (centroids_xy[:, 0] >= bounds.minimum[0])
        & (centroids_xy[:, 0] <= bounds.maximum[0])
        & (centroids_xy[:, 1] >= bounds.minimum[1])
        & (centroids_xy[:, 1] <= bounds.maximum[1])
    )


def plan_regions(
    centroids_xy_m: FloatArray,
    triangle_areas_m2: FloatArray,
    scene_bounds_m: Bounds3D,
    scene_fingerprint: str,
    config: PartitionConfig,
    *,
    meters_per_source_unit: float = 1.0,
) -> list[SceneRegion]:
    """Create full-height, deterministic XY regions for one scene."""
    config.validate()
    centroids = np.asarray(centroids_xy_m, dtype=np.float64)
    areas = np.asarray(triangle_areas_m2, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError("centroids_xy_m must have shape (N, 2)")
    if areas.ndim != 1 or areas.shape[0] != centroids.shape[0]:
        raise ValueError("triangle_areas_m2 must have shape (N,)")
    if centroids.shape[0] == 0:
        raise ValueError("Cannot partition a scene without triangles")
    if not np.isfinite(centroids).all() or not np.isfinite(areas).all():
        raise ValueError("Triangle geometry contains non-finite values")
    if np.any(areas < 0):
        raise ValueError("Triangle areas cannot be negative")
    if meters_per_source_unit <= 0:
        raise ValueError("meters_per_source_unit must be positive")

    all_indices = np.arange(centroids.shape[0], dtype=np.int64)
    total_points = _estimated_points(areas, config.point_density_per_m2)
    if not _over_budget(all_indices.size, total_points, scene_bounds_m, config):
        return [
            SceneRegion(
                region_id="region_full",
                is_identity=True,
                core_bounds=scene_bounds_m,
                context_bounds=scene_bounds_m,
                triangle_count=int(all_indices.size),
                context_triangle_count=int(all_indices.size),
                estimated_point_count=total_points,
                surface_area_m2=float(np.sum(areas, dtype=np.float64)),
                over_budget=False,
                world_from_region=uniform_scale_matrix(meters_per_source_unit),
            )
        ]

    leaves: list[_Node] = []
    pending = [_Node(all_indices, scene_bounds_m)]
    while pending:
        node = pending.pop()
        node_points = _estimated_points(areas[node.indices], config.point_density_per_m2)
        if not _over_budget(node.indices.size, node_points, node.bounds, config):
            leaves.append(node)
            continue
        children = _split_node(node, centroids, config)
        if children is None:
            leaves.append(node)
            continue
        pending.extend(reversed(children))

    leaves.sort(
        key=lambda node: (
            0.5 * (node.bounds.minimum[0] + node.bounds.maximum[0]),
            0.5 * (node.bounds.minimum[1] + node.bounds.maximum[1]),
        )
    )
    regions: list[SceneRegion] = []
    for node in leaves:
        points = _estimated_points(areas[node.indices], config.point_density_per_m2)
        exceeds = _over_budget(node.indices.size, points, node.bounds, config)
        context_bounds = node.bounds.expanded_xy(config.halo_m, scene_bounds_m)
        context_count = int(np.count_nonzero(_inside_xy(centroids, context_bounds)))
        warnings = (
            ("Region remains over budget because no valid XY split exists.",)
            if exceeds
            else ()
        )
        regions.append(
            SceneRegion(
                region_id=_content_region_id(
                    scene_fingerprint, node.bounds, int(node.indices.size)
                ),
                is_identity=False,
                core_bounds=node.bounds,
                context_bounds=context_bounds,
                triangle_count=int(node.indices.size),
                context_triangle_count=context_count,
                estimated_point_count=points,
                surface_area_m2=float(np.sum(areas[node.indices], dtype=np.float64)),
                over_budget=exceeds,
                world_from_region=IDENTITY_MATRIX_4X4,
                warnings=warnings,
            )
        )
    return regions


def write_regions_json(
    path: Path,
    *,
    scene_id: str,
    source_path: Path,
    source_sha256: str,
    config: PartitionConfig,
    regions: list[SceneRegion],
    meters_per_blender_unit: float,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "scene_id": scene_id,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "meters_per_blender_unit": meters_per_blender_unit,
        "config": config.to_dict(),
        "region_count": len(regions),
        "sources": sources or [],
        "regions": [region.to_dict() for region in regions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
