from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CompositionConfig
from .contracts import write_json_atomic


def extract_support_patches(
    observation_dir: Path, segmentation_config: Path,
    config: CompositionConfig, output: Path,
) -> list[dict[str, object]]:
    geometry_path = observation_dir / "segmentation" / "geometry" / "geometry.npz"
    labels_path = observation_dir / "segmentation" / "fusion" / "face_labels.npz"
    if not geometry_path.is_file() or not labels_path.is_file():
        raise ValueError(f"Incomplete segmentation artifacts: {observation_dir}")
    vocabulary = json.loads(segmentation_config.read_text(encoding="utf-8"))["vocabulary"]
    names = {int(item["id"]): str(item["name"]) for item in vocabulary}
    geometry = np.load(geometry_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    vertices = np.asarray(geometry["vertices"], dtype=np.float64)
    triangles = np.asarray(geometry["triangles"], dtype=np.int64)
    semantic = np.asarray(labels["semantic_ids"], dtype=np.int64)
    instances = np.asarray(labels["instance_ids"], dtype=np.int64)
    visibility = np.asarray(labels["visibility_count"], dtype=np.int64)
    observed = np.asarray(labels["is_observed"], dtype=bool)
    core = np.asarray(labels["is_core"], dtype=bool)
    if len(triangles) != len(semantic):
        raise ValueError("geometry triangles and face_labels have different lengths")
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-12)
    allowed_ids = {key for key, value in names.items()
                   if value in config.support.allowed_support_classes}
    slope_cos = np.cos(np.deg2rad(config.support.maximum_slope_deg))
    valid = (np.isin(semantic, list(allowed_ids)) & observed & core &
             (visibility >= config.support.minimum_visible_views) &
             (normals[:, 2] >= slope_cos) & (twice_area > 1e-12))
    patches: list[dict[str, object]] = []
    for semantic_id in sorted(allowed_ids):
        instance_values = np.unique(instances[valid & (semantic == semantic_id)])
        for instance_id in instance_values:
            indexes = np.flatnonzero(valid & (semantic == semantic_id) & (instances == instance_id))
            for component in _connected_components(indexes, triangles):
                area = float(twice_area[component].sum() * 0.5)
                if area < config.support.minimum_patch_area_m2:
                    continue
                component_points = points[component].reshape(-1, 3)
                weights = twice_area[component]
                centroids = points[component].mean(axis=1)
                centroid = np.average(centroids, axis=0, weights=weights)
                order = np.argsort(-weights, kind="stable")
                candidate_centroids = centroids[
                    order[:max(config.placement.positions_per_surface * 4, 32)]
                ]
                boundary = _boundary_segments(component, triangles, vertices)
                candidates = [
                    {"position": candidate.tolist(),
                     "boundary_clearance_m": _boundary_clearance(candidate, boundary)}
                    for candidate in candidate_centroids
                ]
                lower = component_points.min(axis=0)
                upper = component_points.max(axis=0)
                patches.append({
                    "patch_id": f"support_{len(patches):05d}",
                    "semantic_id": int(semantic_id), "semantic_class": names[semantic_id],
                    "instance_id": int(instance_id), "triangle_count": len(component),
                    "area_m2": area, "centroid": centroid.tolist(),
                    "bounds_min": lower.tolist(), "bounds_max": upper.tolist(),
                    "mean_normal": np.average(normals[component], axis=0, weights=weights).tolist(),
                    "candidate_points": candidates,
                })
    patches.sort(key=lambda item: (-float(item["area_m2"]), str(item["patch_id"])))
    write_json_atomic(output, {"schema_version": 1, "observation": str(observation_dir),
                               "patches": patches})
    return patches


def _connected_components(indexes: np.ndarray, triangles: np.ndarray) -> list[np.ndarray]:
    if not len(indexes):
        return []
    parent = list(range(len(indexes)))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    owners: dict[int, int] = {}
    for local, triangle_index in enumerate(indexes):
        for vertex in triangles[int(triangle_index)]:
            previous = owners.setdefault(int(vertex), local)
            union(local, previous)
    groups: dict[int, list[int]] = {}
    for local, triangle_index in enumerate(indexes):
        groups.setdefault(find(local), []).append(int(triangle_index))
    return [np.asarray(value, dtype=np.int64) for value in groups.values()]


def _boundary_segments(
    component: np.ndarray, triangles: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    counts: dict[tuple[int, int], int] = {}
    for triangle_index in component:
        a, b, c = map(int, triangles[int(triangle_index)])
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (min(left, right), max(left, right))
            counts[edge] = counts.get(edge, 0) + 1
    edges = [edge for edge, count in counts.items() if count == 1]
    if not edges:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.asarray([[vertices[a, :2], vertices[b, :2]] for a, b in edges])


def _boundary_clearance(point: np.ndarray, segments: np.ndarray) -> float:
    if not len(segments):
        return 0.0
    start = segments[:, 0]
    delta = segments[:, 1] - start
    denominator = np.maximum(np.einsum("ij,ij->i", delta, delta), 1e-12)
    amount = np.clip(np.einsum("ij,ij->i", point[:2] - start, delta) / denominator, 0, 1)
    closest = start + amount[:, None] * delta
    return float(np.linalg.norm(closest - point[:2], axis=1).min())
