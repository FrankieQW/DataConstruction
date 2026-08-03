from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .artifacts import decode_mask_rle
from .config import FusionConfig


@dataclass(frozen=True)
class LiftedObservation:
    mask_id: str
    view_id: str
    class_id: int
    score: float
    point_indices: np.ndarray
    point_weights: np.ndarray
    centroid: np.ndarray


def read_depth(path: Path) -> np.ndarray:
    try:
        import Imath
        import OpenEXR
        source = OpenEXR.InputFile(str(path))
        window = source.header()["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        channel = "R" if "R" in source.header()["channels"] else next(iter(source.header()["channels"]))
        depth = np.frombuffer(
            source.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32
        ).reshape((height, width))
        source.close()
    except (ImportError, OSError, ValueError):
        import imageio.v3 as iio
        depth = np.asarray(iio.imread(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def unproject_blender_depth(mask: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != depth.shape:
        raise ValueError(f"Mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
    rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if not len(rows):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    rays = np.column_stack(((columns + 0.5 - cx) / fx, -(rows + 0.5 - cy) / fy, -np.ones(len(rows))))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    distance = depth[rows, columns]
    local = rays * distance[:, None]
    homogeneous = np.column_stack((local, np.ones(len(local))))
    world = homogeneous @ camera_to_world.T
    return world[:, :3].astype(np.float32), distance.astype(np.float32)


def lift_mask(
    record: dict[str, object], camera: dict[str, object], depth: np.ndarray,
    scene_points: np.ndarray, config: FusionConfig,
) -> LiftedObservation | None:
    from scipy.spatial import cKDTree
    mask = decode_mask_rle(record["rle"], int(record["height"]), int(record["width"]))
    world, _ = unproject_blender_depth(mask, depth, np.asarray(camera["intrinsic"]), np.asarray(camera["camera_to_world"]))
    if not len(world):
        return None
    tree = cKDTree(scene_points)
    distances, indices = tree.query(world, k=1, distance_upper_bound=config.lift_radius_m, workers=-1)
    valid = np.isfinite(distances) & (indices < len(scene_points))
    if not np.any(valid):
        return None
    indices = indices[valid].astype(np.int64)
    distances = distances[valid]
    unique, inverse = np.unique(indices, return_inverse=True)
    weights = np.zeros(len(unique), dtype=np.float32)
    np.maximum.at(weights, inverse, np.exp(-np.square(distances / max(config.lift_radius_m, 1e-8))).astype(np.float32))
    return LiftedObservation(
        str(record["mask_id"]), str(record["view_id"]), int(record["class_id"]),
        float(record["score"]), unique, weights, scene_points[unique].mean(axis=0),
    )


def lift_all_masks(mask_index: Path, camera_dir: Path, depth_dir: Path, samples_path: Path, config: FusionConfig) -> list[LiftedObservation]:
    with np.load(samples_path, allow_pickle=False) as archive:
        points = archive["points"].astype(np.float32)
    payload = json.loads(mask_index.read_text(encoding="utf-8"))
    cameras = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in camera_dir.glob("*.json")}
    depth_cache: dict[str, np.ndarray] = {}
    observations: list[LiftedObservation] = []
    for record in payload["masks"]:
        view_id = record["view_id"]
        camera = cameras.get(view_id)
        if camera is None:
            raise ValueError(f"SAM3 mask references missing camera: {view_id}")
        if view_id not in depth_cache:
            matches = sorted(depth_dir.glob(f"{view_id}_*.exr"))
            if len(matches) != 1:
                raise ValueError(f"Expected one depth image for {view_id}, found {len(matches)}")
            depth_cache[view_id] = read_depth(matches[0])
        lifted = lift_mask(record, camera, depth_cache[view_id], points, config)
        if lifted is not None:
            observations.append(lifted)
    return observations
