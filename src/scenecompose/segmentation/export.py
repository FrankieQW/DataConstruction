from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import save_json_atomic
from .config import SegmentationConfig
from .fusion import FusedInstance


def _color(value: int) -> tuple[int, int, int]:
    if value < 0:
        return (96, 96, 96)
    rng = np.random.default_rng(value + 1701)
    return tuple(int(channel) for channel in rng.integers(48, 240, size=3))


def write_colored_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.asarray([_color(int(label)) for label in labels], dtype=np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    packed = np.empty(len(points), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        packed[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        packed[name] = colors[:, axis]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        packed.tofile(stream)
    temporary.replace(path)


def _runs(indices: np.ndarray) -> list[list[int]]:
    if not len(indices):
        return []
    values = np.unique(indices)
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    groups = np.split(values, breaks)
    return [[int(group[0]), int(group[-1])] for group in groups]


def export_results(point_labels_path: Path, instances: list[FusedInstance], config: SegmentationConfig, output_root: Path) -> list[Path]:
    with np.load(point_labels_path, allow_pickle=False) as archive:
        points = archive["points"]
        semantic = archive["semantic_ids"]
        instance_ids = archive["instance_ids"]
        visibility = archive["visibility_count"] if "visibility_count" in archive else np.zeros(len(points), dtype=np.int32)
        is_core = archive["is_core"] if "is_core" in archive else np.ones(len(points), dtype=bool)
    label_names = {item.id: item.name for item in config.vocabulary}
    records = []
    for item in instances:
        selected = points[item.point_indices]
        center = selected.mean(axis=0)
        minimum, maximum = selected.min(axis=0), selected.max(axis=0)
        centered = selected - center
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        local = centered @ axes.T
        selected_visibility = visibility[item.point_indices]
        selected_core = is_core[item.point_indices]
        records.append({
            "instance_id": item.instance_id, "class_id": item.class_id,
            "label": label_names[item.class_id], "confidence": item.confidence,
            "point_index_runs": _runs(item.point_indices), "face_index_runs": _runs(item.face_indices),
            "centroid": center.tolist(), "aabb": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
            "obb": {"center": center.tolist(), "axes": axes.tolist(), "minimum": local.min(axis=0).tolist(), "maximum": local.max(axis=0).tolist()},
            "view_ids": list(item.view_ids),
            "visible_point_count": int(np.count_nonzero(selected_visibility)),
            "visible_point_ratio": float(np.count_nonzero(selected_visibility) / len(selected)),
            "core_point_ratio": float(np.count_nonzero(selected_core) / len(selected)),
        })
    instance_path = output_root / "fusion" / "instances.json"
    save_json_atomic(instance_path, {"schema_version": 1, "instances": records})
    semantic_path = output_root / "visualization" / "semantic.ply"
    instance_vis_path = output_root / "visualization" / "instances.ply"
    write_colored_ply(semantic_path, points, semantic)
    write_colored_ply(instance_vis_path, points, instance_ids)
    return [instance_path, semantic_path, instance_vis_path]
