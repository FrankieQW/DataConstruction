from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import save_npz_atomic
from .association import AssociatedInstance
from .config import SegmentationConfig


@dataclass(frozen=True)
class FusedInstance:
    instance_id: int
    class_id: int
    point_indices: np.ndarray
    face_indices: np.ndarray
    confidence: float
    view_ids: tuple[str, ...]


def fuse_predictions(samples_path: Path, geometry_path: Path, scores_path: Path, instances: list[AssociatedInstance], config: SegmentationConfig, output_dir: Path) -> tuple[list[Path], list[FusedInstance]]:
    with np.load(samples_path, allow_pickle=False) as archive:
        points = archive["points"]
        triangle_ids = archive["triangle_ids"].astype(np.int64)
    with np.load(geometry_path, allow_pickle=False) as archive:
        face_count = len(archive["triangles"])
    with np.load(scores_path, allow_pickle=False) as archive:
        top_classes = archive["class_ids"].astype(np.int32)
        top_confidence = archive["confidence"].astype(np.float32)
        unknown = archive["unknown"].astype(bool)
    semantic = top_classes[:, 0].copy()
    semantic_confidence = top_confidence[:, 0].copy()
    semantic[unknown] = -1
    instance_ids = np.full(len(points), -1, dtype=np.int32)
    thing_ids = {item.id for item in config.vocabulary if item.kind == "thing"}
    candidates: list[tuple[AssociatedInstance, float]] = []
    for proposal in instances:
        point_indices = proposal.point_indices
        mosaic_match = top_classes[point_indices, 0] == proposal.class_id
        mosaic_score = float(top_confidence[point_indices, 0][mosaic_match].mean()) if np.any(mosaic_match) else 0.0
        view_score = min(1.0, len(proposal.view_ids) / max(1, config.fusion.min_instance_views * 2))
        fused_score = (
            config.fusion.mosaic_weight * mosaic_score
            + config.fusion.sam3_weight * proposal.score
            + config.fusion.view_weight * view_score
            + config.fusion.visibility_weight * min(1.0, len(point_indices) / max(1, config.fusion.min_instance_points * 4))
        )
        replace = fused_score > semantic_confidence[point_indices]
        chosen = point_indices[replace]
        semantic[chosen] = proposal.class_id
        semantic_confidence[chosen] = fused_score
        unknown[chosen] = False
        if proposal.class_id in thing_ids and len(point_indices):
            candidates.append((proposal, fused_score))
    occupied = np.zeros(len(points), dtype=bool)
    selected_instances: list[tuple[AssociatedInstance, float, np.ndarray, np.ndarray]] = []
    for proposal, fused_score in sorted(
        candidates,
        key=lambda item: (-item[1], item[0].class_id, int(item[0].point_indices.min())),
    ):
        available = proposal.point_indices[~occupied[proposal.point_indices]]
        if len(available) < config.fusion.min_instance_points:
            continue
        occupied[available] = True
        faces = _vote_instance_faces(triangle_ids, available, face_count, config.fusion.face_instance_vote)
        selected_instances.append((proposal, fused_score, available, faces))
    selected_instances.sort(
        key=lambda item: (item[0].class_id, tuple(points[item[2]].mean(axis=0)), int(item[2].min()))
    )
    accepted: list[FusedInstance] = []
    for instance_id, (proposal, fused_score, available, faces) in enumerate(selected_instances):
        instance_ids[available] = instance_id
        accepted.append(FusedInstance(instance_id, proposal.class_id, available, faces, fused_score, proposal.view_ids))
    face_semantic, face_confidence = _vote_face_semantics(triangle_ids, semantic, semantic_confidence, face_count, config.fusion.face_semantic_vote)
    face_instances = np.full(face_count, -1, dtype=np.int32)
    face_instance_scores = np.full(face_count, -np.inf, dtype=np.float32)
    for item in accepted:
        replace = item.confidence > face_instance_scores[item.face_indices]
        faces = item.face_indices[replace]
        face_instances[faces] = item.instance_id
        face_instance_scores[faces] = item.confidence
    output_dir.mkdir(parents=True, exist_ok=True)
    point_path = output_dir / "point_labels.npz"
    face_path = output_dir / "face_labels.npz"
    instance_path = output_dir / "fused_instances.npz"
    save_npz_atomic(point_path, points=points, semantic_ids=semantic, semantic_confidence=semantic_confidence, instance_ids=instance_ids, triangle_ids=triangle_ids, unknown=unknown)
    save_npz_atomic(face_path, semantic_ids=face_semantic, semantic_confidence=face_confidence, instance_ids=face_instances)
    _save_instances(instance_path, accepted)
    return [point_path, face_path, instance_path], accepted


def _save_instances(path: Path, instances: list[FusedInstance]) -> None:
    point_offsets = [0]
    face_offsets = [0]
    all_points: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    views: list[str] = []
    view_offsets = [0]
    for item in instances:
        all_points.append(item.point_indices)
        all_faces.append(item.face_indices)
        point_offsets.append(point_offsets[-1] + len(item.point_indices))
        face_offsets.append(face_offsets[-1] + len(item.face_indices))
        views.extend(item.view_ids)
        view_offsets.append(len(views))
    save_npz_atomic(
        path,
        instance_ids=np.asarray([item.instance_id for item in instances], dtype=np.int32),
        class_ids=np.asarray([item.class_id for item in instances], dtype=np.int32),
        confidence=np.asarray([item.confidence for item in instances], dtype=np.float32),
        point_indices=np.concatenate(all_points) if all_points else np.empty(0, dtype=np.int64),
        point_offsets=np.asarray(point_offsets, dtype=np.int64),
        face_indices=np.concatenate(all_faces) if all_faces else np.empty(0, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        view_ids=np.asarray(views, dtype=np.str_),
        view_offsets=np.asarray(view_offsets, dtype=np.int64),
    )


def load_fused_instances(path: Path) -> list[FusedInstance]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    result = []
    for index, instance_id in enumerate(data["instance_ids"]):
        point_slice = slice(data["point_offsets"][index], data["point_offsets"][index + 1])
        face_slice = slice(data["face_offsets"][index], data["face_offsets"][index + 1])
        view_slice = slice(data["view_offsets"][index], data["view_offsets"][index + 1])
        result.append(FusedInstance(
            int(instance_id), int(data["class_ids"][index]), data["point_indices"][point_slice],
            data["face_indices"][face_slice], float(data["confidence"][index]),
            tuple(str(value) for value in data["view_ids"][view_slice]),
        ))
    return result


def _vote_instance_faces(triangle_ids: np.ndarray, selected_points: np.ndarray, face_count: int, threshold: float) -> np.ndarray:
    total = np.bincount(triangle_ids, minlength=face_count)
    selected = np.bincount(triangle_ids[selected_points], minlength=face_count)
    ratio = np.divide(selected, total, out=np.zeros(face_count, dtype=np.float32), where=total > 0)
    return np.flatnonzero(ratio >= threshold).astype(np.int64)


def _vote_face_semantics(triangle_ids: np.ndarray, semantic: np.ndarray, confidence: np.ndarray, face_count: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    face_labels = np.full(face_count, -1, dtype=np.int32)
    face_confidence = np.zeros(face_count, dtype=np.float32)
    order = np.argsort(triangle_ids, kind="stable")
    sorted_faces = triangle_ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_faces)) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:end]
        face = int(triangle_ids[indices[0]])
        labels = semantic[indices]
        labels = labels[labels >= 0]
        if not len(labels):
            continue
        values, counts = np.unique(labels, return_counts=True)
        winner = int(values[np.argmax(counts)])
        ratio = counts.max() / len(indices)
        if ratio >= threshold:
            face_labels[face] = winner
            face_confidence[face] = float(confidence[indices][semantic[indices] == winner].mean())
    return face_labels, face_confidence
