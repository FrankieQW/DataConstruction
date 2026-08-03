from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import FusionConfig
from .lifting import LiftedObservation


@dataclass(frozen=True)
class AssociatedInstance:
    observation_indices: tuple[int, ...]
    class_id: int
    point_indices: np.ndarray
    score: float
    view_ids: tuple[str, ...]


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def associate_observations(observations: list[LiftedObservation], point_features: np.ndarray | None, config: FusionConfig) -> list[AssociatedInstance]:
    if not observations:
        return []
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        cell = tuple(np.floor(observation.centroid / config.association_grid_m).astype(int))
        buckets[cell].append(index)
    parent = list(range(len(observations)))
    for cell, members in sorted(buckets.items()):
        candidates = set(members)
        for delta in product((-1, 0, 1), repeat=3):
            neighbor = tuple(cell[axis] + delta[axis] for axis in range(3))
            candidates.update(buckets.get(neighbor, ()))
        for left_index in members:
            left = observations[left_index]
            for right_index in sorted(candidate for candidate in candidates if candidate > left_index):
                right = observations[right_index]
                if left.view_id == right.view_id or left.class_id != right.class_id:
                    continue
                overlap = _iou(left.point_indices, right.point_indices)
                distance_score = max(0.0, 1.0 - float(np.linalg.norm(left.centroid - right.centroid)) / config.association_grid_m)
                feature_score = 0.0
                if point_features is not None and len(left.point_indices) and len(right.point_indices):
                    left_feature = point_features[left.point_indices].mean(axis=0)
                    right_feature = point_features[right.point_indices].mean(axis=0)
                    denominator = np.linalg.norm(left_feature) * np.linalg.norm(right_feature)
                    feature_score = float(np.dot(left_feature, right_feature) / denominator) if denominator else 0.0
                score = 0.55 * overlap + 0.20 * distance_score + 0.25 * max(0.0, feature_score)
                if score >= config.association_threshold:
                    left_root, right_root = _find(parent, left_index), _find(parent, right_index)
                    if left_root != right_root:
                        parent[max(left_root, right_root)] = min(left_root, right_root)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(observations)):
        groups[_find(parent, index)].append(index)
    results: list[AssociatedInstance] = []
    for indices in groups.values():
        items = [observations[index] for index in indices]
        views = tuple(sorted({item.view_id for item in items}))
        points = np.unique(np.concatenate([item.point_indices for item in items]))
        if len(points) < config.min_instance_points or len(views) < config.min_instance_views:
            continue
        score = float(np.average([item.score for item in items], weights=[len(item.point_indices) for item in items]))
        results.append(AssociatedInstance(tuple(sorted(indices)), items[0].class_id, points, score, views))
    results.sort(key=lambda item: (item.class_id, int(item.point_indices.min()), item.view_ids))
    return results
