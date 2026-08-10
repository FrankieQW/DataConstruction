from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from . import __version__
from .config import ProjectConfig
from .io_utils import (
    dump_json_atomic,
    dump_jsonl_atomic,
    load_json,
    sha256_file,
    stable_digest,
    utc_now,
)
from .schemas import (
    AnnotationDocument,
    ObjectDocument,
    PreparedGeometryDocument,
    RenderJob,
    RenderJobDocument,
    SceneDocument,
)


def build_render_jobs(
    config: ProjectConfig,
    *,
    annotations_path: Path | None = None,
    output_path: Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    annotation_path = annotations_path or config.path("annotation_output")
    object_path = config.path("object_output")
    scene_path = config.path("scene_output")
    geometry_path = config.path("prepared_geometry_output")
    annotations = AnnotationDocument.model_validate(load_json(annotation_path)).model_dump(mode="json")
    objects = ObjectDocument.model_validate(load_json(object_path)).model_dump(mode="json")
    scenes = SceneDocument.model_validate(load_json(scene_path)).model_dump(mode="json")
    prepared = PreparedGeometryDocument.model_validate(load_json(geometry_path)).model_dump(mode="json")

    current_digests = {
        "annotation": sha256_file(annotation_path),
        "object": sha256_file(object_path),
        "scene": sha256_file(scene_path),
        "prepared_geometry": sha256_file(geometry_path),
    }
    if annotations["object_digest"] != current_digests["object"]:
        raise ValueError("annotation_construction.json references a stale object.json")
    if annotations["scene_digest"] != current_digests["scene"]:
        raise ValueError("annotation_construction.json references a stale scene.json")
    if prepared["object_digest"] != current_digests["object"]:
        raise ValueError("prepared geometry references a stale object.json")

    m4 = config.section("m4")
    selected_seed = int(seed if seed is not None else config.section("project").get("seed", 0))
    relation_priority = list(m4.get("relation_priority", ["replace", "place_on"]))
    if sorted(relation_priority) != ["place_on", "replace"]:
        raise ValueError("m4.relation_priority must contain place_on and replace exactly once")
    yaw_choices = [float(value) for value in m4.get("yaw_degrees", [0, 90, 180, 270])]
    if not yaw_choices:
        raise ValueError("m4.yaw_degrees cannot be empty")
    fixture_categories = {str(value) for value in m4.get("fixture_categories", [])}
    replacement_fill = float(m4.get("replacement_fill_ratio", 0.9))
    if not 0 < replacement_fill <= 1:
        raise ValueError("m4.replacement_fill_ratio must be in (0, 1]")

    object_by_uid = {row["uid"]: row for row in objects["objects"]}
    geometry_by_uid = {row["object_uid"]: row for row in prepared["geometries"]}
    targets = annotations["targets_by_object_category"]
    jobs: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    max_jobs = m4.get("max_render_jobs")

    for geometry in sorted(prepared["geometries"], key=lambda row: row["object_uid"]):
        uid = geometry["object_uid"]
        object_row = object_by_uid.get(uid)
        if object_row is None:
            rejects.append({"object_uid": uid, "reason": "missing_object_row"})
            continue
        category_targets = targets.get(geometry["object_category"], {})
        allowed = {
            "place_on": set(category_targets.get("place_on_entity_ids", [])),
            "replace": set(category_targets.get("replace_entity_ids", [])),
        }
        for scene in sorted(scenes["scenes"], key=lambda row: row["scene_id"]):
            blend_path = (config.root / scene["normalized_blend"]).resolve()
            if not blend_path.is_file():
                rejects.append(
                    {
                        "object_uid": uid,
                        "base_scene_id": scene["scene_id"],
                        "reason": f"missing_base_scene_blend:{blend_path}",
                    }
                )
                continue
            entities = {row["entity_id"]: row for row in scene["entities"]}
            candidates = {
                relation: sorted(entities[entity_id] for entity_id in ids if entity_id in entities)
                for relation, ids in allowed.items()
            }
            relation = next((item for item in relation_priority if candidates[item]), None)
            if relation is None:
                continue
            rng = random.Random(_seed_for(selected_seed, uid, scene["scene_id"]))
            target = candidates[relation][rng.randrange(len(candidates[relation]))]
            dimensions = _three_positive(target.get("obb_world", {}).get("dimensions"), "target dimensions")
            center = _three(target.get("obb_world", {}).get("center"), "target center")
            if relation == "replace":
                desired_dimensions = [value * replacement_fill for value in dimensions]
                bottom_center = [center[0], center[1], center[2] - dimensions[2] * 0.5]
            else:
                desired_dimensions = list(geometry["target_dimensions"])
                bottom_center = [center[0], center[1], center[2] + dimensions[2] * 0.5]
            yaw = yaw_choices[rng.randrange(len(yaw_choices))]
            annotation_id = stable_digest(
                {
                    "annotation": current_digests["annotation"],
                    "category": geometry["object_category"],
                    "relation": relation,
                    "entity_id": target["entity_id"],
                }
            )
            job_payload = {
                "schema_version": str(config.section("project").get("schema_version", "1.0")),
                "seed": _seed_for(selected_seed, uid, scene["scene_id"], target["entity_id"]),
                "annotation_id": annotation_id,
                "object_uid": uid,
                "object_category": geometry["object_category"],
                "object_asset_path": geometry["asset_path"],
                "prepared_geometry": geometry,
                "base_scene_id": scene["scene_id"],
                "base_scene_blend": scene["normalized_blend"],
                "base_scene_digest": scene["source_digest"],
                "target": {
                    "relation": relation,
                    "entity_id": target["entity_id"],
                    "node_ids": target["node_ids"],
                    "category": target["category"],
                    "center_world": center,
                    "dimensions_world": dimensions,
                    "bottom_center_world": bottom_center,
                    "desired_dimensions": desired_dimensions,
                    "yaw_degrees": yaw,
                },
                "camera": {
                    "strategy": str(m4.get("camera_strategy", "existing_or_target_framed")),
                    "name": _scene_override(config, scene["scene_id"]).get("camera"),
                    "focal_length": float(m4.get("camera_focal_length", 50.0)),
                    "distance_scale": float(m4.get("camera_distance_scale", 3.0)),
                },
                "lighting_profile": {
                    "name": str(m4.get("lighting_profile", "tokenlight_linear_v1")),
                    "native_analytic": False,
                    "native_emissive": False,
                    "component_world_strength": 0.0,
                },
                "fixture_candidate_entity_ids": sorted(
                    entity["entity_id"]
                    for entity in scene["entities"]
                    if entity["category"] in fixture_categories
                ),
                "lineage": {
                    "annotation_digest": current_digests["annotation"],
                    "object_digest": current_digests["object"],
                    "scene_digest": current_digests["scene"],
                    "prepared_geometry_digest": current_digests["prepared_geometry"],
                    "config_digest": config.digest,
                    "generator_version": __version__,
                },
                "license": {
                    "object": geometry["license"],
                    "base_scene": m4.get("base_scene_license", {}),
                    "policy_version": str(m4.get("license_policy_version", "unconfigured")),
                    "decision": "allowed"
                    if geometry["license"].get("decision") == "allowed"
                    and m4.get("base_scene_license", {}).get("decision") == "allowed"
                    else "unverified",
                },
            }
            job_payload["job_id"] = _job_id(job_payload)
            jobs.append(RenderJob.model_validate(job_payload).model_dump(mode="json"))
            if max_jobs is not None and len(jobs) >= int(max_jobs):
                break
        if max_jobs is not None and len(jobs) >= int(max_jobs):
            break

    jobs.sort(key=lambda row: row["job_id"])
    document = RenderJobDocument(
        schema_version=str(config.section("project").get("schema_version", "1.0")),
        generated_at=utc_now(),
        generator_version=__version__,
        config_digest=config.digest,
        annotation_digest=current_digests["annotation"],
        object_digest=current_digests["object"],
        scene_digest=current_digests["scene"],
        prepared_geometry_digest=current_digests["prepared_geometry"],
        jobs=jobs,
        rejects=rejects,
        stats={
            "prepared_objects": len(geometry_by_uid),
            "scenes": len(scenes["scenes"]),
            "jobs": len(jobs),
            "place_on": sum(row["target"]["relation"] == "place_on" for row in jobs),
            "replace": sum(row["target"]["relation"] == "replace" for row in jobs),
            "rejects": len(rejects),
        },
    ).model_dump(mode="json")
    selected_output = output_path or config.path("render_jobs_output")
    dump_jsonl_atomic(selected_output, document["jobs"])
    dump_json_atomic(config.path("render_jobs_summary"), document)
    dump_jsonl_atomic(config.path("render_job_rejects"), document["rejects"])
    return document


def _seed_for(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _job_id(payload: dict[str, Any]) -> str:
    identity = {
        key: payload[key]
        for key in ("object_uid", "base_scene_id", "target", "camera", "lighting_profile", "seed")
    }
    return "composition_" + stable_digest(identity).removeprefix("sha256:")[:20]


def _three(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    return [float(item) for item in value]


def _three_positive(value: Any, label: str) -> list[float]:
    result = _three(value, label)
    if any(item <= 0 for item in result):
        raise ValueError(f"{label} must be positive")
    return result


def _scene_override(config: ProjectConfig, scene_id: str) -> dict[str, Any]:
    overrides = config.section("m4").get("scene_render_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("m4.scene_render_overrides must be a mapping")
    value = overrides.get(scene_id, {})
    if not isinstance(value, dict):
        raise TypeError(f"scene render override must be a mapping: {scene_id}")
    return value
