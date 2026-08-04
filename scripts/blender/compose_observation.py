from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import traceback

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def _atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _world_bvh(objects):
    vertices, polygons = [], []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        offset = len(vertices)
        vertices.extend(obj.matrix_world @ vertex.co for vertex in mesh.vertices)
        polygons.extend(tuple(offset + index for index in triangle.vertices)
                        for triangle in mesh.loop_triangles)
        evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=True), vertices, polygons


def _bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects if obj.type == "MESH"
              for corner in obj.bound_box]
    lower = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    upper = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return lower, upper


def _delete(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.delete(use_global=False)


def _axis_rotation(axis):
    return {
        "+Z": Matrix.Identity(4), "-Z": Matrix.Rotation(math.pi, 4, "X"),
        "+X": Matrix.Rotation(-math.pi / 2, 4, "Y"),
        "-X": Matrix.Rotation(math.pi / 2, 4, "Y"),
        "+Y": Matrix.Rotation(math.pi / 2, 4, "X"),
        "-Y": Matrix.Rotation(-math.pi / 2, 4, "X"),
    }[axis]


def _import_object(path, axis, target_dimension, target_size, yaw, point, clearance):
    before = set(bpy.context.scene.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Object import failed: {result}")
    objects = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        _delete(objects)
        raise RuntimeError("Object contains no mesh")
    rotation = Matrix.Rotation(yaw, 4, "Z") @ _axis_rotation(axis)
    for obj in objects:
        if obj.parent not in objects:
            obj.matrix_world = rotation @ obj.matrix_world
    lower, upper = _bounds(meshes)
    extent = upper - lower
    denominator = extent.z if target_dimension == "height" else max(extent)
    if denominator <= 1e-8:
        _delete(objects)
        raise RuntimeError("Object has degenerate bounds")
    scale = target_size / denominator
    for obj in objects:
        if obj.parent not in objects:
            obj.scale *= scale
    bpy.context.view_layer.update()
    lower, upper = _bounds(meshes)
    shift = Vector((point[0] - (lower.x + upper.x) / 2,
                    point[1] - (lower.y + upper.y) / 2,
                    point[2] + clearance - lower.z))
    for obj in objects:
        if obj.parent not in objects:
            obj.location += shift
    bpy.context.view_layer.update()
    return objects, meshes, scale


def _inside_support(meshes, support, margin):
    lower, upper = _bounds(meshes)
    patch_min, patch_max = support["bounds_min"], support["bounds_max"]
    return (lower.x >= patch_min[0] + margin and upper.x <= patch_max[0] - margin and
            lower.y >= patch_min[1] + margin and upper.y <= patch_max[1] - margin)


def _visible(scene_bvh, eye, meshes):
    lower, upper = _bounds(meshes)
    target = (lower + upper) * 0.5
    direction = target - eye
    distance = direction.length
    if distance <= 1e-6:
        return False
    hit = scene_bvh.ray_cast(eye, direction.normalized(), distance - 1e-3)
    return hit[0] is None


def main():
    job = json.loads(_args().job.read_text(encoding="utf-8"))
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(int(job["seed"]))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    imported = bpy.ops.import_scene.gltf(filepath=job["partition_glb"])
    if "FINISHED" not in imported:
        raise RuntimeError(f"Partition import failed: {imported}")
    scene_objects = list(bpy.context.scene.objects)
    scene_bvh, _, _ = _world_bvh(scene_objects)
    failures = []
    eye = Vector(job["observation"]["anchor_eye"])
    placement = job["placement"]
    for record in job["objects"]:
        classification = record["classification"]
        profile = record["profile_data"]
        supports = [value for value in job["supports"]
                    if value["semantic_class"] in classification["support_classes"]]
        randomizer.shuffle(supports)
        orientations = list(profile["orientations"])
        # Objaverse GLBs are commonly authored Z-up. Keep that convention as the
        # first deterministic candidate; the profile preserves five alternatives
        # for a later image-based upright estimator.
        orientations.sort(key=lambda item: item["up_axis"] != "+Z")
        for support in supports:
            points = list(support["candidate_points"])
            randomizer.shuffle(points)
            points = points[:int(placement["positions_per_surface"])]
            for candidate_point in points:
                point = candidate_point["position"]
                for _ in range(int(placement["yaw_trials"])):
                    yaw = randomizer.uniform(-math.pi, math.pi)
                    target = randomizer.uniform(float(classification["target_min_m"]),
                                                float(classification["target_max_m"]))
                    objects = []
                    try:
                        objects, meshes, scale = _import_object(
                            record["glb"], orientations[0]["up_axis"],
                            classification["target_dimension"], target, yaw, point,
                            float(placement["minimum_clearance_m"]),
                        )
                        normalization = job["normalization"]
                        if not (float(normalization["minimum_scale_factor"]) <= scale <=
                                float(normalization["maximum_scale_factor"])):
                            raise RuntimeError("normalization_scale_outside_limits")
                        if not _inside_support(meshes, support, 0.0):
                            raise RuntimeError("footprint_outside_support_bounds")
                        lower, upper = _bounds(meshes)
                        footprint_radius = math.hypot(upper.x - lower.x, upper.y - lower.y) / 2
                        required_clearance = (
                            footprint_radius + float(placement["support_boundary_margin_m"])
                        )
                        if float(candidate_point["boundary_clearance_m"]) < required_clearance:
                            raise RuntimeError("insufficient_support_boundary_clearance")
                        object_bvh, _, _ = _world_bvh(meshes)
                        if scene_bvh.overlap(object_bvh):
                            raise RuntimeError("scene_collision")
                        if not _visible(scene_bvh, eye, meshes):
                            raise RuntimeError("object_center_occluded_from_anchor")
                        combined = output / "combined_scene.glb"
                        bpy.ops.export_scene.gltf(filepath=str(combined), export_format="GLB")
                        lower, upper = _bounds(meshes)
                        _atomic(output / "placement.json", {
                            "schema_version": 1, "status": "success",
                            "observation_id": job["observation_id"], "object_uid": record["uid"],
                            "canonical_class": classification["canonical_class"],
                            "support_patch_id": support["patch_id"],
                            "support_class": support["semantic_class"], "position": point,
                            "yaw_rad": yaw, "up_axis": orientations[0]["up_axis"],
                            "uniform_scale": scale, "bounds_min": list(lower),
                            "bounds_max": list(upper), "failed_trials": failures,
                        })
                        _atomic(output / "camera.json", {
                            "schema_version": 1, "reference_position": list(eye),
                            "look_at": list((lower + upper) * 0.5),
                            "horizontal_fov_deg": placement["horizontal_fov_deg"],
                            "rendered": False,
                        })
                        return
                    except Exception as error:
                        failures.append({"object_uid": record["uid"],
                                         "support_patch_id": support["patch_id"],
                                         "reason": str(error)})
                        if objects:
                            _delete(objects)
    _atomic(output / "placement.json", {"schema_version": 1, "status": "failed",
            "observation_id": job["observation_id"], "failed_trials": failures})
    raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
