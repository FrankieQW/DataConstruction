from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback
from typing import Any

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    runtime = json.loads(Path(args.runtime_config).read_text(encoding="utf-8"))
    project_root = Path(runtime["project_root"])
    helpers = _load_helpers(project_root)
    jobs = _read_jsonl(Path(runtime["jobs_path"]))
    worker_index = int(runtime["worker_index"])
    worker_count = int(runtime["worker_count"])
    selected = [job for index, job in enumerate(jobs) if index % worker_count == worker_index]
    output_root = Path(runtime["output_root"])
    worker_root = output_root / "render_workers" / f"worker_{worker_index:03d}"
    errors: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for job in selected:
        try:
            metadata = render_job(job, runtime, helpers)
            completed.append({"job_id": job["job_id"], "metadata": metadata})
            print(f"COMPOSITION_DONE {job['job_id']}", flush=True)
        except Exception as error:  # noqa: BLE001 - every failed render remains recoverable by job id.
            errors.append(
                {
                    "job_id": job.get("job_id"),
                    "error": type(error).__name__,
                    "detail": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"COMPOSITION_FAILED {job.get('job_id')}: {error}", flush=True)
    _write_jsonl(worker_root / "completed.jsonl", completed)
    _write_jsonl(worker_root / "errors.jsonl", errors)
    if errors and not bool(runtime.get("allow_partial", False)):
        raise RuntimeError(f"worker {worker_index} failed {len(errors)} render job(s)")


def render_job(job: dict[str, Any], runtime: dict[str, Any], helpers) -> str:
    output_root = Path(runtime["output_root"])
    final_directory = output_root / "components" / job["job_id"]
    metadata_path = final_directory / "metadata.json"
    if metadata_path.is_file() and not bool(runtime.get("overwrite", False)):
        return metadata_path.relative_to(output_root).as_posix()
    if final_directory.exists():
        if not bool(runtime.get("overwrite", False)):
            raise FileExistsError(f"incomplete output exists and overwrite=false: {final_directory}")
        shutil.rmtree(final_directory)
    partial = final_directory.with_name(final_directory.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    project_root = Path(runtime["project_root"])
    blend_path = (project_root / job["base_scene_blend"]).resolve()
    object_path = (Path(runtime["object_root"]) / job["object_asset_path"]).resolve()
    if not blend_path.is_file():
        raise FileNotFoundError(blend_path)
    if not object_path.is_file():
        raise FileNotFoundError(object_path)
    if _sha256_file(object_path) != job["prepared_geometry"]["asset_digest"]:
        raise ValueError("object asset digest changed after render-job generation")
    if job.get("license", {}).get("decision") != "allowed" and bool(
        runtime.get("render", {}).get("require_verified_license", True)
    ):
        raise ValueError("render job license decision is not allowed")

    try:
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        render_config = _render_config(runtime.get("render", {}))
        helpers.configure_render(render_config)
        base_fingerprint = _scene_fingerprint()
        target_objects = _entity_objects(job["target"]["entity_id"])
        if not target_objects:
            raise ValueError(f"target entity not found in blend: {job['target']['entity_id']}")
        hidden_target_names: list[str] = []
        if job["target"]["relation"] == "replace":
            for obj in target_objects:
                obj.hide_render = True
                obj.hide_viewport = True
                hidden_target_names.append(obj.name)

        imported, meshes = helpers.import_asset(object_path)
        root, scale = _normalize_imported(imported, meshes, job["prepared_geometry"], job["target"])
        if job["target"]["relation"] == "place_on":
            _validate_place_footprint(meshes, job["target"], runtime.get("render", {}))
        collision_pairs = _validate_scene_collisions(
            meshes,
            target_objects,
            runtime.get("render", {}),
        )
        camera, camera_target = _select_camera(job, target_objects)
        bpy.context.scene.camera = camera
        _disable_native_lighting()
        background = _world_background(float(render_config["ambient_world_strength"]))

        visibility_path = partial / "diagnostics" / "inserted_visibility.png"
        inserted_pixels = _render_mask(visibility_path, meshes, render_config)
        if inserted_pixels < int(render_config["minimum_visible_pixels"]):
            raise ValueError(
                f"inserted object has only {inserted_pixels} visible pixels; "
                f"minimum={render_config['minimum_visible_pixels']}"
            )

        rng = random.Random(int(job["seed"]))
        fixture = _select_or_create_fixture(job, runtime, camera, camera_target, partial, helpers, rng)
        ambient_path = partial / "ambient.exr"
        background.inputs["Strength"].default_value = float(render_config["ambient_world_strength"])
        helpers.render_exr(ambient_path)
        background.inputs["Strength"].default_value = 0.0
        dark_path = partial / "dark.exr"
        helpers.render_exr(dark_path)

        fixture_component = _render_fixture_component(
            fixture, partial, output_root, final_directory, render_config, helpers
        )
        point_components = _render_point_components(
            partial, output_root, final_directory, camera, camera_target, render_config, helpers, rng
        )
        diffuse_components = _render_diffuse_components(
            partial, output_root, final_directory, camera, camera_target, render_config, helpers
        )
        canonical = {
            "origin": [float(value) for value in camera_target],
            "asset_size": max(float(value) for value in job["target"]["desired_dimensions"]),
            "position_axes": "x=right,y=camera-forward,z=up",
            "units": "meter",
        }
        metadata = {
            "schema_version": job["schema_version"],
            "id": job["job_id"],
            "asset_uid": job["object_uid"],
            "asset": job["object_asset_path"],
            "ambient": _future_relative(ambient_path, partial, final_directory, output_root),
            "dark": _future_relative(dark_path, partial, final_directory, output_root),
            "point_lights": point_components,
            "diffuse": diffuse_components,
            "in_scene_lights": [fixture_component],
            "camera": {
                "name": camera.name,
                "location": [float(value) for value in camera.location],
                "rotation_euler": [float(value) for value in camera.rotation_euler],
                "focal_length": float(camera.data.lens),
                "target": [float(value) for value in camera_target],
                "coordinate_space": "blender_world_meter",
            },
            "canonical": canonical,
            "composition": {
                "relation": job["target"]["relation"],
                "target_entity_id": job["target"]["entity_id"],
                "target_node_ids": job["target"]["node_ids"],
                "object_transform_world": _matrix_flat(root.matrix_world),
                "target_transform_before": [
                    {"name": obj.name, "matrix_world": _matrix_flat(obj.matrix_world)}
                    for obj in target_objects
                ],
                "hidden_target_objects": hidden_target_names,
                "asset_scale": [float(value) for value in scale],
                "inserted_visible_pixels": inserted_pixels,
                "collision_pairs": collision_pairs,
            },
            "lighting_profile": job["lighting_profile"],
            "base_scene_id": job["base_scene_id"],
            "base_scene_fingerprint": base_fingerprint,
            "lineage": job["lineage"],
            "license": job["license"],
        }
        (partial / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final_directory)
        return (final_directory / "metadata.json").relative_to(output_root).as_posix()
    except BaseException:
        if partial.exists():
            failure = partial / "failure.json"
            failure.write_text(
                json.dumps({"job_id": job.get("job_id"), "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8",
            )
        raise


def _normalize_imported(imported, meshes, geometry: dict[str, Any], target: dict[str, Any]):
    root = bpy.data.objects.new(f"LC_ObjectRoot_{geometry['object_uid']}", None)
    bpy.context.collection.objects.link(root)
    for obj in imported:
        if obj.parent is None:
            matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = matrix
    source_up = _axis_vector(geometry["up_axis"])
    source_front = _axis_vector(geometry["front_axis"])
    up_rotation = source_up.rotation_difference(Vector((0.0, 0.0, 1.0)))
    rotated_front = up_rotation @ source_front
    rotated_front.z = 0.0
    if rotated_front.length <= 1e-8:
        raise ValueError("front axis collapses after up-axis normalization")
    rotated_front.normalize()
    target_front = Vector((0.0, -1.0, 0.0))
    signed_angle = math.atan2(rotated_front.cross(target_front).z, rotated_front.dot(target_front))
    yaw = math.radians(float(target["yaw_degrees"]))
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), yaw + signed_angle) @ up_rotation
    bpy.context.view_layer.update()
    minimum, maximum = _world_bounds(meshes)
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    if any(value <= 1e-8 for value in dimensions):
        raise ValueError(f"imported asset has a degenerate bounding box: {dimensions}")
    desired = [float(value) for value in target["desired_dimensions"]]
    if geometry["fit_mode"] == "uniform_fit":
        scalar = min(desired[index] / dimensions[index] for index in range(3))
        scale = [scalar, scalar, scalar]
    else:
        scale = [desired[index] / dimensions[index] for index in range(3)]
    root.scale = scale
    bpy.context.view_layer.update()
    minimum, maximum = _world_bounds(meshes)
    bottom_center = Vector(
        ((minimum.x + maximum.x) * 0.5, (minimum.y + maximum.y) * 0.5, minimum.z)
    )
    destination = Vector(target["bottom_center_world"])
    root.location += destination - bottom_center
    bpy.context.view_layer.update()
    return root, scale


def _validate_place_footprint(meshes, target: dict[str, Any], render: dict[str, Any]) -> None:
    minimum, maximum = _world_bounds(meshes)
    object_xy = (maximum.x - minimum.x, maximum.y - minimum.y)
    target_xy = target["dimensions_world"][:2]
    limit = float(render.get("place_footprint_limit", 1.0))
    if any(object_xy[index] > float(target_xy[index]) * limit for index in range(2)):
        raise ValueError(f"placed object footprint {object_xy} exceeds support {target_xy}")


def _validate_scene_collisions(meshes, target_objects, render: dict[str, Any]) -> list[list[str]]:
    """Reject broad-phase intersections with scene meshes other than the relation target.

    This intentionally records an empty collision-pair list in accepted metadata.  The
    target is excluded because place_on needs contact and replace hides that geometry.
    """
    tolerance = float(render.get("collision_aabb_tolerance", 0.002))
    inserted = set(meshes)
    excluded = inserted | set(target_objects)
    scene_meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj not in excluded and not obj.hide_render and not obj.hide_viewport
    ]
    collisions: list[list[str]] = []
    for inserted_obj in sorted(inserted, key=lambda obj: obj.name):
        inserted_minimum, inserted_maximum = _world_bounds([inserted_obj])
        for scene_obj in sorted(scene_meshes, key=lambda obj: obj.name):
            scene_minimum, scene_maximum = _world_bounds([scene_obj])
            overlaps = [
                min(inserted_maximum[axis], scene_maximum[axis])
                - max(inserted_minimum[axis], scene_minimum[axis])
                for axis in range(3)
            ]
            if all(value > tolerance for value in overlaps):
                collisions.append([inserted_obj.name, scene_obj.name])
    if collisions:
        preview = ", ".join(f"{left}<->{right}" for left, right in collisions[:5])
        raise ValueError(f"inserted object intersects non-target scene geometry: {preview}")
    return collisions


def _select_camera(job: dict[str, Any], target_objects):
    requested = job["camera"].get("name")
    if requested and requested in bpy.data.objects and bpy.data.objects[requested].type == "CAMERA":
        camera = bpy.data.objects[requested]
    elif bpy.context.scene.camera and bpy.context.scene.camera.type == "CAMERA":
        camera = bpy.context.scene.camera
    else:
        cameras = sorted((obj for obj in bpy.data.objects if obj.type == "CAMERA"), key=lambda obj: obj.name)
        camera = cameras[0] if cameras else None
    center = Vector(job["target"]["center_world"])
    if camera is None:
        dimensions = job["target"]["dimensions_world"]
        distance = max(float(value) for value in dimensions) * float(job["camera"]["distance_scale"])
        distance = max(distance, 2.0)
        data = bpy.data.cameras.new("LC_CompositionCamera")
        camera = bpy.data.objects.new("LC_CompositionCamera", data)
        bpy.context.collection.objects.link(camera)
        camera.location = center + Vector((distance * 0.7, -distance, distance * 0.5))
        camera.data.lens = float(job["camera"]["focal_length"])
        _look_at(camera, center)
    target = center
    target_meshes = [obj for obj in target_objects if obj.type == "MESH"]
    if target_meshes:
        minimum, maximum = _world_bounds(target_meshes)
        target = (minimum + maximum) * 0.5
    return camera, target


def _select_or_create_fixture(job, runtime, camera, camera_target, partial, helpers, rng):
    config = runtime.get("fixture", {})
    candidates: list[dict[str, Any]] = []
    for entity_id in job.get("fixture_candidate_entity_ids", []):
        objects = [obj for obj in _entity_objects(entity_id) if obj.type == "MESH" and not obj.hide_render]
        if not objects:
            continue
        mask_path = partial / "diagnostics" / f"fixture_candidate_{_safe_name(entity_id)}.png"
        pixels = _render_mask(mask_path, objects, _render_config(runtime.get("render", {})))
        if pixels < int(config.get("minimum_visible_pixels", 64)):
            continue
        center = sum((_object_center(obj) for obj in objects), Vector()) / len(objects)
        ndc = world_to_camera_view(bpy.context.scene, camera, center)
        center_distance = math.hypot(float(ndc.x) - 0.5, float(ndc.y) - 0.5)
        candidates.append(
            {
                "source": "scene_native",
                "entity_id": entity_id,
                "objects": objects,
                "world_position": center,
                "visible_pixels": pixels,
                "center_distance": center_distance,
                "candidate_mask": mask_path,
                "material": None,
            }
        )
    if candidates:
        candidates.sort(key=lambda row: (-row["visible_pixels"], row["center_distance"], row["entity_id"]))
        selected = candidates[0]
        selected["position"] = _world_to_canonical(camera, camera_target, selected["world_position"])
        return selected

    ranges = config.get(
        "fallback_position_ranges",
        {"x": [-0.8, 0.8], "y": [-0.6, -0.1], "z": [0.4, 1.2]},
    )
    canonical = [rng.uniform(*ranges[axis]) for axis in ("x", "y", "z")]
    world_position = helpers.camera_to_world(camera, camera_target, canonical)
    mesh, material = helpers.add_fixture(
        "LC_ProceduralFixture",
        world_position,
        float(config.get("fallback_fixture_size", 0.08)),
    )
    mask_path = partial / "diagnostics" / "fixture_fallback.png"
    pixels = _render_mask(mask_path, [mesh], _render_config(runtime.get("render", {})))
    if pixels < int(config.get("minimum_visible_pixels", 64)):
        raise ValueError(f"procedural fallback fixture is not sufficiently visible: {pixels} pixels")
    return {
        "source": "procedural_fallback",
        "entity_id": None,
        "objects": [mesh],
        "world_position": world_position,
        "position": canonical,
        "visible_pixels": pixels,
        "candidate_mask": mask_path,
        "material": material,
    }


def _render_fixture_component(fixture, partial, output_root, final_directory, config, helpers):
    mask_path = partial / "in_scene_lights" / "fixture_000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture["candidate_mask"], mask_path)
    energy = float(config["fixture_energy"])
    radius = float(config["fixture_light_radius"])
    light = helpers.add_light("LC_FixtureLight", "POINT", fixture["world_position"], energy, radius)
    if fixture.get("material") is not None:
        helpers.set_fixture_emission(
            fixture["material"],
            (1.0, 1.0, 1.0, 1.0),
            float(config["fallback_emission_strength"]),
        )
    try:
        component_path = partial / "in_scene_lights" / "fixture_000_on.exr"
        helpers.render_exr(component_path)
    finally:
        if fixture.get("material") is not None:
            helpers.set_fixture_emission(fixture["material"], (1.0, 1.0, 1.0, 1.0), 0.0)
        helpers.remove_light(light)
    return {
        "path": _future_relative(component_path, partial, final_directory, output_root),
        "mask": _future_relative(mask_path, partial, final_directory, output_root),
        "position": [float(value) for value in fixture["position"]],
        "renderer_position": [float(value) for value in fixture["world_position"]],
        "base_energy": energy,
        "diffuse": radius,
        "fixture_source": fixture["source"],
        "fixture_entity_id": fixture["entity_id"],
        "visible_pixels": int(fixture["visible_pixels"]),
        "binding": "controlled_point_at_fixture_center",
    }


def _render_point_components(partial, output_root, final_directory, camera, target, config, helpers, rng):
    count = int(config["point_lights_per_scene"])
    if count < 1:
        return []
    light = helpers.add_light("LC_PointLight", "POINT", (0, 0, 0), float(config["point_energy"]), 0.0)
    rows = []
    try:
        for index in range(count):
            position = [rng.uniform(*config["point_position_ranges"][axis]) for axis in ("x", "y", "z")]
            world_position = helpers.camera_to_world(camera, target, position)
            radius = rng.uniform(*config["point_radius_range"])
            light.location = world_position
            light.data.shadow_soft_size = radius
            bpy.context.view_layer.update()
            path = partial / "point_lights" / f"light_{index:03d}.exr"
            helpers.render_exr(path)
            rows.append(
                {
                    "path": _future_relative(path, partial, final_directory, output_root),
                    "position": position,
                    "renderer_position": [float(value) for value in world_position],
                    "base_energy": float(config["point_energy"]),
                    "diffuse": radius,
                }
            )
    finally:
        helpers.remove_light(light)
    return rows


def _render_diffuse_components(partial, output_root, final_directory, camera, target, config, helpers):
    spreads = [float(value) for value in config["diffuse_sizes"]]
    if len(spreads) < 2:
        raise ValueError("at least two diffuse sizes are required")
    world_position = helpers.camera_to_world(camera, target, config["diffuse_position"])
    light = helpers.add_light(
        "LC_DiffuseLight", "AREA", world_position, float(config["point_energy"]), spreads[0]
    )
    _look_at(light, target)
    rows = []
    try:
        for index, spread in enumerate(spreads):
            light.data.size = spread
            bpy.context.view_layer.update()
            path = partial / "diffuse" / f"spread_{index:02d}.exr"
            helpers.render_exr(path)
            rows.append(
                {
                    "path": _future_relative(path, partial, final_directory, output_root),
                    "level": index / (len(spreads) - 1),
                    "size": spread,
                }
            )
    finally:
        helpers.remove_light(light)
    return rows


def _render_mask(path: Path, selected_objects, config: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    original_engine = scene.render.engine
    original_path = scene.render.filepath
    image_settings = scene.render.image_settings
    original_format = image_settings.file_format
    original_mode = image_settings.color_mode
    original_depth = image_settings.color_depth
    original_film = scene.render.film_transparent
    colors = {obj: tuple(obj.color) for obj in bpy.data.objects if obj.type == "MESH"}
    selected = set(selected_objects)
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "OBJECT"
        scene.display.shading.show_shadows = False
        scene.display.shading.show_cavity = False
        scene.display.shading.show_specular_highlight = False
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = (0.0, 0.0, 0.0)
        for obj in colors:
            obj.color = (1.0, 1.0, 1.0, 1.0) if obj in selected else (0.0, 0.0, 0.0, 1.0)
        image_settings.file_format = "PNG"
        image_settings.color_mode = "BW"
        image_settings.color_depth = "8"
        scene.render.film_transparent = False
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    finally:
        for obj, color in colors.items():
            obj.color = color
        scene.render.engine = original_engine
        scene.render.filepath = original_path
        image_settings.file_format = original_format
        image_settings.color_mode = original_mode
        image_settings.color_depth = original_depth
        scene.render.film_transparent = original_film
        _apply_cycles_settings(scene, config)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = list(image.pixels)
        return sum(pixels[index] > 0.5 for index in range(0, len(pixels), 4))
    finally:
        bpy.data.images.remove(image)


def _disable_native_lighting() -> None:
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.data.energy = 0.0
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                strength = node.inputs.get("Emission Strength")
                color = node.inputs.get("Emission Color")
                if color is None:
                    color = node.inputs.get("Emission")
                if strength is not None:
                    strength.default_value = 0.0
                if color is not None:
                    color.default_value = (0.0, 0.0, 0.0, 1.0)
            elif node.type == "EMISSION" and node.inputs.get("Strength") is not None:
                node.inputs["Strength"].default_value = 0.0


def _world_background(strength: float):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("LC_ControlledWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
    if background is None:
        world.node_tree.nodes.clear()
        output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        background = world.node_tree.nodes.new("ShaderNodeBackground")
        world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
        background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    background.inputs["Strength"].default_value = strength
    return background


def _render_config(value: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "resolution": 512,
        "engine": "CYCLES",
        "samples": 64,
        "denoise": True,
        "persistent_data": False,
        "threads_per_worker": "auto",
        "device": "GPU",
        "compute_device_type": "CUDA",
        "ambient_world_strength": 1.0,
        "minimum_visible_pixels": 256,
        "place_footprint_limit": 1.0,
        "collision_aabb_tolerance": 0.002,
        "point_lights_per_scene": 16,
        "point_energy": 500.0,
        "point_position_ranges": {
            "x": [-1.2, 1.2],
            "y": [-0.6, 1.2],
            "z": [0.15, 1.8],
        },
        "point_radius_range": [0.03, 0.25],
        "diffuse_sizes": [0.05, 0.2, 0.4, 0.7, 1.0, 1.5],
        "diffuse_position": [0.0, 0.3, 1.3],
        "fixture_energy": 300.0,
        "fixture_light_radius": 0.12,
        "fallback_emission_strength": 4.0,
        "require_verified_license": True,
    }
    result = {**defaults, **(value or {})}
    return result


def _apply_cycles_settings(scene, config: dict[str, Any]) -> None:
    scene.render.engine = "BLENDER_EEVEE_NEXT" if config["engine"] == "EEVEE" else "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(config["samples"])
        scene.cycles.use_denoising = bool(config["denoise"])


def _scene_fingerprint() -> str:
    rows = []
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        rows.append(
            {
                "name": obj.name,
                "type": obj.type,
                "matrix": _matrix_flat(obj.matrix_world),
                "hide_render": bool(obj.hide_render),
                "entity_id": obj.get("lc_scene_entity_id"),
                "light_energy": float(obj.data.energy) if obj.type == "LIGHT" else None,
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _entity_objects(entity_id: str):
    return sorted(
        (obj for obj in bpy.data.objects if obj.get("lc_scene_entity_id") == entity_id),
        key=lambda obj: obj.name,
    )


def _world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise ValueError("cannot compute bounds for an empty mesh set")
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def _object_center(obj) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def _world_to_canonical(camera, target, position) -> list[float]:
    forward = (Vector(target) - camera.location).normalized()
    right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
    up = right.cross(forward).normalized()
    delta = Vector(position) - Vector(target)
    return [float(delta.dot(right)), float(delta.dot(forward)), float(delta.dot(up))]


def _axis_vector(axis: str) -> Vector:
    sign = -1.0 if axis.startswith("-") else 1.0
    base = {
        "X": Vector((1.0, 0.0, 0.0)),
        "Y": Vector((0.0, 1.0, 0.0)),
        "Z": Vector((0.0, 0.0, 1.0)),
    }[axis[-1]]
    return base * sign


def _look_at(obj, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _matrix_flat(matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _future_relative(path: Path, partial: Path, final: Path, output_root: Path) -> str:
    future = final / path.relative_to(partial)
    return future.relative_to(output_root).as_posix()


def _safe_name(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_helpers(project_root: Path):
    path = project_root / "Lumina-T2X" / "tools" / "tokenlight_data" / "render_assets.py"
    spec = importlib.util.spec_from_file_location("tokenlight_render_assets", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
