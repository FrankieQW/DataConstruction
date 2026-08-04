from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import uuid

import bpy
from mathutils import Matrix, Vector
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenecompose.observation.config import ObservationPartitionConfig
from scenecompose.observation.contracts import ObservationRegion, observation_id, write_json_atomic


@dataclass(frozen=True)
class MeshSource:
    source_object: bpy.types.Object
    source_name: str
    instance_name: str
    matrix_world: tuple[float, ...]
    is_instance: bool


@dataclass(frozen=True)
class SourceGeometry:
    source: MeshSource
    vertices: np.ndarray
    triangles: np.ndarray
    loops: np.ndarray
    polygons: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    polygon_smooth: np.ndarray
    polygon_material: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    uv_name: str | None
    loop_uv: np.ndarray | None
    materials: tuple[object | None, ...]


@dataclass(frozen=True)
class AnchorCandidate:
    floor: tuple[float, float, float]
    component_id: int
    component_area_m2: float


def _log(message: str) -> None:
    print(f"[SceneCompose] {message}", flush=True)


def _seconds(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-relative", default=None)
    parser.add_argument("--force", action="store_true")
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_scene_id(path: Path) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in path.stem)
    return safe.strip("._-") or "scene"


def _import_scene(path: Path, config: ObservationPartitionConfig) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"FBX import failed with status: {sorted(result)}")
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    scale = config.meters_per_blender_unit
    if scale is None:
        source_scale = float(bpy.context.scene.unit_settings.scale_length)
        scale = source_scale if source_scale > 0 else 1.0
    if scale != 1.0:
        transform = Matrix.Scale(float(scale), 4)
        for obj in bpy.context.scene.objects:
            if obj.parent is None:
                obj.matrix_world = transform @ obj.matrix_world


def _mesh_sources(depsgraph: bpy.types.Depsgraph) -> list[MeshSource]:
    sources: list[MeshSource] = []
    for instance in depsgraph.object_instances:
        evaluated = instance.object
        if evaluated.type != "MESH":
            continue
        original = evaluated.original
        matrix = tuple(float(value) for row in instance.matrix_world for value in row)
        token = hashlib.sha256(repr((tuple(instance.persistent_id), matrix)).encode()).hexdigest()[:12]
        name = f"{original.name_full}__instance_{token}" if instance.is_instance else original.name_full
        sources.append(MeshSource(original, original.name_full, name, matrix, bool(instance.is_instance)))
    sources.sort(key=lambda item: (item.source_name.casefold(), item.instance_name, item.matrix_world))
    if not sources:
        raise RuntimeError("Imported scene contains no mesh geometry")
    return sources


def _extract_source(source: MeshSource, depsgraph: bpy.types.Depsgraph) -> SourceGeometry:
    evaluated = source.source_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        vertices_local = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", vertices_local.ravel())
        homogeneous = np.ones((len(vertices_local), 4), dtype=np.float64)
        homogeneous[:, :3] = vertices_local
        matrix = np.asarray(source.matrix_world, dtype=np.float64).reshape((4, 4))
        vertices = (homogeneous @ matrix.T)[:, :3].astype(np.float32)
        triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        loops = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        polygons = np.empty(len(mesh.loop_triangles), dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
        mesh.loop_triangles.foreach_get("loops", loops.ravel())
        mesh.loop_triangles.foreach_get("polygon_index", polygons)
        corners = vertices[triangles]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        normals = np.divide(
            cross, double_area[:, None], out=np.zeros_like(cross),
            where=double_area[:, None] > 1e-12,
        ).astype(np.float32)
        centroids = corners.mean(axis=1, dtype=np.float32)
        polygon_smooth = np.empty(len(mesh.polygons), dtype=bool)
        polygon_material = np.empty(len(mesh.polygons), dtype=np.int32)
        if len(mesh.polygons):
            mesh.polygons.foreach_get("use_smooth", polygon_smooth)
            mesh.polygons.foreach_get("material_index", polygon_material)
        active_uv = mesh.uv_layers.active
        loop_uv = None
        uv_name = None
        if active_uv is not None:
            uv_name = active_uv.name
            loop_uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
            active_uv.data.foreach_get("uv", loop_uv.ravel())
        return SourceGeometry(
            source=source,
            vertices=vertices,
            triangles=triangles,
            loops=loops,
            polygons=polygons,
            centroids=centroids,
            normals=normals,
            areas=(double_area * 0.5).astype(np.float32),
            polygon_smooth=polygon_smooth,
            polygon_material=polygon_material,
            bounds_min=(vertices.min(axis=0) if len(vertices) else np.zeros(3, dtype=np.float32)),
            bounds_max=(vertices.max(axis=0) if len(vertices) else np.zeros(3, dtype=np.float32)),
            uv_name=uv_name,
            loop_uv=loop_uv,
            materials=tuple(mesh.materials),
        )
    finally:
        evaluated.to_mesh_clear()


def _collect_geometry(
    sources: list[MeshSource], depsgraph: bpy.types.Depsgraph,
) -> tuple[dict[str, np.ndarray], tuple[SourceGeometry, ...]]:
    result: dict[str, list[np.ndarray]] = {key: [] for key in ("centroids", "areas", "normals", "source_indices", "triangle_indices")}
    cached: list[SourceGeometry] = []
    for source_index, source in enumerate(sources):
        if source_index and source_index % 500 == 0:
            _log(f"geometry cache progress: {source_index}/{len(sources)} sources")
        geometry = _extract_source(source, depsgraph)
        cached.append(geometry)
        count = len(geometry.triangles)
        if not count:
            continue
        result["centroids"].append(geometry.centroids)
        result["areas"].append(geometry.areas)
        result["normals"].append(geometry.normals)
        result["source_indices"].append(np.full(count, source_index, dtype=np.int32))
        result["triangle_indices"].append(np.arange(count, dtype=np.int64))
    if not result["centroids"]:
        raise RuntimeError("Scene contains no triangles")
    return (
        {key: np.concatenate(parts, axis=0) for key, parts in result.items()},
        tuple(cached),
    )


def _floor_components(geometry: dict[str, np.ndarray], config: ObservationPartitionConfig) -> list[np.ndarray]:
    centroids, normals, areas = geometry["centroids"], geometry["normals"], geometry["areas"]
    threshold = math.cos(math.radians(config.floor_max_slope_deg))
    candidates = np.flatnonzero((normals[:, 2] >= threshold) & (areas > 1e-10))
    if not len(candidates):
        return []
    selected = centroids[candidates]
    keys = np.column_stack((
        np.floor(selected[:, 0] / config.floor_component_cell_m),
        np.floor(selected[:, 1] / config.floor_component_cell_m),
        np.floor(selected[:, 2] / config.floor_height_band_m),
    )).astype(np.int64)
    cell_members: dict[tuple[int, int, int], list[int]] = {}
    for local_index, key_array in enumerate(keys):
        cell_members.setdefault(tuple(int(value) for value in key_array), []).append(local_index)
    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    for start in sorted(cell_members):
        if start in visited:
            continue
        pending = [start]
        visited.add(start)
        local_members: list[int] = []
        while pending:
            cell = pending.pop()
            local_members.extend(cell_members[cell])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        if neighbor in cell_members and neighbor not in visited:
                            z_delta = abs(neighbor[2] - cell[2]) * config.floor_height_band_m
                            if z_delta <= config.floor_height_band_m + 1e-8:
                                visited.add(neighbor)
                                pending.append(neighbor)
        global_indices = candidates[np.asarray(local_members, dtype=np.int64)]
        if float(areas[global_indices].sum()) >= config.min_floor_component_area_m2:
            components.append(global_indices)
    components.sort(key=lambda item: (-float(areas[item].sum()), float(np.median(centroids[item, 2]))))
    return components


def _sample_on_triangles(
    indices: np.ndarray, geometry: dict[str, np.ndarray],
    source_geometry: tuple[SourceGeometry, ...], rng,
) -> np.ndarray:
    areas = geometry["areas"][indices]
    selected_global = int(rng.choice(indices, p=areas / areas.sum()))
    source_index = int(geometry["source_indices"][selected_global])
    triangle_index = int(geometry["triangle_indices"][selected_global])
    cached = source_geometry[source_index]
    corners = cached.vertices[cached.triangles[triangle_index]]
    uv = rng.random(2)
    if uv.sum() > 1.0:
        uv = 1.0 - uv
    barycentric = np.asarray((1.0 - uv.sum(), uv[0], uv[1]))
    return barycentric @ corners


def _ray_clear(origin: np.ndarray, direction: np.ndarray, distance: float, depsgraph) -> bool:
    hit, _, _, _, _, _ = bpy.context.scene.ray_cast(
        depsgraph, Vector(tuple(origin)), Vector(tuple(direction)), distance=float(distance)
    )
    return not hit


def _valid_anchor(point: np.ndarray, config: ObservationPartitionConfig, depsgraph) -> bool:
    up = np.asarray((0.0, 0.0, 1.0))
    if not _ray_clear(point + up * 0.05, up, config.min_head_clearance_m, depsgraph):
        return False
    for height in (0.3, min(0.9, config.observer_height_m), config.observer_height_m):
        origin = point + up * height
        for index in range(config.body_clearance_ray_count):
            angle = 2.0 * math.pi * index / config.body_clearance_ray_count
            direction = np.asarray((math.cos(angle), math.sin(angle), 0.0))
            if not _ray_clear(origin, direction, config.min_body_clearance_radius_m, depsgraph):
                return False
    ground_radius = config.min_body_clearance_radius_m * 0.8
    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        origin = point + np.asarray((ground_radius * math.cos(angle), ground_radius * math.sin(angle), 0.35))
        hit, location, normal, _, _, _ = bpy.context.scene.ray_cast(
            depsgraph, Vector(tuple(origin)), Vector((0.0, 0.0, -1.0)), distance=0.7
        )
        if not hit or abs(float(location.z) - float(point[2])) > 0.3 or float(normal.z) < 0.7:
            return False
    return True


def _inside_sector(points: np.ndarray, anchor: np.ndarray, forward: np.ndarray, radius: float, half_angle: float, z_min: float, z_max: float) -> np.ndarray:
    relative = points - anchor
    distance = np.linalg.norm(relative[:, :2], axis=1)
    unit = np.divide(relative[:, :2], distance[:, None], out=np.zeros_like(relative[:, :2]), where=distance[:, None] > 1e-9)
    cosine = unit @ forward[:2]
    return (distance <= radius) & (cosine >= math.cos(half_angle)) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)


def _choose_direction(anchor: np.ndarray, geometry: dict[str, np.ndarray], config: ObservationPartitionConfig, rng):
    best = None
    half_angle = math.radians(config.horizontal_angle_deg * 0.5)
    z_min = anchor[2] - config.vertical_below_anchor_m
    z_max = anchor[2] + config.vertical_above_anchor_m
    for _ in range(config.direction_trials):
        yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        forward = np.asarray((math.cos(yaw), math.sin(yaw), 0.0))
        inside = _inside_sector(geometry["centroids"], anchor, forward, config.radius_m, half_angle, z_min, z_max)
        count = int(np.count_nonzero(inside))
        area = float(geometry["areas"][inside].sum())
        score = (count, area)
        if best is None or score > best[0]:
            best = (score, forward)
    return best


def _sample_anchors(components, geometry, source_geometry, depsgraph, config, scene_hash):
    seed = config.seed + int(scene_hash[:8], 16)
    rng = np.random.default_rng(seed % (2**63 - 1))
    component_areas = np.asarray([geometry["areas"][component].sum() for component in components], dtype=np.float64)
    anchors: list[tuple[AnchorCandidate, np.ndarray]] = []
    max_attempts = config.observations_per_scene * config.anchor_attempts_per_output
    for _ in range(max_attempts):
        if len(anchors) >= config.observations_per_scene or not len(components):
            break
        component_id = int(rng.choice(len(components), p=component_areas / component_areas.sum()))
        point = _sample_on_triangles(
            components[component_id], geometry, source_geometry, rng
        )
        if any(np.linalg.norm(point[:2] - np.asarray(item[0].floor[:2])) < config.min_anchor_spacing_m for item in anchors):
            continue
        if not _valid_anchor(point, config, depsgraph):
            continue
        direction = _choose_direction(point, geometry, config, rng)
        if direction is None or direction[0][0] < config.minimum_core_triangles:
            continue
        candidate = AnchorCandidate(tuple(float(value) for value in point), component_id, float(component_areas[component_id]))
        anchors.append((candidate, direction[1]))
    return anchors


def _select_source_triangles(
    geometry: SourceGeometry, anchor: np.ndarray, forward: np.ndarray,
    config: ObservationPartitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    expansion = config.camera_motion_radius_m + config.context_margin_m
    context_radius = config.radius_m + expansion
    context_half_angle = math.radians(config.horizontal_angle_deg * 0.5) + math.atan2(expansion, config.radius_m)
    core_half_angle = math.radians(config.horizontal_angle_deg * 0.5)
    core_min = anchor[2] - config.vertical_below_anchor_m
    core_max = anchor[2] + config.vertical_above_anchor_m
    context_min = core_min - config.context_margin_m
    context_max = core_max + config.context_margin_m
    if geometry.bounds_max[2] < context_min or geometry.bounds_min[2] > context_max:
        empty = np.zeros(len(geometry.triangles), dtype=bool)
        return empty, empty.copy()
    delta_x = max(
        float(geometry.bounds_min[0] - anchor[0]), 0.0,
        float(anchor[0] - geometry.bounds_max[0]),
    )
    delta_y = max(
        float(geometry.bounds_min[1] - anchor[1]), 0.0,
        float(anchor[1] - geometry.bounds_max[1]),
    )
    if math.hypot(delta_x, delta_y) > context_radius:
        empty = np.zeros(len(geometry.triangles), dtype=bool)
        return empty, empty.copy()
    context = _inside_sector(
        geometry.centroids, anchor, forward, context_radius,
        context_half_angle, context_min, context_max,
    )
    triangle_vertices = geometry.triangles
    for corner_index in range(3):
        corners = geometry.vertices[triangle_vertices[:, corner_index]]
        context |= _inside_sector(
            corners, anchor, forward, context_radius,
            context_half_angle, context_min, context_max,
        )
    for left, right in ((0, 1), (1, 2), (2, 0)):
        midpoint = (
            geometry.vertices[triangle_vertices[:, left]]
            + geometry.vertices[triangle_vertices[:, right]]
        ) * 0.5
        context |= _inside_sector(
            midpoint, anchor, forward, context_radius,
            context_half_angle, context_min, context_max,
        )
    core = _inside_sector(
        geometry.centroids, anchor, forward, config.radius_m,
        core_half_angle, core_min, core_max,
    )
    return context, core


def _sanitized_material_copy(
    material, resolved: dict[int, object], owned: list[object],
    removed_images: set[str],
):
    key = int(material.as_pointer())
    if key in resolved:
        return resolved[key]
    invalid_names = []
    if material.use_nodes and material.node_tree is not None:
        for node in material.node_tree.nodes:
            image = getattr(node, "image", None)
            if node.type == "TEX_IMAGE" and image is not None:
                size = tuple(int(value) for value in image.size)
                if len(size) < 2 or size[0] <= 0 or size[1] <= 0:
                    invalid_names.append(str(image.name))
    if not invalid_names:
        resolved[key] = material
        return material
    copied = material.copy()
    copied.name = f"SceneCompose_{material.name}"
    for node in list(copied.node_tree.nodes):
        image = getattr(node, "image", None)
        if node.type == "TEX_IMAGE" and image is not None and str(image.name) in invalid_names:
            copied.node_tree.nodes.remove(node)
    removed_images.update(invalid_names)
    owned.append(copied)
    resolved[key] = copied
    return copied


def _create_partition_object(
    source_geometry: tuple[SourceGeometry, ...], anchor: np.ndarray,
    forward: np.ndarray, config: ObservationPartitionConfig,
):
    vertex_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    smooth_parts: list[np.ndarray] = []
    material_index_parts: list[np.ndarray] = []
    uv_layer_names = sorted({
        geometry.uv_name for geometry in source_geometry
        if geometry.uv_name is not None and geometry.loop_uv is not None
    })
    uv_parts: dict[str, list[np.ndarray]] = {name: [] for name in uv_layer_names}
    mapping_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "source_object_index", "source_instance_index", "source_polygon_index",
            "is_core", "is_context",
        )
    }
    resolved_materials: dict[int, object] = {}
    owned_materials: list[object] = []
    output_materials: list[object] = []
    output_material_slots: dict[int, int] = {}
    removed_images: set[str] = set()
    selected_source_count = 0
    vertex_offset = 0

    def material_slot(material) -> int:
        key = int(material.as_pointer()) if material is not None else 0
        if key not in output_material_slots:
            if material is None:
                copied = bpy.data.materials.new("SceneCompose_MissingMaterial")
                copied.diffuse_color = (0.5, 0.5, 0.5, 1.0)
                owned_materials.append(copied)
                resolved_materials[key] = copied
            else:
                copied = _sanitized_material_copy(
                    material, resolved_materials, owned_materials, removed_images
                )
            output_material_slots[key] = len(output_materials)
            output_materials.append(copied)
        return output_material_slots[key]

    for source_index, geometry in enumerate(source_geometry):
        if source_index and source_index % 500 == 0:
            _log(
                f"partition build progress: {source_index}/{len(source_geometry)} sources; "
                f"selected={selected_source_count}"
            )
        if not len(geometry.triangles):
            continue
        selected, core_all = _select_source_triangles(geometry, anchor, forward, config)
        selected_indices = np.flatnonzero(selected)
        if not len(selected_indices):
            continue
        selected_source_count += 1
        selected_triangles = geometry.triangles[selected_indices]
        unique_vertices, inverse = np.unique(selected_triangles.ravel(), return_inverse=True)
        output_triangles = inverse.reshape((-1, 3)).astype(np.int32)
        vertex_parts.append(geometry.vertices[unique_vertices])
        triangle_parts.append(output_triangles + vertex_offset)
        vertex_offset += len(unique_vertices)
        selected_polygons = geometry.polygons[selected_indices]
        smooth_parts.append(geometry.polygon_smooth[selected_polygons])
        if config.include_materials:
            source_slots = geometry.polygon_material[selected_polygons]
            target_slots = np.zeros(len(source_slots), dtype=np.int32)
            for old_slot in np.unique(source_slots):
                old_slot_int = int(old_slot)
                material = (
                    geometry.materials[old_slot_int]
                    if 0 <= old_slot_int < len(geometry.materials) else None
                )
                target_slots[source_slots == old_slot_int] = material_slot(material)
            material_index_parts.append(target_slots)
        else:
            material_index_parts.append(np.zeros(len(selected_indices), dtype=np.int32))
        selected_uv = (
            geometry.loop_uv[geometry.loops[selected_indices].ravel()]
            if geometry.loop_uv is not None
            else np.zeros((len(selected_indices) * 3, 2), dtype=np.float32)
        )
        for name in uv_layer_names:
            # Mirroring the source's active UV into every retained active-layer
            # name preserves both implicit and explicitly named material lookups.
            uv_parts[name].append(selected_uv)
        mapping_parts["source_object_index"].append(
            np.full(len(selected_indices), source_index, dtype=np.int32)
        )
        mapping_parts["source_instance_index"].append(
            np.full(len(selected_indices), source_index, dtype=np.int32)
        )
        mapping_parts["source_polygon_index"].append(selected_polygons.astype(np.int64))
        mapping_parts["is_core"].append(core_all[selected_indices])
        mapping_parts["is_context"].append(np.ones(len(selected_indices), dtype=bool))

    if not triangle_parts:
        return None, None, None
    vertices = np.concatenate(vertex_parts, axis=0)
    triangles = np.concatenate(triangle_parts, axis=0)
    output_mesh = bpy.data.meshes.new("observation_partition_merged_mesh")
    output_mesh.from_pydata(vertices, [], triangles)
    smooth = np.concatenate(smooth_parts)
    output_mesh.polygons.foreach_set("use_smooth", smooth)
    if config.include_materials:
        for material in output_materials:
            output_mesh.materials.append(material)
        if output_materials:
            output_mesh.polygons.foreach_set(
                "material_index", np.concatenate(material_index_parts)
            )
    for name, parts in uv_parts.items():
        target_uv = output_mesh.uv_layers.new(name=name)
        target_uv.data.foreach_set("uv", np.concatenate(parts, axis=0).ravel())
    output_mesh.update(calc_edges=True)
    output_name = "partition_merged"
    output_object = bpy.data.objects.new(output_name, output_mesh)
    bpy.context.scene.collection.objects.link(output_object)
    mapping = {key: np.concatenate(parts) for key, parts in mapping_parts.items()}
    triangle_count = len(triangles)
    mapping["output_object_name"] = np.full(
        triangle_count, output_name, dtype=f"<U{len(output_name)}"
    )
    mapping["output_triangle_index_within_object"] = np.arange(
        triangle_count, dtype=np.int64
    )
    stats = {
        "triangle_count": triangle_count,
        "vertex_count": len(vertices),
        "source_count": selected_source_count,
        "material_count": len(output_materials),
        "removed_images": sorted(removed_images),
        "owned_materials": owned_materials,
    }
    return output_object, mapping, stats


def _export_glb(path: Path, objects: list[bpy.types.Object], include_materials: bool) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    requested = {
        "filepath": str(path), "export_format": "GLB", "use_selection": True,
        "export_yup": False, "export_materials": "EXPORT" if include_materials else "NONE",
    }
    supported = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    result = bpy.ops.export_scene.gltf(**{key: value for key, value in requested.items() if key in supported})
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed with status: {sorted(result)}")


def _remove_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _remove_materials(materials: list[object]) -> None:
    for material in materials:
        if material is not None and material.users == 0:
            bpy.data.materials.remove(material)


def _export_observation(
    staging, source_reference, scene_hash, anchor_candidate, forward,
    source_geometry, config, observation_index, observation_total,
):
    anchor = np.asarray(anchor_candidate.floor, dtype=np.float64)
    right = np.asarray((-forward[1], forward[0], 0.0))
    config_digest = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = observation_id(
        scene_hash, tuple(anchor), tuple(forward), config.radius_m,
        config.horizontal_angle_deg, config_digest,
    )
    directory = staging / identity
    partition_dir = directory / "partition"
    partition_dir.mkdir(parents=True)
    objects: list[bpy.types.Object] = []
    owned_materials: list[object] = []
    try:
        build_started = time.perf_counter()
        _log(
            f"observation {observation_index}/{observation_total} {identity}: "
            "building merged partition"
        )
        obj, mapping, stats = _create_partition_object(
            source_geometry, anchor, forward, config
        )
        if obj is None or mapping is None or stats is None:
            raise RuntimeError(f"{identity} selected no geometry")
        objects.append(obj)
        owned_materials = stats.pop("owned_materials")
        removed = stats["removed_images"]
        _log(
            f"observation {observation_index}/{observation_total} {identity}: "
            f"mesh ready in {_seconds(build_started)}; "
            f"triangles={stats['triangle_count']}, vertices={stats['vertex_count']}, "
            f"sources={stats['source_count']}, materials={stats['material_count']}, "
            f"invalid_images_removed={len(removed)}"
        )
        if removed:
            _log(f"{identity}: removed invalid images: {', '.join(removed[:16])}")
        geometry_path = partition_dir / "scene_partition.glb"
        export_started = time.perf_counter()
        _log(
            f"observation {observation_index}/{observation_total} {identity}: "
            "exporting one merged GLB object"
        )
        _export_glb(geometry_path, objects, config.include_materials)
        _log(
            f"observation {observation_index}/{observation_total} {identity}: "
            f"GLB exported in {_seconds(export_started)}; bytes={geometry_path.stat().st_size}"
        )
        output_count = len(mapping["is_core"])
        mapping_path = partition_dir / "source_faces.npz"
        arrays = dict(mapping)
        arrays["output_triangle_index"] = np.arange(output_count, dtype=np.int64)
        np.savez_compressed(mapping_path, **arrays)
        region = ObservationRegion(
            observation_id=identity,
            source_scene=source_reference,
            source_sha256=scene_hash,
            anchor_floor=tuple(float(value) for value in anchor),
            anchor_eye=(float(anchor[0]), float(anchor[1]), float(anchor[2] + config.observer_height_m)),
            reference_forward=tuple(float(value) for value in forward),
            reference_right=tuple(float(value) for value in right),
            up=(0.0, 0.0, 1.0),
            radius_m=config.radius_m,
            horizontal_angle_deg=config.horizontal_angle_deg,
            vertical_range_world=(float(anchor[2] - config.vertical_below_anchor_m), float(anchor[2] + config.vertical_above_anchor_m)),
            camera_motion_radius_m=config.camera_motion_radius_m,
            context_margin_m=config.context_margin_m,
            partition_geometry="partition/scene_partition.glb",
            source_faces="partition/source_faces.npz",
            core_triangle_count=int(np.count_nonzero(arrays["is_core"])),
            context_triangle_count=int(output_count),
            quality={"floor_component_id": anchor_candidate.component_id, "floor_component_area_m2": anchor_candidate.component_area_m2},
        )
        write_json_atomic(directory / "observation.json", region.to_dict())
        _log(
            f"observation {observation_index}/{observation_total} {identity}: complete; "
            f"core_triangles={region.core_triangle_count}, "
            f"context_triangles={region.context_triangle_count}"
        )
        return region
    finally:
        _remove_objects(objects)
        _remove_materials(owned_materials)


def _publish(staging: Path, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"Output exists; use --force to replace it: {output}")
    backup = None
    if output.exists():
        backup = output.with_name(f"{output.name}.backup-{uuid.uuid4().hex}")
        output.replace(backup)
    try:
        staging.replace(output)
    except Exception:
        if backup is not None and not output.exists():
            backup.replace(output)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def main() -> None:
    args = _arguments()
    source = args.scene.resolve()
    output = args.output.resolve()
    config = ObservationPartitionConfig.from_json(args.config.resolve())
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        total_started = time.perf_counter()
        stage_started = time.perf_counter()
        _log(f"importing FBX: {source}")
        _import_scene(source, config)
        _log(f"FBX imported in {_seconds(stage_started)}")
        depsgraph = bpy.context.evaluated_depsgraph_get()
        stage_started = time.perf_counter()
        sources = _mesh_sources(depsgraph)
        _log(f"caching evaluated geometry for {len(sources)} sources")
        geometry, source_geometry = _collect_geometry(sources, depsgraph)
        cache_bytes = sum(
            array.nbytes
            for item in source_geometry
            for array in (
                (
                    item.vertices, item.triangles, item.loops, item.polygons,
                    item.centroids, item.normals, item.areas, item.polygon_smooth,
                    item.polygon_material,
                ) + ((item.loop_uv,) if item.loop_uv is not None else ())
            )
        )
        _log(
            f"geometry cached in {_seconds(stage_started)}; "
            f"triangles={len(geometry['centroids'])}, cache={cache_bytes / (1024 ** 2):.1f} MiB"
        )
        stage_started = time.perf_counter()
        components = _floor_components(geometry, config)
        _log(f"found {len(components)} floor components in {_seconds(stage_started)}")
        stage_started = time.perf_counter()
        scene_hash = _sha256_file(source)
        anchors = _sample_anchors(
            components, geometry, source_geometry, depsgraph, config, scene_hash
        )
        _log(
            f"selected {len(anchors)}/{config.observations_per_scene} anchors "
            f"in {_seconds(stage_started)}"
        )
        source_reference = args.source_relative or str(source)
        regions = []
        for index, (anchor, forward) in enumerate(anchors, start=1):
            regions.append(
                _export_observation(
                    staging, source_reference, scene_hash, anchor, forward,
                    source_geometry, config, index, len(anchors),
                )
            )
        payload = {
            "schema_version": 1, "scene_id": _safe_scene_id(source),
            "source_scene": source_reference, "source_sha256": scene_hash,
            "config": config.to_dict(), "observation_count": len(regions),
            "observations": [region.to_dict() for region in regions],
            "warnings": [] if len(regions) == config.observations_per_scene else [f"Generated {len(regions)} of {config.observations_per_scene} requested observations"],
        }
        write_json_atomic(staging / "observations.json", payload)
        write_json_atomic(staging / "summary.json", {
            "schema_version": 1, "candidate_floor_component_count": len(components),
            "requested_observations": config.observations_per_scene,
            "generated_observations": len(regions), "source_triangle_count": len(geometry["centroids"]),
        })
        _publish(staging, output, args.force)
        _log(
            f"published {len(regions)} observations to {output} "
            f"in {_seconds(total_started)}"
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
