from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import bpy
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scenecompose.config import PartitionConfig
from scenecompose.contracts import Bounds3D, SceneRegion
from scenecompose.regions import plan_regions, sha256_file, write_regions_json


@dataclass(frozen=True)
class MeshSource:
    source_object: bpy.types.Object
    source_object_name: str
    instance_name: str
    matrix_world: tuple[float, ...]
    is_instance: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _safe_scene_id(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-.")
    return value or "scene"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _prepare_staging(output: Path, force: bool) -> tuple[Path, Path | None]:
    output = output.resolve()
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError(f"Refusing unsafe output path: {output}")
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex[:10]}")
    staging.mkdir(parents=True, exist_ok=False)
    backup = None
    if output.exists():
        if not force:
            staging.rmdir()
            raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")
        backup = output.with_name(
            f"{output.name}.backup-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        )
        output.replace(backup)
    return staging, backup


def _import_fbx(scene_path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.fbx(filepath=str(scene_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"FBX import failed with status: {sorted(result)}")


def _meter_scale(config: PartitionConfig) -> float:
    if config.meters_per_blender_unit is not None:
        return config.meters_per_blender_unit
    scale = float(bpy.context.scene.unit_settings.scale_length)
    return scale if scale > 0 else 1.0


def _mesh_sources(depsgraph: bpy.types.Depsgraph) -> list[MeshSource]:
    sources: list[MeshSource] = []
    for instance in depsgraph.object_instances:
        evaluated = instance.object
        if evaluated.type != "MESH":
            continue
        original = evaluated.original
        matrix = tuple(float(value) for row in instance.matrix_world for value in row)
        persistent = tuple(int(value) for value in instance.persistent_id)
        instance_token = hashlib.sha256(
            repr((persistent, matrix)).encode("utf-8")
        ).hexdigest()[:12]
        instance_name = (
            f"{original.name_full}__instance_{instance_token}"
            if instance.is_instance
            else original.name_full
        )
        sources.append(
            MeshSource(
                source_object=original,
                source_object_name=original.name_full,
                instance_name=instance_name,
                matrix_world=matrix,
                is_instance=bool(instance.is_instance),
            )
        )
    sources.sort(
        key=lambda source: (
            source.source_object_name.casefold(),
            source.instance_name,
            source.matrix_world,
        )
    )
    if not sources:
        raise ValueError("Imported scene contains no mesh objects")
    return sources


def _extract_arrays(
    source: MeshSource,
    depsgraph: bpy.types.Depsgraph,
    meter_scale: float,
) -> tuple[
    bpy.types.Object,
    bpy.types.Mesh,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    evaluated = source.source_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    mesh.calc_loop_triangles()

    vertices_local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    mesh.vertices.foreach_get("co", vertices_local.ravel())
    matrix_world = np.asarray(source.matrix_world, dtype=np.float64).reshape((4, 4))
    homogeneous = np.ones((vertices_local.shape[0], 4), dtype=np.float64)
    homogeneous[:, :3] = vertices_local
    vertices_world_m = (homogeneous @ matrix_world.T)[:, :3] * meter_scale

    triangle_vertices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    triangle_loops = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    polygon_indices = np.empty(len(mesh.loop_triangles), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", triangle_vertices.ravel())
    mesh.loop_triangles.foreach_get("loops", triangle_loops.ravel())
    mesh.loop_triangles.foreach_get("polygon_index", polygon_indices)

    corners = vertices_world_m[triangle_vertices]
    centroids = np.mean(corners, axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    return (
        evaluated,
        mesh,
        vertices_world_m,
        triangle_vertices,
        triangle_loops,
        polygon_indices,
        np.column_stack((centroids, areas)),
    )


def _collect_scene_geometry(
    sources: list[MeshSource],
    depsgraph: bpy.types.Depsgraph,
    meter_scale: float,
) -> tuple[np.ndarray, np.ndarray, Bounds3D, list[dict]]:
    centroid_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    object_stats: list[dict] = []

    for source_index, source in enumerate(sources):
        evaluated, mesh, vertices, _, _, _, triangle_data = _extract_arrays(
            source, depsgraph, meter_scale
        )
        try:
            if triangle_data.shape[0] == 0:
                continue
            centroid_parts.append(triangle_data[:, :2])
            area_parts.append(triangle_data[:, 3].astype(np.float64, copy=False))
            bounds_min = np.minimum(bounds_min, np.min(vertices, axis=0))
            bounds_max = np.maximum(bounds_max, np.max(vertices, axis=0))
            object_stats.append(
                {
                    "source_index": source_index,
                    "source_object_name": source.source_object_name,
                    "instance_name": source.instance_name,
                    "is_instance": source.is_instance,
                    "matrix_world_source_units": source.matrix_world,
                    "triangle_count": int(triangle_data.shape[0]),
                    "vertex_count": int(vertices.shape[0]),
                }
            )
        finally:
            evaluated.to_mesh_clear()

    if not centroid_parts:
        raise ValueError("Imported mesh objects contain no triangles")
    centroids = np.concatenate(centroid_parts, axis=0)
    areas = np.concatenate(area_parts, axis=0)
    bounds = Bounds3D(tuple(bounds_min.tolist()), tuple(bounds_max.tolist()))
    return centroids, areas, bounds, object_stats


def _inside_xy(points: np.ndarray, bounds: Bounds3D) -> np.ndarray:
    return (
        (points[:, 0] >= bounds.minimum[0])
        & (points[:, 0] <= bounds.maximum[0])
        & (points[:, 1] >= bounds.minimum[1])
        & (points[:, 1] <= bounds.maximum[1])
    )


def _create_region_object(
    source: MeshSource,
    source_index: int,
    depsgraph: bpy.types.Depsgraph,
    meter_scale: float,
    region: SceneRegion,
    include_materials: bool,
) -> tuple[bpy.types.Object | None, dict[str, np.ndarray]]:
    evaluated, mesh, vertices, triangles, triangle_loops, polygon_indices, triangle_data = (
        _extract_arrays(source, depsgraph, meter_scale)
    )
    try:
        selected = _inside_xy(triangle_data[:, :2], region.context_bounds)
        if not np.any(selected):
            return None, {}

        selected_triangles = triangles[selected]
        unique_vertices, inverse = np.unique(selected_triangles.ravel(), return_inverse=True)
        output_vertices = vertices[unique_vertices].astype(np.float32, copy=False)
        output_triangles = inverse.reshape((-1, 3)).astype(np.int32, copy=False)
        selected_polygon_indices = polygon_indices[selected]
        selected_triangle_loops = triangle_loops[selected]

        output_mesh = bpy.data.meshes.new(f"{region.region_id}_{source.instance_name}_mesh")
        output_mesh.vertices.add(output_vertices.shape[0])
        output_mesh.vertices.foreach_set("co", output_vertices.ravel())
        output_mesh.loops.add(output_triangles.size)
        output_mesh.loops.foreach_set("vertex_index", output_triangles.ravel())
        output_mesh.polygons.add(output_triangles.shape[0])
        output_mesh.polygons.foreach_set(
            "loop_start", np.arange(output_triangles.shape[0], dtype=np.int32) * 3
        )
        output_mesh.polygons.foreach_set(
            "loop_total", np.full(output_triangles.shape[0], 3, dtype=np.int32)
        )

        if len(mesh.polygons):
            smooth_flags = np.empty(len(mesh.polygons), dtype=np.bool_)
            mesh.polygons.foreach_get("use_smooth", smooth_flags)
            output_mesh.polygons.foreach_set("use_smooth", smooth_flags[selected_polygon_indices])
            if include_materials:
                material_indices = np.empty(len(mesh.polygons), dtype=np.int32)
                mesh.polygons.foreach_get("material_index", material_indices)
                output_mesh.polygons.foreach_set(
                    "material_index", material_indices[selected_polygon_indices]
                )

        if include_materials:
            for material in mesh.materials:
                output_mesh.materials.append(material)

        if mesh.uv_layers.active is not None:
            source_uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
            mesh.uv_layers.active.data.foreach_get("uv", source_uv.ravel())
            output_uv = source_uv[selected_triangle_loops.ravel()]
            target_uv = output_mesh.uv_layers.new(name=mesh.uv_layers.active.name)
            target_uv.data.foreach_set("uv", output_uv.ravel())

        output_mesh.update(calc_edges=True)
        output_object = bpy.data.objects.new(
            f"{region.region_id}_{source.instance_name}", output_mesh
        )
        bpy.context.scene.collection.objects.link(output_object)

        selected_centroids = triangle_data[selected, :2]
        mapping = {
            "source_index": np.full(
                selected_polygon_indices.shape[0], source_index, dtype=np.int32
            ),
            "source_polygon_index": selected_polygon_indices.astype(np.int64, copy=False),
            "is_core": _inside_xy(selected_centroids, region.core_bounds),
        }
        return output_object, mapping
    finally:
        evaluated.to_mesh_clear()


def _export_glb(path: Path, objects: list[bpy.types.Object], include_materials: bool) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    requested = {
        "filepath": str(path),
        "export_format": "GLB",
        "use_selection": True,
        "export_yup": False,
        "export_materials": "EXPORT" if include_materials else "NONE",
    }
    supported = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    result = bpy.ops.export_scene.gltf(**{k: v for k, v in requested.items() if k in supported})
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed with status: {sorted(result)}")


def _export_blend(path: Path, objects: list[bpy.types.Object]) -> None:
    region_scene = bpy.data.scenes.new(path.stem)
    try:
        for obj in objects:
            region_scene.collection.objects.link(obj)
        bpy.data.libraries.write(str(path), {region_scene}, path_remap="RELATIVE")
    finally:
        bpy.data.scenes.remove(region_scene)


def _remove_region_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _export_region(
    staging: Path,
    region: SceneRegion,
    sources: list[MeshSource],
    depsgraph: bpy.types.Depsgraph,
    meter_scale: float,
    config: PartitionConfig,
) -> SceneRegion:
    region_directory = staging / "regions" / region.region_id
    region_directory.mkdir(parents=True, exist_ok=False)
    if region.is_identity:
        _write_json(
            region_directory / "manifest.json",
            {
                "schema_version": 1,
                "region": region.to_dict(),
                "geometry_reference": "source_scene",
            },
        )
        return region

    output_objects: list[bpy.types.Object] = []
    mappings: dict[str, list[np.ndarray]] = {
        "output_object_index": [],
        "source_index": [],
        "source_polygon_index": [],
        "is_core": [],
    }
    try:
        for source_index, source in enumerate(sources):
            output_object, mapping = _create_region_object(
                source,
                source_index,
                depsgraph,
                meter_scale,
                region,
                config.include_materials,
            )
            if output_object is None:
                continue
            output_index = len(output_objects)
            output_objects.append(output_object)
            count = mapping["source_polygon_index"].shape[0]
            mappings["output_object_index"].append(
                np.full(count, output_index, dtype=np.int32)
            )
            for key in ("source_index", "source_polygon_index", "is_core"):
                mappings[key].append(mapping[key])

        if not output_objects:
            raise RuntimeError(f"Region {region.region_id} exported no geometry")

        extension = ".glb" if config.export_format == "glb" else ".blend"
        geometry_path = region_directory / f"scene_region{extension}"
        if config.export_format == "glb":
            _export_glb(geometry_path, output_objects, config.include_materials)
        else:
            _export_blend(geometry_path, output_objects)

        mapping_path = region_directory / "source_faces.npz"
        np.savez_compressed(
            mapping_path,
            **{
                key: np.concatenate(parts, axis=0)
                for key, parts in mappings.items()
            },
        )
        relative_geometry = geometry_path.relative_to(staging).as_posix()
        relative_mapping = mapping_path.relative_to(staging).as_posix()
        exported = replace(
            region,
            geometry_path=relative_geometry,
            source_faces_path=relative_mapping,
        )
        _write_json(
            region_directory / "manifest.json",
            {
                "schema_version": 1,
                "region": exported.to_dict(),
                "exported_object_count": len(output_objects),
                "output_object_names": [obj.name_full for obj in output_objects],
                "exported_triangle_count": int(
                    sum(part.shape[0] for part in mappings["is_core"])
                ),
                "core_triangle_count": int(
                    sum(np.count_nonzero(part) for part in mappings["is_core"])
                ),
            },
        )
        return exported
    finally:
        _remove_region_objects(output_objects)


def main() -> None:
    args = _arguments()
    scene_path = args.scene.resolve()
    output_path = args.output.resolve()
    config = PartitionConfig.from_json(args.config.resolve())
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if scene_path.suffix.casefold() != ".fbx":
        raise ValueError(f"Only FBX scenes are accepted: {scene_path}")

    staging, backup = _prepare_staging(output_path, args.force)
    started = time.time()
    try:
        source_sha256 = sha256_file(scene_path)
        _import_fbx(scene_path)
        meter_scale = _meter_scale(config)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        sources = _mesh_sources(depsgraph)
        centroids, areas, scene_bounds, object_stats = _collect_scene_geometry(
            sources, depsgraph, meter_scale
        )
        regions = plan_regions(
            centroids,
            areas,
            scene_bounds,
            source_sha256,
            config,
            meters_per_source_unit=meter_scale,
        )

        exported_regions = [
            _export_region(
                staging,
                region,
                sources,
                depsgraph,
                meter_scale,
                config,
            )
            for region in regions
        ]
        scene_id = _safe_scene_id(scene_path)
        write_regions_json(
            staging / "regions.json",
            scene_id=scene_id,
            source_path=scene_path,
            source_sha256=source_sha256,
            config=config,
            regions=exported_regions,
            meters_per_blender_unit=meter_scale,
            sources=object_stats,
        )
        _write_json(
            staging / "summary.json",
            {
                "schema_version": 1,
                "scene_id": scene_id,
                "source_path": str(scene_path),
                "source_sha256": source_sha256,
                "mesh_source_count": len(sources),
                "instanced_mesh_source_count": sum(source.is_instance for source in sources),
                "triangle_count": int(centroids.shape[0]),
                "surface_area_m2": float(np.sum(areas, dtype=np.float64)),
                "scene_bounds_m": {
                    "minimum": scene_bounds.minimum,
                    "maximum": scene_bounds.maximum,
                },
                "region_count": len(exported_regions),
                "identity_region": len(exported_regions) == 1
                and exported_regions[0].is_identity,
                "over_budget_region_count": sum(
                    int(region.over_budget) for region in exported_regions
                ),
                "meters_per_blender_unit": meter_scale,
                "objects": object_stats,
                "elapsed_seconds": time.time() - started,
            },
        )
        staging.replace(output_path)
    except Exception:
        if staging.exists():
            failure_path = staging / "FAILED.txt"
            failure_path.write_text(
                "Partitioning failed. The staging directory is retained for diagnosis.\n",
                encoding="utf-8",
            )
        if backup is not None and backup.exists() and not output_path.exists():
            backup.replace(output_path)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


if __name__ == "__main__":
    main()
