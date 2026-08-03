from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("geometry", "views", "all"), default="all")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(values)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _import_scene(path: Path, meters_per_blender_unit: float | None) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = path.suffix.casefold()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported scene format: {suffix}")
    for obj in list(bpy.data.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    if meters_per_blender_unit is not None and meters_per_blender_unit != 1.0:
        scale = Matrix.Scale(float(meters_per_blender_unit), 4)
        for obj in bpy.context.scene.objects:
            if obj.parent is None:
                obj.matrix_world = scale @ obj.matrix_world


def _material_rgb(obj: bpy.types.Object, material_index: int, neutral: tuple[int, int, int]) -> np.ndarray:
    if material_index < len(obj.material_slots):
        material = obj.material_slots[material_index].material
        if material is not None:
            color = material.diffuse_color
            return np.clip(np.asarray(color[:3]) * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(neutral, dtype=np.uint8)


def _triangle_corner_rgb(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    tri: bpy.types.MeshLoopTriangle,
    neutral: tuple[int, int, int],
    image_cache: dict[str, np.ndarray],
) -> np.ndarray:
    base = _material_rgb(obj, tri.material_index, neutral)
    if tri.material_index >= len(obj.material_slots):
        return np.repeat(base[None, :], 3, axis=0)
    material = obj.material_slots[tri.material_index].material
    if material is None or not material.use_nodes or mesh.uv_layers.active is None:
        return np.repeat(base[None, :], 3, axis=0)
    image = next(
        (
            node.image for node in material.node_tree.nodes
            if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None
        ),
        None,
    )
    if image is None or image.size[0] <= 0 or image.size[1] <= 0:
        return np.repeat(base[None, :], 3, axis=0)
    if image.name not in image_cache:
        pixels = np.asarray(image.pixels[:], dtype=np.float32)
        image_cache[image.name] = pixels.reshape((image.size[1], image.size[0], 4))
    pixels = image_cache[image.name]
    uv_data = mesh.uv_layers.active.data
    result = []
    for loop_index in tri.loops:
        uv = uv_data[loop_index].uv
        x = min(pixels.shape[1] - 1, max(0, int((float(uv.x) % 1.0) * pixels.shape[1])))
        y = min(pixels.shape[0] - 1, max(0, int((float(uv.y) % 1.0) * pixels.shape[0])))
        result.append(np.clip(pixels[y, x, :3] * 255.0, 0, 255))
    return np.asarray(result, dtype=np.uint8)


def _extract_geometry(neutral: tuple[int, int, int]) -> dict[str, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    triangle_rgb: list[np.ndarray] = []
    triangle_object: list[int] = []
    triangle_polygon: list[int] = []
    object_names: list[str] = []
    triangle_corner_rgb: list[np.ndarray] = []
    image_cache: dict[str, np.ndarray] = {}
    vertex_offset = 0
    for instance in depsgraph.object_instances:
        source = instance.object
        if source.type != "MESH":
            continue
        evaluated = source.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            world = instance.matrix_world.copy()
            object_id = len(object_names)
            object_names.append(source.name)
            local_vertices = np.asarray([tuple(world @ vertex.co) for vertex in mesh.vertices], dtype=np.float32)
            vertices.append(local_vertices)
            for tri in mesh.loop_triangles:
                triangles.append(np.asarray(tri.vertices, dtype=np.int64) + vertex_offset)
                triangle_rgb.append(_material_rgb(source, tri.material_index, neutral))
                triangle_corner_rgb.append(
                    _triangle_corner_rgb(source, mesh, tri, neutral, image_cache)
                )
                triangle_object.append(object_id)
                triangle_polygon.append(int(tri.polygon_index))
            vertex_offset += len(local_vertices)
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise RuntimeError("Imported scene contains no evaluated mesh triangles")
    return {
        "vertices": np.concatenate(vertices, axis=0),
        "triangles": np.stack(triangles),
        "triangle_rgb": np.stack(triangle_rgb),
        "triangle_corner_rgb": np.stack(triangle_corner_rgb),
        "triangle_object": np.asarray(triangle_object, dtype=np.int32),
        "triangle_polygon": np.asarray(triangle_polygon, dtype=np.int32),
        "object_names": np.asarray(object_names, dtype=np.str_),
    }


def _sample_surface(geometry: dict[str, np.ndarray], count: int, seed: int) -> dict[str, np.ndarray]:
    vertices = geometry["vertices"]
    triangles = geometry["triangles"]
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    if not np.any(valid):
        raise RuntimeError("Scene mesh has no non-degenerate triangles")
    probability = np.where(valid, double_area, 0.0)
    probability /= probability.sum()
    rng = np.random.default_rng(seed)
    triangle_ids = rng.choice(len(triangles), size=count, replace=True, p=probability)
    random_uv = rng.random((count, 2))
    reflected = random_uv.sum(axis=1) > 1.0
    random_uv[reflected] = 1.0 - random_uv[reflected]
    barycentric = np.column_stack((1.0 - random_uv.sum(axis=1), random_uv)).astype(np.float32)
    selected = corners[triangle_ids]
    points = np.einsum("ni,nij->nj", barycentric, selected).astype(np.float32)
    normals = cross[triangle_ids] / np.maximum(double_area[triangle_ids, None], 1e-12)
    colors = np.einsum(
        "ni,nij->nj", barycentric, geometry["triangle_corner_rgb"][triangle_ids].astype(np.float32)
    )
    return {
        "points": points, "colors": np.clip(colors, 0, 255).astype(np.uint8), "normals": normals.astype(np.float32),
        "triangle_ids": triangle_ids.astype(np.int64), "barycentric": barycentric,
    }


def _write_point_ply(path: Path, samples: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points, colors, normals = samples["points"], samples["colors"], samples["normals"]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    dtype = np.dtype([(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz")] + [(name, "u1") for name in ("red", "green", "blue")])
    packed = np.empty(len(points), dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        packed[name] = points[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        packed[name] = normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        packed[name] = colors[:, index]
    temporary = path.with_suffix(".ply.tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        packed.tofile(stream)
    temporary.replace(path)


def _look_at(location: Vector, target: Vector) -> Matrix:
    direction = target - location
    return direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()


def _camera_candidates(points: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray, render: dict[str, object]) -> list[tuple[Vector, Vector]]:
    center = (bounds_min + bounds_max) * 0.5
    extent = np.maximum(bounds_max - bounds_min, 1.0)
    cell_size = float(render["cell_size_m"])
    cells = np.floor((points[:, :2] - bounds_min[:2]) / cell_size).astype(np.int32)
    unique_cells, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    occupied = [index for index, count in enumerate(counts) if count >= 8]
    targets = []
    for index in occupied:
        selected = points[inverse == index]
        targets.append(Vector(tuple(np.median(selected, axis=0).astype(float))))
    if not targets:
        targets = [Vector(tuple(center.astype(float)))]
    radius = max(cell_size * 0.8, float(max(extent[0], extent[1]) * 0.15), 1.5)
    candidates: list[tuple[Vector, Vector]] = []
    for target in targets:
        for elevation in render["elevations_deg"]:
            elevation_rad = math.radians(float(elevation))
            for azimuth in render["azimuths_deg"]:
                azimuth_rad = math.radians(float(azimuth))
                offset = Vector((radius * math.cos(elevation_rad) * math.cos(azimuth_rad), radius * math.cos(elevation_rad) * math.sin(azimuth_rad), radius * math.sin(elevation_rad)))
                candidates.append((target + offset, target))
    return candidates[: int(render["max_views"])]


def _render_views(output: Path, geometry: dict[str, np.ndarray], samples: dict[str, np.ndarray], config: dict[str, object]) -> None:
    render = config["render"]
    scene = bpy.context.scene
    scene.render.engine = str(render["engine"])
    scene.render.resolution_x = int(render["width"])
    scene.render.resolution_y = int(render["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_layers[0].use_pass_z = True
    scene.world.color = (0.08, 0.08, 0.08)
    light_data = bpy.data.lights.new("SceneComposeSun", type="SUN")
    light_data.energy = 2.0
    light_data.angle = math.radians(20.0)
    light = bpy.data.objects.new("SceneComposeSun", light_data)
    light.rotation_euler = (math.radians(25.0), math.radians(-20.0), math.radians(35.0))
    scene.collection.objects.link(light)
    camera_data = bpy.data.cameras.new("SceneComposeCamera")
    camera = bpy.data.objects.new("SceneComposeCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.angle = math.radians(float(render["horizontal_fov_deg"]))
    scene.use_nodes = True
    nodes = scene.node_tree
    nodes.nodes.clear()
    render_layers = nodes.nodes.new("CompositorNodeRLayers")
    depth_output = nodes.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = str((output / "views" / "depth").resolve())
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.file_slots[0].path = "depth_"
    nodes.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    vertices = geometry["vertices"]
    bounds_min, bounds_max = vertices.min(axis=0), vertices.max(axis=0)
    rgb_dir = output / "views" / "rgb"
    camera_dir = output / "views" / "cameras"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)
    for index, (location, target) in enumerate(_camera_candidates(samples["points"], bounds_min, bounds_max, render)):
        view_id = f"view_{index:05d}"
        camera.location = location
        camera.matrix_world = Matrix.Translation(location) @ _look_at(location, target)
        scene.render.filepath = str((rgb_dir / f"{view_id}.png").resolve())
        depth_output.file_slots[0].path = f"{view_id}_"
        bpy.ops.render.render(write_still=True)
        scale = scene.render.resolution_percentage / 100.0
        width, height = scene.render.resolution_x * scale, scene.render.resolution_y * scale
        fx = 0.5 * width / math.tan(camera_data.angle_x * 0.5)
        fy = 0.5 * height / math.tan(camera_data.angle_y * 0.5)
        intrinsic = [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]]
        camera_to_world = np.asarray(camera.matrix_world, dtype=np.float64).tolist()
        world_to_camera = np.linalg.inv(np.asarray(camera.matrix_world, dtype=np.float64)).tolist()
        _write_json(camera_dir / f"{view_id}.json", {"schema_version": 1, "view_id": view_id, "width": int(width), "height": int(height), "intrinsic": intrinsic, "camera_to_world": camera_to_world, "world_to_camera": world_to_camera, "rgb": f"../rgb/{view_id}.png", "depth_glob": f"../depth/{view_id}_*.exr"})
    _write_json(output / "views" / "views.json", {"schema_version": 1, "view_count": len(list(camera_dir.glob("*.json"))), "bounds_min": bounds_min.tolist(), "bounds_max": bounds_max.tolist()})


def _validate(output: Path) -> None:
    geometry_path = output / "geometry" / "geometry.npz"
    if not geometry_path.is_file():
        raise RuntimeError(f"Missing geometry artifact: {geometry_path}")
    with np.load(geometry_path, allow_pickle=False) as data:
        if len(data["triangles"]) == 0 or not np.isfinite(data["vertices"]).all():
            raise RuntimeError("Geometry artifact is empty or non-finite")
    for camera_path in (output / "views" / "cameras").glob("*.json"):
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        if not np.isfinite(np.asarray(camera["camera_to_world"])).all():
            raise RuntimeError(f"Non-finite camera transform: {camera_path}")


def main() -> None:
    args = _arguments()
    if args.validate_only:
        _validate(args.output)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _import_scene(args.scene, config["geometry"].get("meters_per_blender_unit"))
    geometry_path = args.output / "geometry" / "geometry.npz"
    if args.stage in {"geometry", "all"}:
        geometry = _extract_geometry(tuple(config["geometry"]["neutral_rgb"]))
        samples = _sample_surface(geometry, int(config["geometry"]["sample_count"]), int(config["geometry"]["seed"]))
        _write_npz(geometry_path, **geometry)
        _write_npz(args.output / "geometry" / "mesh_mapping.npz", triangle_ids=samples["triangle_ids"], barycentric=samples["barycentric"], triangle_object=geometry["triangle_object"], triangle_polygon=geometry["triangle_polygon"])
        _write_npz(args.output / "geometry" / "samples.npz", **samples)
        _write_point_ply(args.output / "geometry" / "sampled_points.ply", samples)
        _write_json(args.output / "geometry" / "geometry.json", {"schema_version": 1, "vertex_count": len(geometry["vertices"]), "triangle_count": len(geometry["triangles"]), "sample_count": len(samples["points"]), "bounds_min": geometry["vertices"].min(axis=0).tolist(), "bounds_max": geometry["vertices"].max(axis=0).tolist()})
    if args.stage in {"views", "all"}:
        if not geometry_path.is_file():
            raise RuntimeError("Views stage requires geometry/geometry.npz")
        with np.load(geometry_path, allow_pickle=False) as archive:
            geometry = {key: archive[key] for key in archive.files}
        with np.load(args.output / "geometry" / "samples.npz", allow_pickle=False) as archive:
            samples = {key: archive[key] for key in archive.files}
        _render_views(args.output, geometry, samples, config)
    _validate(args.output)


if __name__ == "__main__":
    main()
