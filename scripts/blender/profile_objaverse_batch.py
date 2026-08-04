from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def _clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _profile(job):
    _clear()
    result = bpy.ops.import_scene.gltf(filepath=job["glb"])
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import failed: {result}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB contains no mesh")
    points = []
    faces = 0
    for obj in meshes:
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        faces += len(mesh.loop_triangles)
        points.extend(obj.matrix_world @ vertex.co for vertex in mesh.vertices)
        evaluated.to_mesh_clear()
    lower = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    upper = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    extent = upper - lower
    orientations = []
    for axis, dims in (("+Z", (extent.x, extent.y, extent.z)), ("-Z", (extent.x, extent.y, extent.z)),
                       ("+X", (extent.y, extent.z, extent.x)), ("-X", (extent.y, extent.z, extent.x)),
                       ("+Y", (extent.x, extent.z, extent.y)), ("-Y", (extent.x, extent.z, extent.y))):
        footprint = max(float(dims[0] * dims[1]), 1e-12)
        orientations.append({"up_axis": axis, "extent": list(map(float, dims)),
                             "heuristic_score": float(dims[2] / footprint ** 0.5)})
    _atomic(Path(job["output"]), {
        "schema_version": 1, "uid": job["uid"], "status": "complete",
        "mesh_count": len(meshes), "triangle_count": faces,
        "bounds_min": list(map(float, lower)), "bounds_max": list(map(float, upper)),
        "extent": list(map(float, extent)), "orientations": orientations,
    })


def main():
    payload = json.loads(_args().job.read_text(encoding="utf-8"))
    failures = 0
    for job in payload["jobs"]:
        try:
            _profile(job)
        except Exception as error:
            failures += 1
            _atomic(Path(job["output"]), {"schema_version": 1, "uid": job["uid"],
                    "status": "failed", "error": str(error), "traceback": traceback.format_exc()})
    print(json.dumps({"profile_jobs": len(payload["jobs"]), "failed": failures}))


if __name__ == "__main__":
    main()
