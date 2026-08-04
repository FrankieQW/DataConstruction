# Observation Partition Artifacts

The default first stage samples observer-centered local scenes without using semantic labels.

```text
data/work/<scene-id>/observations/
|-- observations.json
|-- summary.json
`-- observation_<hash>/
    |-- observation.json
    |-- partition/
    |   |-- scene_partition.glb
    |   `-- source_faces.npz
    `-- segmentation/
```

`observation.json` stores the floor/eye anchor, orthonormal reference frame, sector radius and horizontal angle, vertical bounds, and camera-motion/context margins. All coordinates remain in the original scene world frame.

`scene_partition.glb` contains complete selected triangles. A triangle is retained when its centroid, a vertex, or an edge midpoint intersects the expanded Context sector. Geometry is not removed based on camera occlusion.

All selected sources are merged into one `partition_merged` mesh before GLB export. Only materials used by selected polygons are attached. Valid materials are reused; a private copy is created only when invalid zero-size image nodes must be removed. This reduces glTF work without changing face provenance.

`source_faces.npz` stores one row per exported triangle:

- `output_triangle_index`
- `output_object_name`
- `output_triangle_index_within_object`
- `source_object_index`
- `source_instance_index`
- `source_polygon_index`
- `is_core`
- `is_context`

The object name and object-local triangle index are used after GLB re-import, so provenance does not depend on object enumeration order.

Generation can yield fewer observations than requested when no additional anchor satisfies floor continuity, head/body clearance, minimum spacing, and minimum core geometry. This is recorded as a warning in `observations.json`; invalid anchors are not duplicated.

After segmentation, composition should normally accept support faces only when `is_core` is true and `visibility_count` meets its configured minimum. A final-camera BVH ray check is still required after selecting a placement candidate.
