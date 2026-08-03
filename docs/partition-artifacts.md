# Adaptive Partition Artifacts

## Directory layout

```text
<partition-output>/
|-- regions.json
|-- summary.json
`-- regions/
    |-- region_full/
    |   `-- manifest.json
    `-- region_<content-hash>/
        |-- manifest.json
        |-- source_faces.npz
        `-- scene_region.blend
```

When `export_format=glb`, `scene_region.blend` is replaced by `scene_region.glb`.

## `regions.json`

This is the primary contract for semantic segmentation and composition. It contains:

- source FBX path and SHA-256;
- source mesh/instance table and source transforms;
- meter conversion used during import;
- exact partition configuration;
- deterministic Region IDs;
- Core and context bounds in world meters;
- per-Region triangle, context-triangle, area, and estimated-point statistics;
- geometry and source-face artifact paths.

`world_from_region` is a row-major 4x4 matrix. Split geometry is baked into meter-valued world coordinates, so its matrix is identity. `region_full` references the original FBX, so the matrix includes the source-unit-to-meter scale.

## Core and context ownership

Triangle ownership is determined from its world-space XY centroid:

- a triangle belongs to exactly one Region Core;
- a triangle may appear in several context halos;
- Z is never used as a partition axis;
- triangles crossing a boundary are kept whole rather than geometrically cut.

The last rule preserves topology, UVs, and material assignments. It also means Region geometry can extend slightly beyond `context_bounds` when a large triangle crosses the boundary.

## `source_faces.npz`

Arrays have one entry per exported triangle:

| Array | Type | Meaning |
|---|---|---|
| `output_object_index` | `int32` | Index into `manifest.json.output_object_names` |
| `source_index` | `int32` | Index into `regions.json.sources` |
| `source_polygon_index` | `int64` | Polygon index in the evaluated source mesh |
| `is_core` | `bool` | Whether the triangle centroid belongs to the Core |

Downstream segmentation may use all context triangles as input, but it must publish owned predictions only for entries where `is_core=true`. Predictions near borders are fused later by source identity and world-space geometry.

## Failure and replacement behavior

Artifacts are built in a sibling staging directory. A completed directory is renamed into place only after every Region and both JSON manifests are written. On failure, the staging directory is retained with `FAILED.txt` for diagnosis.

Existing output is never replaced unless `--force` is passed. Forced replacement first moves the previous output to a uniquely named backup; the backup is restored if processing fails.

