# Scene Semantic Segmentation Design

## Purpose

Build an independent scene-understanding stage that converts each unlabeled scene file under `data/scene` into open-vocabulary 3D semantic and instance segmentation artifacts. This stage must not read or depend on adaptive-partition outputs. Its results will later provide object identities and semantic context to the scene-object composition stage.

The selected model combination is:

- Mosaic3D for global 3D open-vocabulary semantic features and point-level class scores.
- SAM3 for promptable instance masks over rendered RGB views.
- Open3DIS-style projection, association, and fusion for lifting 2D masks into coherent 3D instances.

## Scope

### Included

- Recursive discovery of scene files under `data/scene`.
- Independent processing and restart state for every scene.
- Blender-based mesh import, surface sampling, and multi-view RGB-D rendering.
- A custom Mosaic3D inference adapter for arbitrary scene point clouds.
- Batched SAM3 text-prompt inference over rendered views.
- Depth-based 2D-to-3D mask lifting and cross-view instance association.
- Fusion of Mosaic3D semantic evidence with lifted SAM3 instances.
- Mapping point predictions back to source mesh faces.
- Machine-readable outputs and colored PLY visualizations.
- Preflight validation for executables, repositories, configuration, and checkpoint paths.

### Excluded

- Reading `data/work/*/partition` or any region manifest.
- Improving or invoking adaptive scene partitioning.
- Downloading model weights or datasets.
- Modifying code inside Mosaic3D, SAM3, Open3DIS, or WarpConvNet.
- Detecting final support surfaces or choosing object placement poses. Those belong to the scene-object composition stage.
- LLM-based dynamic vocabulary generation in the first implementation.

## Command-Line Interface

The batch command is:

```bash
scenecompose segment-scenes \
  --scene-root data/scene \
  --output-root data/work \
  --config configs/segmentation.json
```

The command recursively discovers supported files, sorts them by relative POSIX path for deterministic execution, and writes one output directory per input scene. The initial importer must support FBX. The discovery and importer registry may recognize GLB, OBJ, and PLY later without changing downstream contracts.

Useful operational flags are:

```text
--scene-root PATH       Input root; defaults to data/scene
--output-root PATH      Work root; defaults to data/work
--config PATH           Segmentation configuration
--blender PATH          Blender executable
--gpus LIST             Comma-separated GPU IDs
--workers INTEGER       Concurrent scene workers
--resume                Reuse valid completed stages
--force-stage NAME      Recompute one stage and its dependants
--dry-run               Validate and print work without inference
```

No command accepts a partition manifest or partition output directory.

## Architecture

The implementation is an inline, artifact-driven pipeline. Each stage reads only committed artifacts from the preceding stages and records its own status in `manifest.json`. Model-specific code is isolated behind adapters so that the project does not import dataset loaders or evaluation entry points as its application interface.

```text
scene file
  -> geometry preparation
  -> multi-view RGB-D rendering
  -> Mosaic3D point semantics -----------+
  -> SAM3 per-view instance masks        |
  -> 2D-to-3D lifting and association ---+
  -> semantic-instance fusion
  -> point-to-face projection
  -> export and visualization
```

### 1. Scene Discovery and Identity

Every scene receives a stable `scene_id` derived from its path relative to `scene_root`, with a short hash appended to prevent collisions. The manifest stores the source relative path, file size, modification time, configuration digest, checkpoint identifiers, stage states, warnings, and output schema version.

Resume is allowed only when the source fingerprint, relevant configuration subset, model checkpoint fingerprint, and upstream artifact digest still match. Otherwise the affected stage and all downstream stages are recomputed.

### 2. Geometry Preparation

Blender runs headlessly with `--factory-startup`. It imports the source mesh, evaluates object transforms and modifiers, triangulates evaluated geometry, and exports a canonical world-space representation without saving a partial `.blend` file.

The preparation stage produces:

- World-space vertices and triangle indices.
- Per-corner UV coordinates and material assignments when present.
- Surface samples containing XYZ, RGB, normal, source triangle ID, and barycentric coordinates.
- Scene bounds, inferred up axis, scale metadata, and geometry statistics.

Sampling is area-weighted and deterministic for a configured seed. Texture color is evaluated through UV/material data where possible. Missing textures do not abort the scene; affected samples receive a configured neutral color and the manifest records a warning.

The first version assumes that scene units and up axis are supplied in configuration when Blender metadata is ambiguous. It must not silently rescale geometry based only on bounding-box size.

### 3. Multi-View Rendering

Camera generation operates on the complete scene. It uses horizontal coverage cells over the scene bounds and several azimuth/elevation patterns per cell. Cameras point toward sampled occupied-space targets rather than the global bounding-box center. Near and far planes are computed per view.

Each accepted view exports:

- RGB image.
- Metric depth image in a lossless floating-point format.
- Camera intrinsics.
- Camera-to-world and world-to-camera transforms.
- Visibility metadata and a deterministic view ID.

Empty or nearly empty views are rejected using depth coverage. Rendering stops only after configured surface-coverage and minimum-observation targets are met or the maximum view budget is exhausted. Insufficient coverage is a warning that lowers output confidence; it is not hidden.

### 4. Mosaic3D Adapter

The adapter constructs the tensor fields expected by the Mosaic3D encoder directly from sampled XYZ, RGB, and normals. It does not require ScanNet annotations or a ScanNet directory layout. Voxelization parameters, condition token, test-time augmentation, checkpoint path, and text encoder settings are configuration values.

The configured vocabulary is encoded once and compared with language-aligned point features. Outputs include:

- Per-point normalized feature vectors.
- Per-point top-k class IDs, logits, and confidence.
- Optional unknown assignments when confidence and class margin fall below thresholds.

Mosaic3D is the primary source of global semantic evidence. A failure in this required stage fails that scene instead of silently falling back to SAM3-only output.

### 5. SAM3 Adapter

SAM3 processes rendered RGB images using the same configured vocabulary, with prompts optionally containing class-specific synonyms. Prompts are short noun phrases, not free-form placement instructions. The adapter batches views and prompts according to GPU memory limits.

Each mask record contains the scene ID, view ID, canonical class ID, prompt text, SAM3 score, bounding box, area, and compressed binary mask. Masks below configured score, area, or border-quality thresholds are discarded. Raw masks remain available as intermediate artifacts so fusion can be rerun without repeating SAM3 inference.

### 6. 2D-to-3D Lifting and Association

Depth and camera matrices unproject mask pixels into world coordinates. Lifted observations are associated with the sampled point cloud using a configurable radius and visibility consistency check. Points inconsistent with rendered depth are excluded.

Per-view observations become provisional 3D masks. Provisional masks are associated across views using a weighted combination of:

- 3D intersection-over-union.
- Geometric proximity and connectedness.
- Mosaic3D feature similarity.
- Semantic class compatibility.
- Cross-view visibility consistency.

Association uses deterministic graph construction and connected components after thresholding. Mutually exclusive class evidence prevents incompatible masks from merging. Small disconnected components are removed or split according to configuration. This module follows Open3DIS principles but exposes project-native data types instead of requiring Open3DIS dataset loaders.

### 7. Semantic and Instance Fusion

For points covered by an associated instance, final class scores combine Mosaic3D logits, SAM3 mask confidence, view count, and visibility quality. Mosaic3D alone labels uncovered points. Low-confidence points remain `unknown`; they are not forced into the closest vocabulary class.

Overlapping instances are resolved by fused score and geometric consistency. Stuff classes such as wall, floor, and ceiling receive semantic labels but no object instance ID. Thing classes receive stable scene-local instance IDs. Mesh adjacency is used for boundary cleanup only when adjacent faces have compatible semantic evidence; smoothing must not bridge disconnected components.

### 8. Point-to-Face Projection

Surface samples retain source triangle IDs and barycentric coordinates, providing a direct mapping back to the original evaluated mesh. Face semantics are selected from weighted sample evidence. Face instance IDs require both a winning instance vote and a minimum vote ratio; otherwise they remain unassigned.

The exported mapping refers to the evaluated, triangulated geometry produced by this stage. The manifest retains original Blender object and polygon provenance where available so later export can relate predictions to source content.

## Vocabulary

`configs/segmentation.json` defines canonical class IDs, names, synonyms, and thing/stuff status. The default vocabulary targets indoor composition and includes structural regions, furniture, storage, fixtures, and common tabletop objects. Vocabulary changes invalidate Mosaic3D, SAM3, and fusion artifacts but do not invalidate geometry or rendering.

Adding a category must require configuration changes only. LLM-generated or object-conditioned vocabulary expansion is deferred until the composition stage establishes its query contract.

## Output Contract

Each scene writes:

```text
data/work/<scene_id>/segmentation/
|-- manifest.json
|-- geometry/
|   |-- sampled_points.ply
|   `-- mesh_mapping.npz
|-- views/
|   |-- rgb/
|   |-- depth/
|   `-- cameras/
|-- mosaic3d/
|   |-- point_features.pt
|   `-- semantic_scores.npz
|-- sam3/
|   `-- masks/
|-- fusion/
|   |-- point_labels.npz
|   |-- face_labels.npz
|   `-- instances.json
`-- visualization/
    |-- semantic.ply
    `-- instances.ply
```

`point_labels.npz` contains point coordinates, semantic IDs, semantic confidence, instance IDs, source triangle IDs, and unknown flags. `face_labels.npz` contains evaluated mesh triangle indices, semantic IDs, semantic confidence, instance IDs, and provenance fields.

Each entry in `instances.json` contains:

- Stable scene-local instance ID.
- Canonical class ID and label.
- Fused, Mosaic3D, and SAM3 confidence values.
- Point and face membership encoded compactly.
- Centroid, axis-aligned bounding box, and oriented bounding box.
- Observing view IDs and per-view scores.
- Quality flags such as partial visibility, low coverage, and disconnected geometry.

Support surfaces are not part of this schema. They will be derived later from semantic instances and geometry.

## GPU Execution

GPU allocation is explicit through `--gpus`; workers never select devices implicitly. Independent scenes are distributed across GPUs. For a single scene, SAM3 work may be sharded by view batches while Mosaic3D runs on one assigned GPU. Fusion operates in bounded chunks and may use the assigned GPU for tensor similarity operations.

Eight A100 GPUs improve throughput but do not change output ordering or IDs. Deterministic seeds and sorted work units keep artifacts reproducible within the limits of underlying CUDA kernels.

## Error Handling

Preflight checks report all missing requirements together: Blender executable, third-party repository paths, checkpoint files, CUDA availability, output permissions, configuration validity, and supported input formats.

Each stage writes to a temporary stage directory and atomically promotes it only after validation. A failed stage records the command, exception summary, and log path in the manifest. Completed upstream artifacts remain reusable. The batch command continues processing other scenes and returns a nonzero exit code when any scene fails.

No stage downloads weights, datasets, Python packages, or repositories automatically.

## Verification Strategy

The implementation will include unit-level checks for deterministic scene discovery, configuration validation, manifest invalidation, camera matrix conventions, depth unprojection, mask association, semantic fusion, and point-to-face voting. Small synthetic fixtures will exercise geometry and projection without model weights.

Adapter contract tests will use fake Mosaic3D and SAM3 outputs. A preflight/dry-run command will verify a real server installation without running inference. The user will run tests and model-backed validation; implementation work in this workspace will not download weights or execute those tests unless separately requested.

Model-backed acceptance for one scene is:

- All accepted views have valid finite intrinsics, poses, and depth.
- Surface observation coverage meets the configured target or is explicitly flagged.
- Point and face arrays agree with geometry counts and contain valid class/instance IDs.
- Every exported instance references existing points/faces and has finite bounds.
- Stuff classes have no instance IDs.
- Semantic and instance visualization files open as nonempty colored point clouds.
- Rerunning with `--resume` performs no model inference when fingerprints match.

## Documentation

Both `README.md` and `READMECHINESE.md` will gain a top-level Scene Semantic Segmentation section covering installation boundaries, checkpoint placement, configuration, batch commands, output structure, resume behavior, and server execution. English and Chinese instructions must describe the same command and artifact contract.

