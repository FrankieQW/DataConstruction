# Scene Semantic Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `data/scene` to open-vocabulary 3D semantic/instance segmentation pipeline using Mosaic3D, SAM3, and Open3DIS-style fusion.

**Architecture:** SceneCompose owns discovery, manifests, Blender preprocessing, model adapters, fusion, and exports. Third-party repositories remain unmodified and are loaded only inside their adapters. Every stage commits restartable artifacts under `data/work/<scene_id>/segmentation`.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Blender 4.5 Python API, Pillow, OpenEXR/imageio, SciPy, scikit-learn, trimesh, plyfile, Mosaic3D, SAM3, Open3DIS.

**Execution constraint:** Add test files and document exact verification commands, but do not download weights or run tests/model inference; the user performs those operations.

---

## File Structure

Create these focused modules:

```text
src/scenecompose/segmentation/
|-- __init__.py          Public segmentation API
|-- config.py            Strict JSON configuration and vocabulary types
|-- discovery.py         Deterministic scene discovery and stable IDs
|-- manifest.py          Stage fingerprints, state, and atomic updates
|-- artifacts.py         Typed NPZ/JSON artifact readers and writers
|-- preflight.py         External repository/checkpoint/runtime validation
|-- blender.py           Blender subprocess command construction
|-- mosaic.py            Mosaic3D arbitrary-point-cloud adapter
|-- sam3_adapter.py      Batched promptable image segmentation adapter
|-- lifting.py           Depth unprojection and 2D mask lifting
|-- association.py       Cross-view graph association
|-- fusion.py            Semantic/instance score fusion and face voting
|-- export.py            PLY and instance JSON export
`-- pipeline.py          Restartable per-scene and batch orchestration
scripts/blender/prepare_segmentation_scene.py
configs/segmentation.json
scripts/run_segmentation.sh
tests/segmentation/
```

Modify `src/scenecompose/cli.py`, `pyproject.toml`, `README.md`, and `READMECHINESE.md`. Do not modify third-party repositories.

### Task 1: Configuration and Vocabulary Contract

**Files:**
- Create: `src/scenecompose/segmentation/__init__.py`
- Create: `src/scenecompose/segmentation/config.py`
- Create: `configs/segmentation.json`
- Create: `tests/segmentation/test_config.py`

- [ ] **Step 1: Add strict configuration tests**

Cover valid loading, unknown keys, duplicate class IDs/names, missing thing/stuff type, invalid thresholds, invalid paths, and vocabulary digest stability. Tests construct temporary JSON and assert `SegmentationConfig.from_json()` either returns an immutable config or raises `ValueError` naming the offending field.

- [ ] **Step 2: Implement immutable configuration types**

Define `VocabularyClass`, `GeometryConfig`, `RenderConfig`, `MosaicConfig`, `Sam3Config`, `FusionConfig`, `RuntimeConfig`, and `SegmentationConfig`. Use a shared strict dataclass loader that rejects unknown keys recursively. `SegmentationConfig.validate()` enforces unique canonical IDs/names, nonempty prompts, probabilities in `[0, 1]`, positive sampling/render dimensions, and existing-value syntax without requiring files to exist.

The public methods are:

```python
@classmethod
def from_json(cls, path: Path) -> SegmentationConfig: ...

def vocabulary_digest(self) -> str: ...
def stage_digest(self, stage: str) -> str: ...
def to_dict(self) -> dict[str, object]: ...
```

- [ ] **Step 3: Add the default indoor composition vocabulary**

`configs/segmentation.json` includes structural stuff (`wall`, `floor`, `ceiling`) and placement-relevant things (`table`, `desk`, `counter`, `shelf`, `cabinet`, `chair`, `sofa`, `bed`, `sink`, `toilet`, `door`, `window`, `lamp`, `plant`, `cup`, `bottle`, `book`, `monitor`). Each entry defines `id`, `name`, `synonyms`, and `kind`.

- [ ] **Step 4: Document user verification**

Command: `pytest tests/segmentation/test_config.py -v`

Expected: all configuration contract tests pass without importing model repositories.

- [ ] **Step 5: Commit**

```bash
git add src/scenecompose/segmentation configs/segmentation.json tests/segmentation/test_config.py
git commit -m "feat: define scene segmentation configuration"
```

### Task 2: Scene Discovery, IDs, and Restart Manifest

**Files:**
- Create: `src/scenecompose/segmentation/discovery.py`
- Create: `src/scenecompose/segmentation/manifest.py`
- Create: `tests/segmentation/test_discovery.py`
- Create: `tests/segmentation/test_manifest.py`

- [ ] **Step 1: Add discovery and manifest contract tests**

Verify recursive FBX discovery, case-insensitive suffixes, deterministic relative-path sorting, collision-resistant IDs, empty-root behavior, atomic manifest updates, failed-stage recording, and downstream invalidation after source/config/checkpoint changes.

- [ ] **Step 2: Implement stable scene discovery**

Expose:

```python
@dataclass(frozen=True)
class SceneInput:
    scene_id: str
    source: Path
    relative_source: str

def discover_scenes(scene_root: Path, suffixes: tuple[str, ...] = (".fbx",)) -> list[SceneInput]: ...
```

Generate `scene_id` from a sanitized relative stem plus the first eight hex characters of SHA-256 over the relative POSIX path.

- [ ] **Step 3: Implement manifest state and fingerprinting**

Use stages `geometry`, `views`, `mosaic3d`, `sam3`, `fusion`, and `export`. Store `pending/running/complete/failed`, timestamps, artifact digests, relevant config digest, source fingerprint, checkpoint fingerprint, warnings, and log path. Write JSON through a sibling temporary file followed by `Path.replace()`.

- [ ] **Step 4: Document user verification**

Command: `pytest tests/segmentation/test_discovery.py tests/segmentation/test_manifest.py -v`

Expected: deterministic IDs and invalidation tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/scenecompose/segmentation/discovery.py src/scenecompose/segmentation/manifest.py tests/segmentation
git commit -m "feat: add segmentation discovery and manifests"
```

### Task 3: Typed Artifacts and Preflight

**Files:**
- Create: `src/scenecompose/segmentation/artifacts.py`
- Create: `src/scenecompose/segmentation/preflight.py`
- Create: `tests/segmentation/test_artifacts.py`
- Create: `tests/segmentation/test_preflight.py`

- [ ] **Step 1: Add artifact round-trip and preflight tests**

Round-trip geometry, cameras, semantic scores, masks, labels, and instances through NPZ/JSON. Assert dtype, shape, finite-value, and index-bound validation. Preflight tests use temporary fake repository/checkpoint trees and assert all missing requirements appear in one exception.

- [ ] **Step 2: Implement artifact schemas**

Define dataclasses `GeometryArtifact`, `CameraArtifact`, `SemanticArtifact`, `MaskObservation`, `PointLabels`, `FaceLabels`, and `InstanceRecord`. Save large arrays with `numpy.savez_compressed`; JSON contains paths and metadata rather than expanded point/face index lists. Mask pixels use COCO-compatible uncompressed RLE implemented locally to avoid a pycocotools dependency in orchestration code.

- [ ] **Step 3: Implement preflight aggregation**

Validate Blender discovery, Linux platform unless `--dry-run`, third-party roots, configured checkpoint paths, writable output parent, GPU list syntax, and import availability inside the inference environment. Return a structured report; raise one `PreflightError` containing every hard failure.

- [ ] **Step 4: Document user verification**

Command: `pytest tests/segmentation/test_artifacts.py tests/segmentation/test_preflight.py -v`

Expected: schema round trips and aggregated diagnostics pass.

- [ ] **Step 5: Commit**

```bash
git add src/scenecompose/segmentation/artifacts.py src/scenecompose/segmentation/preflight.py tests/segmentation
git commit -m "feat: add segmentation artifacts and preflight"
```

### Task 4: Blender Geometry Preparation and RGB-D Rendering

**Files:**
- Create: `scripts/blender/prepare_segmentation_scene.py`
- Create: `src/scenecompose/segmentation/blender.py`
- Create: `tests/segmentation/test_blender_command.py`

- [ ] **Step 1: Add subprocess command tests**

Assert the command uses `--background --factory-startup`, passes only absolute paths after `--`, never writes a `.blend`, and includes the configured stage (`geometry`, `views`, or `all`).

- [ ] **Step 2: Implement the Blender command adapter**

Expose `build_prepare_command()` and `run_prepare_scene()`. Capture stdout/stderr into the scene log directory and propagate Blender's exit code without shell interpolation.

- [ ] **Step 3: Implement evaluated geometry extraction**

The Blender script imports FBX, removes non-mesh objects after import, evaluates dependency-graph meshes and instances, applies world transforms, triangulates loop triangles, and records object/polygon provenance. It area-samples deterministic surface points and evaluates material base color/UV texture pixels where supported. It writes `geometry.npz`, `sampled_points.ply`, `mesh_mapping.npz`, and `geometry.json` atomically.

- [ ] **Step 4: Implement occupancy-aware cameras and rendering**

Generate camera targets from occupied XY cells and surface-sample height quantiles. Render configured azimuth/elevation candidates using Cycles/Eevee headless mode, export PNG RGB and OpenEXR metric depth, and write a camera JSON per accepted view. Reject low-depth-coverage images and stop after surface observation targets or the maximum view budget.

- [ ] **Step 5: Add Blender-side validation mode**

`--validate-only` loads generated arrays and reports nonfinite transforms, invalid triangle IDs, nonpositive depth, missing files, and achieved observation coverage without rendering again.

- [ ] **Step 6: Document user verification**

Commands:

```bash
pytest tests/segmentation/test_blender_command.py -v
blender --background --factory-startup --python scripts/blender/prepare_segmentation_scene.py -- \
  --scene data/scene/<scene>.fbx --output data/work/<scene>/segmentation --config configs/segmentation.json --stage all
```

Expected: command tests pass; Blender writes nonempty geometry and views with no partial `.blend` write.

- [ ] **Step 7: Commit**

```bash
git add scripts/blender/prepare_segmentation_scene.py src/scenecompose/segmentation/blender.py tests/segmentation/test_blender_command.py
git commit -m "feat: prepare segmentation geometry and views"
```

### Task 5: Mosaic3D Arbitrary-Scene Adapter

**Files:**
- Create: `src/scenecompose/segmentation/mosaic.py`
- Create: `tests/segmentation/test_mosaic_adapter.py`

- [ ] **Step 1: Add fake-model adapter tests**

Use a fake encoder and text classifier to verify voxel batching, inverse-map restoration, feature normalization, top-k ordering, unknown thresholds, checkpoint/config forwarding, and deterministic chunk concatenation without importing Mosaic3D.

- [ ] **Step 2: Implement lazy repository loading**

Temporarily prepend the configured Mosaic3D root only inside adapter construction, import its Hydra model factory, load the Lightning checkpoint explicitly, remove training/evaluation hooks, switch to evaluation mode, and restore `sys.path`. Emit actionable errors for incompatible config targets or checkpoint keys.

- [ ] **Step 3: Implement arbitrary point-cloud inference**

Convert geometry artifacts into Mosaic3D fields (`coord`, `grid_coord`, `feat`, `offset`, and configured condition). Process bounded spatial chunks with overlap, restore logits to original samples via voxel inverse maps, and merge overlap using normalized distance weights.

- [ ] **Step 4: Implement open-vocabulary classification**

Encode canonical labels and synonyms through the checkpoint-compatible text encoder, average normalized synonym embeddings, calculate cosine logits, retain top-k, and assign `unknown` when confidence or top-two margin misses configured thresholds. Save `point_features.pt` and `semantic_scores.npz`.

- [ ] **Step 5: Document user verification**

Commands:

```bash
pytest tests/segmentation/test_mosaic_adapter.py -v
CUDA_VISIBLE_DEVICES=0 scenecompose segment-scenes --scene-root data/scene --output-root data/work \
  --config configs/segmentation.json --force-stage mosaic3d
```

Expected: fake adapter tests pass; server run writes scores whose point count equals `sampled_points.ply`.

- [ ] **Step 6: Commit**

```bash
git add src/scenecompose/segmentation/mosaic.py tests/segmentation/test_mosaic_adapter.py
git commit -m "feat: adapt Mosaic3D scene inference"
```

### Task 6: SAM3 Multi-View Adapter

**Files:**
- Create: `src/scenecompose/segmentation/sam3_adapter.py`
- Create: `tests/segmentation/test_sam3_adapter.py`

- [ ] **Step 1: Add fake-processor tests**

Verify view batching, one image embedding per view batch, canonical prompt mapping, synonym handling, score/area/border filtering, stable mask IDs, RLE round trips, and deterministic ordering.

- [ ] **Step 2: Implement lazy SAM3 construction**

Load `build_sam3_image_model` and `Sam3Processor` from the configured repository, pass the configured checkpoint/tokenizer paths, assign the requested CUDA device, enable inference mode, and avoid Hugging Face downloads by requiring local paths.

- [ ] **Step 3: Implement batched prompt inference**

Group RGB paths by configured image batch size, call `set_image_batch`, apply canonical names and synonyms with `set_text_prompt`, normalize the returned masks/scores/boxes, filter them, and write per-view compressed mask artifacts plus an index JSON.

- [ ] **Step 4: Document user verification**

Commands:

```bash
pytest tests/segmentation/test_sam3_adapter.py -v
CUDA_VISIBLE_DEVICES=0 scenecompose segment-scenes --scene-root data/scene --output-root data/work \
  --config configs/segmentation.json --force-stage sam3
```

Expected: fake adapter tests pass; all mask index records reference an existing RGB view and canonical class.

- [ ] **Step 5: Commit**

```bash
git add src/scenecompose/segmentation/sam3_adapter.py tests/segmentation/test_sam3_adapter.py
git commit -m "feat: add SAM3 multi-view segmentation"
```

### Task 7: Depth Lifting and Cross-View Association

**Files:**
- Create: `src/scenecompose/segmentation/lifting.py`
- Create: `src/scenecompose/segmentation/association.py`
- Create: `tests/segmentation/test_lifting.py`
- Create: `tests/segmentation/test_association.py`

- [ ] **Step 1: Add analytic projection tests**

Use a small pinhole camera and planar point set with known projections. Verify camera convention, pixel-center handling, metric depth tolerance, nearest-point radius, invalid-depth exclusion, and visibility filtering.

- [ ] **Step 2: Implement vectorized lifting**

Unproject selected mask pixels through inverse intrinsics and camera-to-world transforms. Query sampled points with `scipy.spatial.cKDTree`, require depth/radius consistency, and return sorted unique point indices plus per-point observation weights.

- [ ] **Step 3: Add association graph tests**

Cover same-instance multi-view merges, incompatible-class rejection, disconnected-component splitting, deterministic connected components, minimum-observation filtering, and no quadratic dense matrix allocation.

- [ ] **Step 4: Implement sparse candidate association**

Build candidate edges only for observations sharing spatial grid cells. Score edges from 3D IoU, centroid distance, normalized Mosaic feature cosine similarity, class compatibility, and visibility. Threshold edges and compute deterministic connected components; post-split disconnected point components.

- [ ] **Step 5: Document user verification**

Command: `pytest tests/segmentation/test_lifting.py tests/segmentation/test_association.py -v`

Expected: analytic geometry and graph cases pass without model weights.

- [ ] **Step 6: Commit**

```bash
git add src/scenecompose/segmentation/lifting.py src/scenecompose/segmentation/association.py tests/segmentation
git commit -m "feat: lift and associate multi-view masks"
```

### Task 8: Semantic/Instance Fusion and Mesh Projection

**Files:**
- Create: `src/scenecompose/segmentation/fusion.py`
- Create: `tests/segmentation/test_fusion.py`

- [ ] **Step 1: Add score fusion and voting tests**

Cover Mosaic-only points, fused instance points, unknown preservation, overlap resolution, stuff instance suppression, stable instance IDs, face vote thresholds, disconnected boundary cleanup, and array-bound validation.

- [ ] **Step 2: Implement point score fusion**

Combine normalized Mosaic logits with observation-weighted SAM3 confidence, view count saturation, and visibility quality using config weights that must sum to one. Resolve overlaps by fused score; preserve unknown below confidence/margin thresholds.

- [ ] **Step 3: Implement thing/stuff handling and stable IDs**

Assign instance IDs only to vocabulary entries marked `thing`. Sort accepted instances by canonical class ID, centroid, and minimum point index before numbering so GPU scheduling cannot change IDs.

- [ ] **Step 4: Implement face projection and conservative cleanup**

Aggregate sample weights by source triangle. Require minimum semantic and instance vote ratios. Smooth only across mesh-adjacent faces that share compatible scores and geometry; never connect separate components by bounding-box proximity.

- [ ] **Step 5: Document user verification**

Command: `pytest tests/segmentation/test_fusion.py -v`

Expected: all semantic/instance invariants pass.

- [ ] **Step 6: Commit**

```bash
git add src/scenecompose/segmentation/fusion.py tests/segmentation/test_fusion.py
git commit -m "feat: fuse scene semantics and instances"
```

### Task 9: Exporters and Visualization Artifacts

**Files:**
- Create: `src/scenecompose/segmentation/export.py`
- Create: `tests/segmentation/test_export.py`

- [ ] **Step 1: Add export contract tests**

Verify semantic and instance PLY headers/counts, deterministic palette output, finite AABB/OBB values, compact index-run encoding, valid view references, and atomic file replacement.

- [ ] **Step 2: Implement PLY exports**

Write binary little-endian colored point PLY files without importing GUI libraries. Semantic colors derive from a stable class palette; instance colors derive from a stable hash and reserve gray for unknown.

- [ ] **Step 3: Implement instance JSON**

Compute centroid, AABB, PCA-based OBB with degeneracy fallback, confidence components, observing views, quality flags, and compact contiguous index runs. Validate the complete payload before replacing `instances.json`.

- [ ] **Step 4: Document user verification**

Command: `pytest tests/segmentation/test_export.py -v`

Expected: exports are deterministic, nonempty, and schema-valid.

- [ ] **Step 5: Commit**

```bash
git add src/scenecompose/segmentation/export.py tests/segmentation/test_export.py
git commit -m "feat: export scene segmentation artifacts"
```

### Task 10: Restartable Pipeline, CLI, and GPU Scheduling

**Files:**
- Create: `src/scenecompose/segmentation/pipeline.py`
- Modify: `src/scenecompose/cli.py`
- Create: `scripts/run_segmentation.sh`
- Create: `tests/segmentation/test_pipeline.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add orchestration tests with fake stages**

Verify stage order, resume skips, `--force-stage` downstream invalidation, per-scene failure isolation, deterministic summary order, explicit round-robin GPU assignment, no partition path access, and nonzero batch exit when any scene fails.

- [ ] **Step 2: Implement `SegmentationPipeline`**

Inject stage callables for testability. For each stage, mark running, write into a temporary stage directory, validate artifacts, atomically promote, record completion, and preserve upstream outputs on failure. Logs live under `<segmentation>/logs`.

- [ ] **Step 3: Implement batch scheduling**

Use `ProcessPoolExecutor` with one process per configured worker. Assign each scene one explicit GPU ID through the worker environment before importing PyTorch. Do not share loaded CUDA models between processes. Write `data/work/segmentation_summary.json` atomically.

- [ ] **Step 4: Add CLI commands**

Add `segment-scenes` with the exact flags from the design. `--dry-run` performs discovery, config loading, preflight, planned GPU assignment, and output-path reporting without creating model artifacts. Preserve existing partition commands unchanged.

- [ ] **Step 5: Update dependencies without model packages**

Add orchestration dependencies (`numpy`, `scipy`, `Pillow`, `imageio`, `trimesh`, `plyfile`) to `pyproject.toml`. Do not add Mosaic3D/SAM3/Open3DIS as pip dependencies because their environments and CUDA builds remain repository-managed.

- [ ] **Step 6: Document user verification**

Commands:

```bash
pytest tests/segmentation/test_pipeline.py -v
scenecompose segment-scenes --scene-root data/scene --output-root data/work \
  --config configs/segmentation.json --gpus 0,1,2,3,4,5,6,7 --workers 8 --dry-run
```

Expected: fake-stage tests pass; dry-run lists only files below `data/scene` and performs no inference.

- [ ] **Step 7: Commit**

```bash
git add src/scenecompose/segmentation/pipeline.py src/scenecompose/cli.py scripts/run_segmentation.sh tests/segmentation/test_pipeline.py pyproject.toml
git commit -m "feat: orchestrate scene segmentation pipeline"
```

### Task 11: English and Chinese Documentation

**Files:**
- Modify: `README.md`
- Modify: `READMECHINESE.md`

- [ ] **Step 1: Correct the top-level pipeline description**

State that semantic segmentation consumes complete files under `data/scene`, not partition Regions. Keep adaptive partitioning documented as an optional earlier experiment rather than a required input.

- [ ] **Step 2: Add Scene Semantic Segmentation sections**

Document Mamba and Pixi dependency boundaries, local third-party repository paths, checkpoint path configuration without download commands, FBX inputs, `segment-scenes`, eight-GPU example, dry-run, resume/force behavior, output tree, failure logs, and the semantic/instance schemas.

- [ ] **Step 3: Keep both languages contract-equivalent**

Use UTF-8 Chinese text and verify commands, config keys, filenames, and defaults exactly match the English README.

- [ ] **Step 4: Document user verification**

Commands:

```bash
rg -n "segment-scenes|configs/segmentation.json|data/scene|data/work" README.md READMECHINESE.md
scenecompose segment-scenes --help
```

Expected: both READMEs contain the same runnable command and output contract.

- [ ] **Step 5: Commit**

```bash
git add README.md READMECHINESE.md
git commit -m "docs: add scene segmentation guide"
```

### Task 12: User-Run Acceptance Checklist

**Files:**
- Create: `docs/segmentation-artifacts.md`
- Modify: `README.md`
- Modify: `READMECHINESE.md`

- [ ] **Step 1: Document the artifact schema and invariants**

List every NPZ key/dtype/shape, JSON field, coordinate convention, camera matrix direction, depth units, class/instance sentinel values, stage invalidation rules, and schema version.

- [ ] **Step 2: Add server acceptance commands**

Provide commands for preflight, one-scene execution on one GPU, resume verification, and multi-scene eight-GPU execution. Do not include automatic downloads.

- [ ] **Step 3: Add manual visual checks**

Require opening `semantic.ply` and `instances.ply`, checking table/floor/wall boundaries, verifying distinct furniture instances, reviewing unknown regions, and examining low-coverage warnings before composition work begins.

- [ ] **Step 4: Final static review without executing tests**

Inspect `git diff --check`, search for partition dependencies in the segmentation package, ensure no hard-coded Windows paths, and list all commands the user should run. Do not claim model-backed success until the user reports those results.

- [ ] **Step 5: Commit**

```bash
git add docs/segmentation-artifacts.md README.md READMECHINESE.md
git commit -m "docs: define segmentation acceptance contract"
```

