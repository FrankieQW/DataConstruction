# Scene/Object Automatic Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic pipeline that classifies and profiles Objaverse GLBs, extracts semantic support surfaces from each observation, places one compatible object with geometric constraints, and exports a combined GLB plus placement and camera metadata without rendering images.

**Architecture:** Ordinary Python owns strict configuration, metadata/LLM classification, manifests, discovery, resume logic, and batch scheduling. Headless Blender owns evaluated GLB profiling, connected-component/upright analysis, BVH placement validation, camera visibility, and combined GLB export. Numpy artifacts and JSON contracts isolate the stages and make failure per Object or observation recoverable.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Transformers/PyTorch for optional local Qwen3 inference, Blender 4.5 LTS Python API, existing SceneCompose observation/segmentation artifacts.

**Execution constraints:** Inline Execution. Do not perform Git operations, download models, run Blender, load an LLM, render images, or execute automated tests. Use only static Python/JSON/shell inspection; server validation is performed by the user.

---

### Task 1: Composition configuration and contracts

**Files:**
- Create: `configs/composition.json`
- Create: `src/scenecompose/composition/__init__.py`
- Create: `src/scenecompose/composition/config.py`
- Create: `src/scenecompose/composition/contracts.py`

- [ ] Define strict frozen config sections `CatalogConfig`, `LlmConfig`, `NormalizationConfig`, `SupportConfig`, `PlacementConfig`, and `CompositionConfig`.
- [ ] Validate enumerations, positive/range values, unique class rules, local-only LLM behavior, and `objects_per_observation == 1`.
- [ ] Define JSON contracts for `ObjectProfile`, `CatalogEntry`, `SupportSurface`, `PlacementResult`, and stage manifests.
- [ ] Implement deterministic SHA-256 config/input digests and atomic JSON writers.

### Task 2: Object discovery and metadata rules

**Files:**
- Create: `src/scenecompose/composition/discovery.py`
- Create: `src/scenecompose/composition/rules.py`

- [ ] Recursively discover GLBs while excluding metadata/cache/work directories and reject duplicate UIDs.
- [ ] Load `annotations.json`, validate `_scenecompose.uid/glb_relative_path`, and join each discovered UID to its annotation.
- [ ] Normalize LVIS, name, description, tag, and category text without flattening unrelated metadata.
- [ ] Implement ordered exact-token/category rules that reject scenes, architecture, people, characters, animals, vehicles, weapons, and abstract assets.
- [ ] Implement high-confidence canonical mappings for surface/floor objects and their compatible support classes.
- [ ] Return `accepted`, `rejected`, or `needs_review` with evidence, confidence, target-size rule, and reason.

### Task 3: Batched Blender Object profiling

**Files:**
- Create: `scripts/blender/profile_objaverse_batch.py`
- Create: `src/scenecompose/composition/profiling.py`

- [ ] Write batch manifests containing UID, absolute GLB path, output profile path, and config digest.
- [ ] Import one GLB at a time in a factory Blender scene and always clean its objects/materials/images before the next item.
- [ ] Extract evaluated vertices/triangles, mesh and connected-component counts, dominant-component ratio, bounds, animation/skin signals, materials, and texture slots.
- [ ] Score six signed-axis upright candidates using bottom-band contact, footprint area, center-of-mass projection, and height/footprint priors.
- [ ] Record deterministic rejection reasons for empty, degenerate, fragmented, too-small, and too-large geometry.
- [ ] Atomically write one `object_profiles/<uid>.json`; isolate per-object failures and return a batch summary.
- [ ] Schedule bounded external Blender jobs with resume and `--force-profiles` behavior.

### Task 4: Local Transformers/Qwen3 classifier

**Files:**
- Create: `src/scenecompose/composition/llm.py`

- [ ] Load tokenizer/model from `llm.model_path` with `local_files_only=True`, configured dtype/device, and no remote service.
- [ ] Fail before model loading if the path is absent, `trust_remote_code` violates config, or Transformers/Torch is unavailable.
- [ ] Build bounded prompts from normalized metadata and geometry profile fields.
- [ ] Use the model chat template when available, batch tokenization/generation, deterministic decoding, and optional Qwen thinking suppression through supported template arguments.
- [ ] Extract exactly one JSON object and validate decision/class/placement/support/size/confidence/reason.
- [ ] Retry invalid generations up to `max_retries`, then return `needs_review` without failing other records.
- [ ] Store prompt hash, model identity, raw output, parsed result, and error in append-safe `llm_cache.jsonl`; reuse matching cache rows.

### Task 5: Object Catalog orchestration

**Files:**
- Create: `src/scenecompose/composition/catalog.py`
- Modify: `src/scenecompose/cli.py`
- Create: `scripts/run_object_catalog.sh`

- [ ] Add `build-object-catalog` arguments for object root, metadata, output, config, Blender, workers, resume, force profiles, force LLM, and `--no-llm`.
- [ ] Run discovery/rules first, profile non-rejected GLBs, and prevent geometry-rejected objects from reaching the LLM.
- [ ] Invoke local LLM once for unresolved records only; never load it inside Blender workers.
- [ ] Merge rules, profile, and LLM evidence into deterministic catalog entries.
- [ ] Write `object_catalog.json`, status lists, and summary counts atomically.
- [ ] Return success when individual records fail but the catalog contains accepted objects; fail when no accepted surface/floor object exists.

### Task 6: Observation discovery and support surface extraction

**Files:**
- Create: `src/scenecompose/composition/observation.py`
- Create: `src/scenecompose/composition/support.py`

- [ ] Discover observations with complete partition and segmentation artifacts and stable-sort by source scene/observation ID.
- [ ] Validate geometry/face-label lengths, finite vertices, semantic IDs, visibility, and Core masks.
- [ ] Select non-degenerate upward faces from configured support classes with Core/visibility thresholds.
- [ ] Cluster by semantic/instance, quantized height, shared edges, and spatial tolerance without crossing semantic instances.
- [ ] Rasterize each patch into an XY occupancy grid from the actual triangles, preserve gaps, and erode by configured boundary margin.
- [ ] Reject patches below area/occupancy thresholds and assign deterministic support IDs.
- [ ] Atomically write `support_surfaces.json` and `support_occupancy.npz` with segmentation/config digests.

### Task 7: Blender placement solver and final camera

**Files:**
- Create: `scripts/blender/compose_observation.py`

- [ ] Load observation, partition GLB, support artifacts, catalog entry/profile, and composition config.
- [ ] Import the Object, retain dominant connected geometry, apply catalog upright rotation, uniform target-size scale, and bottom-center origin normalization.
- [ ] Sample support cells/triangles and yaw candidates from `seed + observation_id` deterministically.
- [ ] Reject candidates failing support occupancy coverage, boundary margin, center-of-mass support, partition Context, or finite-transform checks.
- [ ] Build scene/object BVHs, allow only the bottom contact tolerance, and reject penetrating or insufficient-clearance candidates.
- [ ] Sample final cameras within the observation camera domain, reject camera collisions, and ray-test Object surface visibility.
- [ ] Score valid placements and cameras, select one stable maximum, and record rejected-reason counts.
- [ ] Export selected partition and transformed Object as `combined_scene.glb` with materials; do not invoke render operators.
- [ ] Write staging `placement.json`, `camera.json`, `candidate_summary.json`, and manifest, then atomically publish `composition/`.

### Task 8: Composition batch orchestration

**Files:**
- Create: `src/scenecompose/composition/pipeline.py`
- Modify: `src/scenecompose/cli.py`
- Create: `scripts/run_composition.sh`

- [ ] Add `compose-observations` arguments for observation root, object root, catalog, config, Blender, workers, resume, and force.
- [ ] Validate accepted catalog paths and build support artifacts before dispatching Blender placement.
- [ ] Select compatible Object trial order per observation from support classes, category-balancing weights, confidence, footprint fit, and stable RNG.
- [ ] Pass a bounded ordered Object candidate list to each Blender worker so failed objects can be replaced without scheduler round trips.
- [ ] Run one external Blender process per observation with ThreadPool scheduling and independent logs.
- [ ] Write a deterministic batch summary with complete/failed/skipped counts and reasons.

### Task 9: Artifact and deployment documentation

**Files:**
- Create: `docs/object-catalog-artifacts.md`
- Create: `docs/composition-artifacts.md`
- Modify: `README.md`
- Modify: `READMECHINESE.md`

- [ ] Document catalog/profile/LLM cache schemas, status meanings, and no-LLM rule mode.
- [ ] Document support, placement, camera, manifest, GLB, resume, and failure artifacts.
- [ ] Add complete Mamba and Pixi installation commands for the optional local LLM dependencies.
- [ ] Using official Qwen and Transformers documentation, describe placing a Qwen3 checkpoint locally, configuring Transformers direct inference, local-files-only behavior, dtype/device/batch size, and memory troubleshooting.
- [ ] State explicitly that no API, HTTP inference service, metadata upload, automatic model download, or image rendering occurs.
- [ ] Document recommended resource allocation: one GPU for catalog LLM inference, CPU/RAM-bound Blender profiling/composition workers, and eight GPUs for segmentation.
- [ ] Add end-to-end commands from metadata through catalog and composition.

### Task 10: Static review and user verification handoff

**Files:**
- Inspect all files above and existing observation/segmentation contracts.

- [ ] Parse every new/modified Python file with `ast` without importing Blender, Transformers, or Torch.
- [ ] Parse `configs/composition.json` with a JSON parser.
- [ ] Scan for remote model/API calls, automatic checkpoint downloads, render operators in composition code, home-directory writes, placeholders, and inconsistent artifact names.
- [ ] Cross-check CLI names/options against both README files and wrappers.
- [ ] Provide exact server commands and an explicit list of Blender/LLM/geometry checks left to the user; do not execute them locally.
