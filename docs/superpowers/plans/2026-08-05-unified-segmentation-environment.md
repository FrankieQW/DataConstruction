# Unified Segmentation Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Mosaic3D and SAM3 sequentially in one Python environment without Mosaic3D downgrading PyTorch below SAM3's supported version.

**Architecture:** Use Python 3.12, PyTorch 2.7.0/cu126, torchvision 0.22.0, and NumPy 1.26.4 as the shared ABI baseline. Install only the Mosaic3D inference path dependencies, load one model per pipeline stage, persist its artifacts, and deterministically release its CUDA state before constructing the next model.

**Tech Stack:** Python 3.12, PyTorch 2.7/cu126, PyG CUDA wheels, spconv, Mosaic3D, SAM3, Pixi/Mamba.

---

### Task 1: Define the shared environment

**Files:**
- Create: `requirements/segmentation-unified-cu126.txt`
- Modify: `pyproject.toml`

- [x] Pin the common ABI baseline and list only dependencies imported by the current inference path.
- [x] Exclude Open3D, cuML, dataset tooling, and Mosaic3D package metadata.
- [x] Document installation of PyTorch and compiled extensions before the Python-only dependency file.

### Task 2: Enforce sequential model lifetime

**Files:**
- Create: `src/scenecompose/segmentation/model_lifecycle.py`
- Modify: `src/scenecompose/segmentation/mosaic.py`
- Modify: `src/scenecompose/segmentation/sam3_adapter.py`
- Modify: `src/scenecompose/segmentation/pipeline.py`

- [x] Give each adapter an idempotent `release()` method.
- [x] Execute release from `finally`, including checkpoint-load and inference failures.
- [x] Remove model references, run garbage collection, and clear unused PyTorch CUDA allocations.

### Task 3: Add an environment-only acceptance check

**Files:**
- Create: `scripts/check_unified_segmentation_env.py`

- [x] Validate Python, PyTorch, torchvision, NumPy, CUDA, spconv, and PyG imports.
- [x] Import both model construction paths without loading checkpoints or downloading weights.
- [x] Leave checkpoint-backed forward validation to a single observation run on the server.

### Task 4: Update operator documentation

**Files:**
- Modify: `README.md`
- Modify: `READMECHINESE.md`
- Modify: `docs/technical-implementation-details.md`

- [x] Replace the obsolete PyTorch 2.2/Python 3.10 instructions with Mamba and Pixi commands for the shared baseline.
- [x] Explain sequential loading, artifact boundaries, GPU release behavior, and server acceptance checks.
- [x] State explicitly that `Mosaic3D/requirements.txt` and `pip install -e Mosaic3D` must not be used in this environment.

### Task 5: Static verification

- [x] Compile changed Python files without importing GPU libraries.
- [x] Search documentation for obsolete Mosaic3D installation commands.
- [x] Do not download packages, load weights, run GPU tests, or perform Git operations.
