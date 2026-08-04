# Local ReCap-CLIP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Hugging Face Hub/cache-based ReCap-CLIP loading with a project-local weight bundle.

**Architecture:** Add a small loader that constructs OpenCLIP directly from the downloaded config, checkpoint, and tokenizer files. Point segmentation configuration and preflight at `weights/recap-clip`, while retaining the Mosaic3D adapter's existing text-encoder interface.

**Tech Stack:** Python, OpenCLIP, Transformers tokenizer files, JSON configuration.

---

### Task 1: Configuration Contract

**Files:**
- Modify: `src/scenecompose/segmentation/config.py`
- Modify: `configs/segmentation.json`

- [x] Replace `text_model_id` with project-relative `text_model_path` and validate that it is non-empty.
- [x] Keep the field in the Mosaic3D stage digest through the existing dataclass serialization.

### Task 2: Local Loader And Adapter Integration

**Files:**
- Create: `src/scenecompose/segmentation/recap_clip.py`
- Modify: `src/scenecompose/segmentation/mosaic.py`

- [x] Validate and parse `open_clip_config.json` without importing Hugging Face Hub helpers.
- [x] Build `open_clip.CLIP`, load the local checkpoint strictly, and build `open_clip.HFTokenizer` from the local directory with `local_files_only=True`.
- [x] Replace Mosaic3D's `hf-hub:` loader call while preserving the `text_tokenizer` and `encode_text` interface.

### Task 3: Preflight Validation

**Files:**
- Modify: `src/scenecompose/segmentation/preflight.py`

- [x] Require the ReCap-CLIP directory and all files needed for local model/tokenizer construction.
- [x] Report each missing artifact with a stable descriptive label.

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `READMECHINESE.md`
- Modify: `docs/technical-implementation-details.md`

- [x] Document the flat `weights/recap-clip` layout and remove the Hugging Face cache instruction.
- [x] Explain direct local model construction and tokenizer loading in the technical document.

### Task 5: Static Verification

**Files:**
- Verify all modified Python, JSON, and Markdown files.

- [x] Parse Python files with `ast.parse` and JSON with `json.load` without importing ML dependencies.
- [x] Search runtime/configuration/docs for stale `text_model_id`, ReCap-CLIP `hf-hub:` identifiers, and cache-based README guidance.
- [x] Confirm documentation code fences remain balanced.

No Git operations, downloads, or model inference are part of this plan.
