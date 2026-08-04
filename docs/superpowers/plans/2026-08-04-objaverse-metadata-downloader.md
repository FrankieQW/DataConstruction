# Objaverse Metadata Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable standard-library downloader that extracts metadata for the GLB UIDs listed in an Objaverse manifest and writes inspectable JSON artifacts entirely below the requested output directory.

**Architecture:** A single CLI script separates pure manifest/index transformations from network/cache I/O. Official gzip JSON shards are atomically cached, validated before use, filtered to requested UIDs, enriched with local path and LVIS category provenance, then published as deterministic JSON/JSONL artifacts.

**Tech Stack:** Python 3.10+ standard library (`argparse`, `dataclasses`, `gzip`, `json`, `pathlib`, `urllib.request`, `tempfile`, `time`).

**Execution constraints:** Do not download data, run network tests, or perform Git operations. The user will run download validation. Only static syntax/configuration inspection is performed locally.

---

### Task 1: Define manifest records and parser

**Files:**
- Create: `scripts/download_objaverse_metadata.py`

- [ ] Add immutable `ObjectRecord(shard, uid, glb_relative_path)` and strict `NNN-NNN/<uid>.glb` parsing.
- [ ] Ignore headings and unrelated Markdown lines, deduplicate identical records, reject cross-shard UID conflicts, and stable-sort by relative path.
- [ ] Reject missing manifests and manifests containing no usable GLB paths.

### Task 2: Implement cache download and validation

**Files:**
- Modify: `scripts/download_objaverse_metadata.py`

- [ ] Add URL normalization and official metadata/LVIS URL construction.
- [ ] Validate cached gzip JSON as a top-level object before treating it as a cache hit.
- [ ] Download to a unique `.part` file, validate it, and atomically replace the cache target.
- [ ] Retry bounded network/HTTP/gzip errors with finite exponential backoff.
- [ ] Move an invalid existing cache file to a unique `.invalid-*` sibling before downloading a replacement.
- [ ] Track `cache_hit` and `downloaded` counts without writing outside `--output`.

### Task 3: Filter annotations and join LVIS categories

**Files:**
- Modify: `scripts/download_objaverse_metadata.py`

- [ ] Group requested records by shard and read each shard once.
- [ ] Require each selected annotation to be a JSON object and reject collision with reserved `_scenecompose`.
- [ ] Invert official `{category: [uid...]}` LVIS data only for requested UIDs.
- [ ] Preserve missing UIDs and nonstandard UID strings explicitly.
- [ ] Inject `_scenecompose.uid`, shard, GLB relative path, and sorted LVIS categories into copied annotations.

### Task 4: Build and atomically publish output artifacts

**Files:**
- Modify: `scripts/download_objaverse_metadata.py`

- [ ] Build deterministic `annotations.json`, `annotations.jsonl`, `sample.json`, `lvis_categories.json`, `missing_uids.txt`, and `summary.json` payloads.
- [ ] Serialize every output to a unique sibling temporary file before replacing its final path.
- [ ] Ensure JSON uses UTF-8, preserves non-ASCII metadata text, and has stable key ordering.
- [ ] Report all output paths relative to the output root in `summary.json`.

### Task 5: Add CLI, exit codes, and wrapper

**Files:**
- Modify: `scripts/download_objaverse_metadata.py`
- Create: `scripts/run_objaverse_metadata.sh`

- [ ] Add and validate all design parameters: manifest/output/sample size/timeout/retries/base URL/force/allow missing/skip LVIS.
- [ ] Print concise shard progress and a final summary without printing annotation contents.
- [ ] Return `0` on success, `2` for reported missing UIDs unless `--allow-missing`, and `1` for fatal validation/download errors.
- [ ] Add a Linux Bash wrapper whose defaults point to `data/obj/Objaverse.md` and `data/obj/metadata`.

### Task 6: Document installation and inspection workflow

**Files:**
- Modify: `README.md`
- Modify: `READMECHINESE.md`
- Create: `docs/objaverse-metadata-artifacts.md`

- [ ] Explain that the downloader uses only the standard library and does not download GLBs or weights.
- [ ] Document the direct command and Bash wrapper.
- [ ] Document cache/output structure, missing UID behavior, resumption, and `sample.json` inspection.
- [ ] Record how later composition should prefer LVIS labels and fall back to name/description/tags/categories.

### Task 7: Static verification and user handoff

**Files:**
- Inspect all files changed above.

- [ ] Parse the downloader with Python `ast` without importing or executing it.
- [ ] Check shell wrapper structure without invoking downloads.
- [ ] Scan the implementation for default home-directory writes and verify all defaults resolve under the project root.
- [ ] Provide the exact user-run command and state explicitly that network/download behavior was not executed locally.
