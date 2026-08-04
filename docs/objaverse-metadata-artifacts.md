# Objaverse Metadata Artifacts

`scripts/download_objaverse_metadata.py` reads a GLB inventory whose object lines use `NNN-NNN/<uid>.glb`. It downloads only official Objaverse metadata and LVIS annotation files. It does not download or modify GLBs and does not download model weights.

## Command

```bash
python scripts/download_objaverse_metadata.py \
  --manifest data/obj/Objaverse.md \
  --output data/obj/metadata
```

The Linux wrapper accepts the manifest and output paths as its first two positional arguments:

```bash
bash scripts/run_objaverse_metadata.sh data/obj/Objaverse.md data/obj/metadata
```

## Layout

```text
data/obj/metadata/
|-- cache/
|   |-- metadata/
|   |   `-- 000-xxx.json.gz
|   `-- lvis-annotations.json.gz
|-- annotations.json
|-- annotations.jsonl
|-- sample.json
|-- lvis_categories.json
|-- missing_uids.txt
`-- summary.json
```

The `cache` directory contains verified official gzip JSON files. Repeating the command reuses these files. Downloads first go to uniquely named `.part` files; only gzip JSON files with an object at the root are published to the cache.

`annotations.json` is keyed by UID. Original Objaverse annotation fields are preserved, and `_scenecompose` adds:

- `uid`
- `shard`
- `glb_relative_path`
- `lvis_categories`

`annotations.jsonl` contains the same records in stable GLB path order for streaming. `sample.json` contains the first 20 records by default and is the fastest way to inspect available metadata fields. Use `--sample-size` to change its size.

`lvis_categories.json` contains only UIDs with an official LVIS match. `missing_uids.txt` includes requested UIDs absent from their expected metadata shard. Missing UIDs produce exit code 2 after valid results are written; `--allow-missing` changes this to exit code 0.

## Composition Use

Object classification should prefer labels in this order:

1. `_scenecompose.lvis_categories`
2. official `categories`
3. official `name`, `description`, and `tags`
4. geometry/render-based classification when text metadata is absent or unreliable

Metadata labels are weak supervision. They do not determine scale, upright orientation, support relation, collision shape, or placement pose.
