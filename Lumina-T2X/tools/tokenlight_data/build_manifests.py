from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild UID-safe TokenLight manifests from rendered metadata.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    output_root = Path(config["paths"]["render_output_root"]).expanduser()
    metadata_paths = sorted((output_root / "components").glob("*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"没有找到 metadata.json: {output_root / 'components'}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    split_profile = str(config["data"].get("split_profile", "object-held-out"))
    if split_profile not in {"object-held-out", "scene-held-out"}:
        raise ValueError("data.split_profile must be object-held-out or scene-held-out")
    require_composition = bool(config["data"].get("require_composition_contract", False))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        uid = str(row.get("asset_uid") or Path(row["asset"]).stem)
        row["asset_uid"] = uid
        if require_composition:
            if not row.get("base_scene_id") or not row.get("lineage"):
                raise ValueError(f"composition metadata incomplete: {row.get('id')}")
            if row.get("license", {}).get("decision") != "allowed":
                raise ValueError(f"license gate rejected sample: {row.get('id')}")
        group = uid if split_profile == "object-held-out" else str(row.get("base_scene_id", ""))
        if not group:
            raise ValueError(f"sample has no split group for {split_profile}: {row.get('id')}")
        grouped.setdefault(group, []).append(row)

    render = config["render"]
    fractions = [float(render[key]) for key in ("train_fraction", "validation_fraction", "test_fraction")]
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError("render split fraction 之和必须为 1")
    group_ids = sorted(grouped)
    if split_profile == "scene-held-out" and len(group_ids) < 3:
        raise ValueError("scene-held-out requires at least three distinct base scenes")
    random.Random(int(render["split_seed"])).shuffle(group_ids)
    validation_count = round(len(group_ids) * fractions[1])
    test_count = round(len(group_ids) * fractions[2])
    validation = set(group_ids[:validation_count])
    test = set(group_ids[validation_count : validation_count + test_count])
    split_rows = {"train": [], "validation": [], "test": []}
    for group, group_rows in grouped.items():
        split = "validation" if group in validation else "test" if group in test else "train"
        split_rows[split].extend(group_rows)
    for split, values in split_rows.items():
        write_jsonl(output_root / "manifests" / f"{split}.jsonl", sorted(values, key=lambda row: row["id"]))
    release = {
        "schema_version": 1,
        "split_profile": split_profile,
        "split_seed": int(render["split_seed"]),
        "counts": {split: len(values) for split, values in split_rows.items()},
        "groups": {split: len({group_key(row, split_profile) for row in values}) for split, values in split_rows.items()},
        "fixture_sources": fixture_source_counts(rows),
        "license_policy_versions": sorted(
            {str(row.get("license", {}).get("policy_version")) for row in rows}
        ),
        "lineage_digests": sorted(
            {
                value
                for row in rows
                for key, value in row.get("lineage", {}).items()
                if key.endswith("digest")
            }
        ),
        "manifest_sha256": {
            split: sha256_file(output_root / "manifests" / f"{split}.jsonl")
            for split in split_rows
        },
    }
    (output_root / "dataset_release.json").write_text(
        json.dumps(release, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release, ensure_ascii=False), flush=True)


def load_yaml(path: str) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config 根节点必须是 mapping")
    return config


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def group_key(row: dict, split_profile: str) -> str:
    return str(row["asset_uid"] if split_profile == "object-held-out" else row["base_scene_id"])


def fixture_source_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"scene_native": 0, "procedural_fallback": 0}
    for row in rows:
        for fixture in row.get("in_scene_lights", []):
            source = fixture.get("fixture_source")
            if source in counts:
                counts[source] += 1
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


if __name__ == "__main__":
    main()
