from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ObjectAsset


EXCLUDED_DIRECTORY_NAMES = {"metadata", "cache", "work", ".cache", "weights"}


def discover_objects(object_root: Path, metadata_path: Path) -> tuple[ObjectAsset, ...]:
    root = object_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Object root not found: {root}")
    if not metadata_path.is_file():
        raise ValueError(f"Object metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Objaverse annotations root must be an object keyed by UID")
    by_uid: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".glb":
            continue
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part.casefold() in EXCLUDED_DIRECTORY_NAMES for part in relative_parts):
            continue
        uid = path.stem
        if uid in by_uid:
            raise ValueError(f"Duplicate Object UID: {uid}: {by_uid[uid]} and {path}")
        by_uid[uid] = path.resolve()
    assets: list[ObjectAsset] = []
    for uid, path in sorted(by_uid.items(), key=lambda item: item[1].relative_to(root).as_posix()):
        annotation = metadata.get(uid)
        if not isinstance(annotation, dict):
            continue
        provenance = annotation.get("_scenecompose")
        if not isinstance(provenance, dict) or provenance.get("uid") != uid:
            raise ValueError(f"Invalid _scenecompose metadata for UID: {uid}")
        relative = path.relative_to(root).as_posix()
        expected = str(provenance.get("glb_relative_path", ""))
        if expected and expected.casefold() != relative.casefold():
            raise ValueError(
                f"Object path disagrees with metadata for {uid}: {relative} != {expected}"
            )
        assets.append(ObjectAsset(uid, path, relative, annotation))
    return tuple(assets)


def normalized_metadata(annotation: dict[str, Any]) -> dict[str, object]:
    def names(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        result = []
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("name"), str):
                result.append(value["name"].strip())
            elif isinstance(value, str):
                result.append(value.strip())
        return [value for value in result if value]

    provenance = annotation.get("_scenecompose", {})
    return {
        "name": str(annotation.get("name") or "").strip(),
        "description": str(annotation.get("description") or "").strip(),
        "tags": names(annotation.get("tags")),
        "categories": names(annotation.get("categories")),
        "lvis_categories": [
            str(value).strip() for value in provenance.get("lvis_categories", [])
            if str(value).strip()
        ] if isinstance(provenance, dict) else [],
        "license": str(annotation.get("license") or "").casefold(),
    }

