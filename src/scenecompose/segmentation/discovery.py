from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


@dataclass(frozen=True)
class SceneInput:
    scene_id: str
    source: Path
    relative_source: str


def discover_scenes(scene_root: Path, suffixes: tuple[str, ...] = (".fbx",)) -> list[SceneInput]:
    root = scene_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Scene root not found: {root}")
    allowed = {suffix.casefold() for suffix in suffixes}
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in allowed),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    return [_scene_input(root, path) for path in paths]


def _scene_input(root: Path, path: Path) -> SceneInput:
    relative = path.relative_to(root).as_posix()
    stem = Path(relative).with_suffix("").as_posix().replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "scene"
    digest = hashlib.sha256(relative.casefold().encode("utf-8")).hexdigest()[:8]
    return SceneInput(f"{safe}-{digest}", path.resolve(), relative)
