from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ObservationInput:
    observation_id: str
    directory: Path
    manifest: Path
    geometry: Path
    source_faces: Path
    source_scene: str


def discover_observations(root: Path) -> list[ObservationInput]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Observation root not found: {root}")
    observations: list[ObservationInput] = []
    for manifest in root.rglob("observation.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        required = {
            "observation_id", "source_scene", "partition_geometry", "source_faces",
            "anchor_floor", "anchor_eye", "reference_forward", "reference_right", "up",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"{manifest} is missing keys: {', '.join(missing)}")
        directory = manifest.parent.resolve()
        geometry = (directory / str(payload["partition_geometry"])).resolve()
        source_faces = (directory / str(payload["source_faces"])).resolve()
        if not geometry.is_file() or not source_faces.is_file():
            raise ValueError(f"Observation artifacts are missing for {manifest}")
        observations.append(ObservationInput(
            str(payload["observation_id"]), directory, manifest.resolve(), geometry,
            source_faces, str(payload["source_scene"]),
        ))
    observations.sort(key=lambda item: (item.source_scene.casefold(), item.observation_id))
    return observations
