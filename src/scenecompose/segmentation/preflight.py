from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

from .config import SegmentationConfig
from .recap_clip import required_recap_clip_paths


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreflightReport:
    blender: str
    warnings: tuple[str, ...]


def run_preflight(project_root: Path, output_root: Path, config: SegmentationConfig, blender: str, dry_run: bool) -> PreflightReport:
    failures: list[str] = []
    warnings: list[str] = []
    resolved_blender = shutil.which(blender) or (str(Path(blender).resolve()) if Path(blender).is_file() else "")
    if not resolved_blender:
        failures.append(f"Blender executable not found: {blender}")
    if not sys.platform.startswith("linux"):
        message = f"Server execution requires Linux; current platform is {sys.platform}"
        (warnings if dry_run else failures).append(message)
    text_model_dir = project_root / config.mosaic3d.text_model_path
    required = {
        "Mosaic3D repository": project_root / config.mosaic3d.repository,
        "SAM3 repository": project_root / config.sam3.repository,
        "Open3DIS repository": project_root / "Open3DIS",
        "Mosaic3D checkpoint": project_root / config.mosaic3d.checkpoint,
        "ReCap-CLIP model directory": text_model_dir,
        "SAM3 checkpoint": project_root / config.sam3.checkpoint,
    }
    if text_model_dir.is_dir():
        required.update({f"ReCap-CLIP {name}": path for name, path in required_recap_clip_paths(text_model_dir).items()})
    if config.sam3.bpe_path:
        required["SAM3 BPE vocabulary"] = project_root / config.sam3.bpe_path
    for label, path in required.items():
        if not path.exists():
            failures.append(f"{label} not found: {path}")
    parent = output_root.resolve()
    existing = next((candidate for candidate in (parent, *parent.parents) if candidate.exists()), None)
    if existing is None or not existing.is_dir():
        failures.append(f"No existing output parent for: {output_root}")
    if failures:
        raise PreflightError("Preflight failed:\n- " + "\n- ".join(failures))
    return PreflightReport(resolved_blender, tuple(warnings))
