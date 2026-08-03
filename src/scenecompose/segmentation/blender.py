from __future__ import annotations

from pathlib import Path
import subprocess


def build_prepare_command(
    blender: str,
    script: Path,
    scene: Path,
    output: Path,
    config: Path,
    stage: str = "all",
) -> list[str]:
    if stage not in {"geometry", "views", "all"}:
        raise ValueError(f"Unsupported Blender preparation stage: {stage}")
    return [
        blender, "--background", "--factory-startup", "--python", str(script.resolve()), "--",
        "--scene", str(scene.resolve()), "--output", str(output.resolve()),
        "--config", str(config.resolve()), "--stage", stage,
    ]


def run_prepare_scene(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Blender preparation failed with exit code {completed.returncode}; see {log_path}")
