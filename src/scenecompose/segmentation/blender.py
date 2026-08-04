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
    observation: Path | None = None,
) -> list[str]:
    if stage not in {"geometry", "views", "all"}:
        raise ValueError(f"Unsupported Blender preparation stage: {stage}")
    command = [
        blender, "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", str(script.resolve()), "--",
        "--scene", str(scene.resolve()), "--output", str(output.resolve()),
        "--config", str(config.resolve()), "--stage", stage,
    ]
    if observation is not None:
        command.extend(("--observation", str(observation.resolve())))
    return command


def run_prepare_scene(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Blender preparation failed with exit code {completed.returncode}; see {log_path}")
