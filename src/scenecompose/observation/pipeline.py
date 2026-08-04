from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess

from scenecompose.segmentation.discovery import SceneInput, discover_scenes


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLENDER_SCRIPT = PROJECT_ROOT / "scripts" / "blender" / "sample_observation_partitions.py"


def resolve_blender(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise ValueError(f"Blender executable not found: {value}")
    return resolved


def build_command(
    blender: str,
    scene: Path,
    output: Path,
    config: Path,
    source_relative: str,
    force: bool,
) -> list[str]:
    command = [
        blender, "--background", "--factory-startup", "--python", str(BLENDER_SCRIPT), "--",
        "--scene", str(scene.resolve()), "--output", str(output.resolve()),
        "--config", str(config.resolve()), "--source-relative", source_relative,
    ]
    if force:
        command.append("--force")
    return command


def run_observation_partition(
    scene: Path,
    output: Path,
    config: Path,
    blender: str,
    source_relative: str | None = None,
    force: bool = False,
) -> int:
    command = build_command(
        resolve_blender(blender), scene, output, config, source_relative or scene.name, force
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def run_observation_partition_batch(
    scene_root: Path,
    output_root: Path,
    config: Path,
    blender: str,
    workers: int,
    force: bool = False,
) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    scenes = discover_scenes(scene_root, (".fbx",))
    executable = resolve_blender(blender)
    output_root.mkdir(parents=True, exist_ok=True)

    def run(scene: SceneInput) -> dict[str, object]:
        output = output_root / scene.scene_id / "observations"
        code = run_observation_partition(
            scene.source, output, config, executable, scene.relative_source, force
        )
        return {
            "scene": scene.relative_source,
            "scene_id": scene.scene_id,
            "output": output.relative_to(output_root).as_posix(),
            "exit_code": code,
        }

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(scenes)))) as executor:
        futures = [executor.submit(run, scene) for scene in scenes]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["exit_code"] == 0 else f"FAILED({result['exit_code']})"
            print(f"[{status}] {result['scene']}", flush=True)
    results.sort(key=lambda item: str(item["scene"]).casefold())
    summary = {
        "schema_version": 1,
        "scene_root": str(scene_root.resolve()),
        "output_root": str(output_root.resolve()),
        "workers": workers,
        "scene_count": len(results),
        "failed_count": sum(item["exit_code"] != 0 for item in results),
        "results": results,
    }
    temporary = output_root / "observation_partition_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_root / "observation_partition_summary.json")
    return 1 if summary["failed_count"] else 0
