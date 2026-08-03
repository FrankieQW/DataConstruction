from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _blender_path(value: str) -> str:
    blender = shutil.which(value)
    if blender is None:
        raise SystemExit(f"Blender executable not found: {value}")
    return blender


def _partition_command(
    *,
    blender: str,
    scene: Path,
    output: Path,
    config: Path,
    force: bool,
) -> list[str]:
    blender_script = PROJECT_ROOT / "scripts" / "blender" / "partition_scene.py"
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(blender_script),
        "--",
        "--scene",
        str(scene),
        "--output",
        str(output),
        "--config",
        str(config),
    ]
    if force:
        command.append("--force")
    return command


def _run_partition(
    *, blender: str, scene: Path, output: Path, config: Path, force: bool
) -> int:
    command = _partition_command(
        blender=blender,
        scene=scene,
        output=output,
        config=config,
        force=force,
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def _partition(args: argparse.Namespace) -> int:
    scene = args.scene.resolve()
    config = args.config.resolve()
    if not scene.is_file():
        raise SystemExit(f"Scene file not found: {scene}")
    if not config.is_file():
        raise SystemExit(f"Partition config not found: {config}")
    return _run_partition(
        blender=_blender_path(args.blender),
        scene=scene,
        output=args.output.resolve(),
        config=config,
        force=args.force,
    )


def _partition_all(args: argparse.Namespace) -> int:
    scene_root = args.scene_root.resolve()
    output_root = args.output_root.resolve()
    config = args.config.resolve()
    if not scene_root.is_dir():
        raise SystemExit(f"Scene root not found: {scene_root}")
    if not config.is_file():
        raise SystemExit(f"Partition config not found: {config}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    blender = _blender_path(args.blender)
    scenes = sorted(
        (path for path in scene_root.rglob("*") if path.is_file() and path.suffix.casefold() == ".fbx"),
        key=lambda path: path.relative_to(scene_root).as_posix(),
    )
    if not scenes:
        print(f"No FBX scenes found under {scene_root}")
        return 0

    def run(scene: Path) -> tuple[Path, Path, int]:
        relative = scene.relative_to(scene_root)
        output = output_root / relative.parent / relative.stem / "partition"
        code = _run_partition(
            blender=blender,
            scene=scene,
            output=output,
            config=config,
            force=args.force,
        )
        return scene, output, code

    results: list[tuple[Path, Path, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, scene): scene for scene in scenes}
        for future in as_completed(futures):
            scene, output, code = future.result()
            results.append((scene, output, code))
            state = "OK" if code == 0 else f"FAILED({code})"
            print(f"[{state}] {scene} -> {output}", flush=True)

    results.sort(key=lambda item: item[0].relative_to(scene_root).as_posix())
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.json"
    summary = {
        "schema_version": 1,
        "scene_root": str(scene_root),
        "output_root": str(output_root),
        "workers": args.workers,
        "scene_count": len(results),
        "failed_count": sum(code != 0 for _, _, code in results),
        "results": [
            {
                "scene": scene.relative_to(scene_root).as_posix(),
                "output": output.relative_to(output_root).as_posix(),
                "exit_code": code,
            }
            for scene, output, code in results
        ],
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(summary_path)
    return 1 if summary["failed_count"] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenecompose")
    subparsers = parser.add_subparsers(dest="command", required=True)
    partition = subparsers.add_parser("partition", help="Adaptively partition one FBX scene")
    partition.add_argument("--scene", type=Path, required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "partition.json",
    )
    partition.add_argument("--blender", default="blender")
    partition.add_argument("--force", action="store_true")
    partition.set_defaults(handler=_partition)

    partition_all = subparsers.add_parser(
        "partition-all", help="Recursively partition every FBX under a scene root"
    )
    partition_all.add_argument("--scene-root", type=Path, required=True)
    partition_all.add_argument("--output-root", type=Path, required=True)
    partition_all.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "partition.json",
    )
    partition_all.add_argument("--blender", default="blender")
    partition_all.add_argument("--workers", type=int, default=1)
    partition_all.add_argument("--force", action="store_true")
    partition_all.set_defaults(handler=_partition_all)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
