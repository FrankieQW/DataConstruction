from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scenecompose.observation.pipeline import (
    run_observation_partition,
    run_observation_partition_batch,
)
from scenecompose.segmentation.pipeline import (
    STAGES,
    run_observation_segmentation_batch,
    run_segmentation_batch,
)


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

    observations = subparsers.add_parser(
        "sample-observations", help="Sample observer-centered partitions from one FBX scene"
    )
    observations.add_argument("--scene", type=Path, required=True)
    observations.add_argument("--output", type=Path, required=True)
    observations.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "observation_partition.json",
    )
    observations.add_argument("--blender", default=os.environ.get("SCENECOMPOSE_BLENDER", "blender"))
    observations.add_argument("--source-relative", default=None)
    observations.add_argument("--force", action="store_true")
    observations.set_defaults(handler=_sample_observations)

    observations_all = subparsers.add_parser(
        "sample-observations-all", help="Sample observer-centered partitions for all FBX scenes"
    )
    observations_all.add_argument("--scene-root", type=Path, default=PROJECT_ROOT / "data" / "scene")
    observations_all.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "work")
    observations_all.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "observation_partition.json",
    )
    observations_all.add_argument("--blender", default=os.environ.get("SCENECOMPOSE_BLENDER", "blender"))
    observations_all.add_argument("--workers", type=int, default=1)
    observations_all.add_argument("--force", action="store_true")
    observations_all.set_defaults(handler=_sample_observations_all)

    segment = subparsers.add_parser(
        "segment-scenes", help="Segment complete scene files under a scene root"
    )
    segment.add_argument("--scene-root", type=Path, default=PROJECT_ROOT / "data" / "scene")
    segment.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "work")
    segment.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "segmentation.json"
    )
    segment.add_argument("--blender", default=os.environ.get("SCENECOMPOSE_BLENDER", "blender"))
    segment.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids")
    segment.add_argument("--workers", type=int, default=None)
    segment.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    segment.add_argument("--force-stage", choices=STAGES)
    segment.add_argument("--dry-run", action="store_true")
    segment.set_defaults(handler=_segment_scenes)

    segment_observations = subparsers.add_parser(
        "segment-observations", help="Segment all observation partitions under a work root"
    )
    segment_observations.add_argument("--observation-root", type=Path, default=PROJECT_ROOT / "data" / "work")
    segment_observations.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "segmentation.json"
    )
    segment_observations.add_argument("--blender", default=os.environ.get("SCENECOMPOSE_BLENDER", "blender"))
    segment_observations.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids")
    segment_observations.add_argument("--workers", type=int, default=None)
    segment_observations.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    segment_observations.add_argument("--force-stage", choices=STAGES)
    segment_observations.add_argument("--dry-run", action="store_true")
    segment_observations.set_defaults(handler=_segment_observations)
    return parser


def _sample_observations(args: argparse.Namespace) -> int:
    return run_observation_partition(
        args.scene.resolve(), args.output.resolve(), args.config.resolve(), args.blender,
        args.source_relative, args.force,
    )


def _sample_observations_all(args: argparse.Namespace) -> int:
    return run_observation_partition_batch(
        args.scene_root.resolve(), args.output_root.resolve(), args.config.resolve(),
        args.blender, args.workers, args.force,
    )


def _segment_scenes(args: argparse.Namespace) -> int:
    gpus = _parse_gpus(args.gpus)
    return run_segmentation_batch(
        scene_root=args.scene_root.resolve(), output_root=args.output_root.resolve(),
        config_path=args.config.resolve(), blender=args.blender, gpus=gpus,
        workers=args.workers, resume=args.resume, force_stage=args.force_stage,
        dry_run=args.dry_run,
    )


def _segment_observations(args: argparse.Namespace) -> int:
    return run_observation_segmentation_batch(
        observation_root=args.observation_root.resolve(), config_path=args.config.resolve(),
        blender=args.blender, gpus=_parse_gpus(args.gpus), workers=args.workers,
        resume=args.resume, force_stage=args.force_stage, dry_run=args.dry_run,
    )


def _parse_gpus(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    try:
        gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit("--gpus must be a comma-separated list of integers") from error
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise SystemExit("--gpus must contain non-negative GPU ids")
    return gpus


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
