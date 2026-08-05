from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .artifacts import save_json_atomic
from .association import associate_observations
from .blender import build_prepare_command, run_prepare_scene
from .config import SegmentationConfig
from .discovery import SceneInput, discover_scenes
from .export import export_results
from .fusion import fuse_predictions, load_fused_instances
from .lifting import lift_all_masks
from .manifest import STAGES, SegmentationManifest
from .mosaic import Mosaic3DAdapter
from .preflight import run_preflight
from .recap_clip import required_recap_clip_paths
from .sam3_adapter import Sam3Adapter
from .observation_discovery import ObservationInput, discover_observations


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLENDER_SCRIPT = PROJECT_ROOT / "scripts" / "blender" / "prepare_segmentation_scene.py"


def _run_model_stage(adapter, infer: Callable[[object], list[Path]]) -> list[Path]:
    try:
        return infer(adapter)
    finally:
        adapter.release()


def _mosaic_external_files(config: SegmentationConfig) -> tuple[Path, ...]:
    text_model_dir = (PROJECT_ROOT / config.mosaic3d.text_model_path).resolve()
    return (
        (PROJECT_ROOT / config.mosaic3d.checkpoint).resolve(),
        *required_recap_clip_paths(text_model_dir).values(),
    )


def _stage(
    manifest: SegmentationManifest,
    name: str,
    config: SegmentationConfig,
    upstream_digest: str | None,
    resume: bool,
    log_path: Path,
    execute: Callable[[], list[Path]],
    external_files: tuple[Path, ...] = (),
) -> str:
    digest_builder = hashlib.sha256(config.stage_digest(name).encode("ascii"))
    for path in external_files:
        if path.is_file():
            stat = path.stat()
            digest_builder.update(str(path.resolve()).encode("utf-8"))
            digest_builder.update(str(stat.st_size).encode("ascii"))
            digest_builder.update(str(stat.st_mtime_ns).encode("ascii"))
        else:
            digest_builder.update(f"missing:{path.resolve()}".encode("utf-8"))
    digest = digest_builder.hexdigest()
    if resume and manifest.can_resume(name, digest, upstream_digest):
        return str(manifest.data["stages"][name]["artifact_digest"])
    manifest.invalidate_from(name)
    with manifest.running(name, digest, upstream_digest, log_path):
        paths = execute()
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"Stage {name} did not produce: {', '.join(missing)}")
        artifact_digest = manifest.complete_artifacts(name, paths)
    return artifact_digest


def run_scene(
    scene: SceneInput,
    output_root: Path,
    config_path: Path,
    blender: str,
    gpu: int,
    resume: bool,
    force_stage: str | None,
) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = SegmentationConfig.from_json(config_path)
    output = output_root / scene.scene_id / "segmentation"
    logs = output / "logs"
    manifest = SegmentationManifest(output / "manifest.json", scene.scene_id, scene.source, scene.relative_source)
    if force_stage:
        manifest.invalidate_from(force_stage)
    command = lambda stage: build_prepare_command(blender, BLENDER_SCRIPT, scene.source, output, config_path, stage)
    upstream: str | None = None
    upstream = _stage(
        manifest, "geometry", config, upstream, resume, logs / "geometry.log",
        lambda: _run_blender(command("geometry"), logs / "geometry.log", [
            output / "geometry" / "geometry.npz", output / "geometry" / "samples.npz",
            output / "geometry" / "mesh_mapping.npz", output / "geometry" / "sampled_points.ply",
        ]),
    )
    upstream = _stage(
        manifest, "views", config, upstream, resume, logs / "views.log",
        lambda: _run_blender(command("views"), logs / "views.log", [output / "views" / "views.json"]),
    )
    upstream = _stage(
        manifest, "mosaic3d", config, upstream, resume, logs / "mosaic3d.log",
        lambda: _run_model_stage(
            Mosaic3DAdapter(PROJECT_ROOT, config, "cuda:0"),
            lambda adapter: adapter.infer(output / "geometry" / "samples.npz", output / "mosaic3d"),
        ),
        external_files=_mosaic_external_files(config),
    )
    upstream = _stage(
        manifest, "sam3", config, upstream, resume, logs / "sam3.log",
        lambda: _run_model_stage(
            Sam3Adapter(PROJECT_ROOT, config, "cuda:0"),
            lambda adapter: adapter.infer(output / "views" / "rgb", output / "sam3"),
        ),
        external_files=((PROJECT_ROOT / config.sam3.checkpoint).resolve(),),
    )

    def run_fusion() -> list[Path]:
        observations = lift_all_masks(
            output / "sam3" / "masks.json", output / "views" / "cameras",
            output / "views" / "depth", output / "geometry" / "samples.npz", config.fusion,
        )
        import torch
        point_features = torch.load(output / "mosaic3d" / "point_features.pt", map_location="cpu", weights_only=True).float().numpy()
        proposals = associate_observations(observations, point_features, config.fusion)
        paths, _ = fuse_predictions(
            output / "geometry" / "samples.npz", output / "geometry" / "geometry.npz",
            output / "mosaic3d" / "semantic_scores.npz", proposals, config, output / "fusion",
        )
        return paths

    upstream = _stage(manifest, "fusion", config, upstream, resume, logs / "fusion.log", run_fusion)

    def run_export() -> list[Path]:
        instances = load_fused_instances(output / "fusion" / "fused_instances.npz")
        return export_results(output / "fusion" / "point_labels.npz", instances, config, output)

    _stage(manifest, "export", config, upstream, resume, logs / "export.log", run_export)
    return {"scene": scene.relative_source, "scene_id": scene.scene_id, "output": str(output), "gpu": gpu, "status": "complete"}


def run_observation(
    observation: ObservationInput,
    config_path: Path,
    blender: str,
    gpu: int,
    resume: bool,
    force_stage: str | None,
) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = SegmentationConfig.from_json(config_path)
    output = observation.directory / "segmentation"
    logs = output / "logs"
    manifest = SegmentationManifest(
        output / "manifest.json", observation.observation_id,
        observation.geometry, observation.source_scene,
    )
    if force_stage:
        manifest.invalidate_from(force_stage)
    command = lambda stage: build_prepare_command(
        blender, BLENDER_SCRIPT, observation.geometry, output, config_path, stage,
        observation=observation.manifest,
    )
    upstream: str | None = None
    upstream = _stage(
        manifest, "geometry", config, upstream, resume, logs / "geometry.log",
        lambda: _run_blender(command("geometry"), logs / "geometry.log", [
            output / "geometry" / "geometry.npz", output / "geometry" / "samples.npz",
            output / "geometry" / "mesh_mapping.npz", output / "geometry" / "sampled_points.ply",
            output / "geometry" / "observation_mapping.npz",
        ]), external_files=(observation.manifest, observation.source_faces),
    )
    upstream = _stage(
        manifest, "views", config, upstream, resume, logs / "views.log",
        lambda: _run_blender(command("views"), logs / "views.log", [output / "views" / "views.json"]),
        external_files=(observation.manifest,),
    )
    upstream = _stage(
        manifest, "mosaic3d", config, upstream, resume, logs / "mosaic3d.log",
        lambda: _run_model_stage(
            Mosaic3DAdapter(PROJECT_ROOT, config, "cuda:0"),
            lambda adapter: adapter.infer(output / "geometry" / "samples.npz", output / "mosaic3d"),
        ), external_files=_mosaic_external_files(config),
    )
    upstream = _stage(
        manifest, "sam3", config, upstream, resume, logs / "sam3.log",
        lambda: _run_model_stage(
            Sam3Adapter(PROJECT_ROOT, config, "cuda:0"),
            lambda adapter: adapter.infer(output / "views" / "rgb", output / "sam3"),
        ), external_files=((PROJECT_ROOT / config.sam3.checkpoint).resolve(),),
    )

    def run_fusion() -> list[Path]:
        observations = lift_all_masks(
            output / "sam3" / "masks.json", output / "views" / "cameras",
            output / "views" / "depth", output / "geometry" / "samples.npz", config.fusion,
        )
        import torch
        point_features = torch.load(
            output / "mosaic3d" / "point_features.pt", map_location="cpu", weights_only=True
        ).float().numpy()
        proposals = associate_observations(observations, point_features, config.fusion)
        paths, _ = fuse_predictions(
            output / "geometry" / "samples.npz", output / "geometry" / "geometry.npz",
            output / "mosaic3d" / "semantic_scores.npz", proposals, config,
            output / "fusion", observation_mapping=output / "geometry" / "observation_mapping.npz",
            camera_dir=output / "views" / "cameras", depth_dir=output / "views" / "depth",
        )
        return paths

    upstream = _stage(manifest, "fusion", config, upstream, resume, logs / "fusion.log", run_fusion)

    def run_export() -> list[Path]:
        instances = load_fused_instances(output / "fusion" / "fused_instances.npz")
        return export_results(output / "fusion" / "point_labels.npz", instances, config, output)

    _stage(manifest, "export", config, upstream, resume, logs / "export.log", run_export)
    return {
        "scene": observation.source_scene, "observation_id": observation.observation_id,
        "output": str(output), "gpu": gpu, "status": "complete",
    }


def _run_blender(command: list[str], log_path: Path, outputs: list[Path]) -> list[Path]:
    run_prepare_scene(command, log_path)
    return outputs


def _worker(payload: dict[str, object]) -> dict[str, object]:
    scene = SceneInput(str(payload["scene_id"]), Path(str(payload["source"])), str(payload["relative_source"]))
    try:
        return run_scene(
            scene, Path(str(payload["output_root"])), Path(str(payload["config_path"])),
            str(payload["blender"]), int(payload["gpu"]), bool(payload["resume"]),
            str(payload["force_stage"]) if payload["force_stage"] else None,
        )
    except Exception as error:
        return {
            "scene": scene.relative_source, "scene_id": scene.scene_id,
            "gpu": int(payload["gpu"]), "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _observation_worker(payload: dict[str, object]) -> dict[str, object]:
    observation = ObservationInput(
        str(payload["observation_id"]), Path(str(payload["directory"])),
        Path(str(payload["manifest"])), Path(str(payload["geometry"])),
        Path(str(payload["source_faces"])), str(payload["source_scene"]),
    )
    try:
        return run_observation(
            observation, Path(str(payload["config_path"])), str(payload["blender"]),
            int(payload["gpu"]), bool(payload["resume"]),
            str(payload["force_stage"]) if payload["force_stage"] else None,
        )
    except Exception as error:
        return {
            "scene": observation.source_scene, "observation_id": observation.observation_id,
            "gpu": int(payload["gpu"]), "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def run_segmentation_batch(
    scene_root: Path,
    output_root: Path,
    config_path: Path,
    blender: str,
    gpus: tuple[int, ...] | None = None,
    workers: int | None = None,
    resume: bool = True,
    force_stage: str | None = None,
    dry_run: bool = False,
) -> int:
    config = SegmentationConfig.from_json(config_path)
    selected_gpus = gpus or config.runtime.gpus
    selected_workers = workers or config.runtime.workers
    if selected_workers < 1 or not selected_gpus:
        raise ValueError("workers and gpus must not be empty")
    if force_stage is not None and force_stage not in STAGES:
        raise ValueError(f"Unknown force stage: {force_stage}")
    report = run_preflight(PROJECT_ROOT, output_root, config, blender, dry_run)
    scenes = discover_scenes(scene_root, config.runtime.supported_suffixes)
    payloads = [
        {
            "scene_id": scene.scene_id, "source": str(scene.source),
            "relative_source": scene.relative_source, "output_root": str(output_root.resolve()),
            "config_path": str(config_path.resolve()), "blender": report.blender,
            "gpu": selected_gpus[index % len(selected_gpus)], "resume": resume,
            "force_stage": force_stage,
        }
        for index, scene in enumerate(scenes)
    ]
    if dry_run:
        print(json.dumps({"warnings": report.warnings, "scenes": payloads}, indent=2))
        return 0
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(selected_workers, max(1, len(payloads)))) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{str(result['status']).upper()}] {result['scene']}", flush=True)
    results.sort(key=lambda item: str(item["scene"]).casefold())
    summary = {
        "schema_version": 1, "scene_root": str(scene_root.resolve()),
        "output_root": str(output_root.resolve()), "gpus": selected_gpus,
        "workers": selected_workers, "scene_count": len(results),
        "failed_count": sum(item["status"] != "complete" for item in results),
        "results": results,
    }
    save_json_atomic(output_root / "segmentation_summary.json", summary)
    return 1 if summary["failed_count"] else 0


def run_observation_segmentation_batch(
    observation_root: Path,
    config_path: Path,
    blender: str,
    gpus: tuple[int, ...] | None = None,
    workers: int | None = None,
    resume: bool = True,
    force_stage: str | None = None,
    dry_run: bool = False,
) -> int:
    config = SegmentationConfig.from_json(config_path)
    selected_gpus = gpus or config.runtime.gpus
    selected_workers = workers or config.runtime.workers
    if selected_workers < 1 or not selected_gpus:
        raise ValueError("workers and gpus must not be empty")
    if force_stage is not None and force_stage not in STAGES:
        raise ValueError(f"Unknown force stage: {force_stage}")
    report = run_preflight(PROJECT_ROOT, observation_root, config, blender, dry_run)
    observations = discover_observations(observation_root)
    payloads = [
        {
            "observation_id": item.observation_id, "directory": str(item.directory),
            "manifest": str(item.manifest), "geometry": str(item.geometry),
            "source_faces": str(item.source_faces), "source_scene": item.source_scene,
            "config_path": str(config_path.resolve()), "blender": report.blender,
            "gpu": selected_gpus[index % len(selected_gpus)], "resume": resume,
            "force_stage": force_stage,
        }
        for index, item in enumerate(observations)
    ]
    if dry_run:
        print(json.dumps({"warnings": report.warnings, "observations": payloads}, indent=2))
        return 0
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(selected_workers, max(1, len(payloads)))) as executor:
        futures = [executor.submit(_observation_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{str(result['status']).upper()}] {result['observation_id']}", flush=True)
    results.sort(key=lambda item: str(item["observation_id"]))
    summary = {
        "schema_version": 1, "observation_root": str(observation_root.resolve()),
        "gpus": selected_gpus, "workers": selected_workers,
        "observation_count": len(results),
        "failed_count": sum(item["status"] != "complete" for item in results),
        "results": results,
    }
    save_json_atomic(observation_root / "observation_segmentation_summary.json", summary)
    return 1 if summary["failed_count"] else 0
