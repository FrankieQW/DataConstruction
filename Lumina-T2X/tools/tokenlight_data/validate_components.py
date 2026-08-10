from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TokenLight linear components and UID-safe splits.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    root = Path(config["paths"]["dataset_root"]).expanduser()
    data_config = config["data"]
    require_composition = bool(data_config.get("require_composition_contract", False))
    split_profile = str(data_config.get("split_profile", "object-held-out"))
    if split_profile not in {"object-held-out", "scene-held-out"}:
        raise ValueError("data.split_profile must be object-held-out or scene-held-out")
    manifests = {
        split: config["paths"].get(f"{split}_manifest")
        for split in ("train", "validation", "test")
        if config["paths"].get(f"{split}_manifest")
    }
    errors: list[dict[str, Any]] = []
    seen_uids: dict[str, str] = {}
    seen_base_scenes: dict[str, str] = {}
    counts = {
        "scenes": 0,
        "point_lights": 0,
        "diffuse": 0,
        "in_scene_lights": 0,
        "scene_native_fixtures": 0,
        "procedural_fallback_fixtures": 0,
        "task_eligible_scenes": {task: 0 for task in data_config["tasks"]},
    }
    manifest_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for split, manifest_value in manifests.items():
        manifest = Path(manifest_value).expanduser()
        if not manifest.is_file():
            errors.append({"split": split, "error": "missing_manifest", "path": str(manifest)})
            continue
        manifest_rows.append((split, read_jsonl(manifest)))
    total_scenes = sum(len(rows) for _, rows in manifest_rows)
    with tqdm(total=total_scenes, desc="validate components", unit="scene", dynamic_ncols=True) as progress:
        for split, rows in manifest_rows:
            for row in rows:
                counts["scenes"] += 1
                uid = str(row.get("asset_uid") or Path(row.get("asset", row["id"])).stem)
                previous = seen_uids.setdefault(uid, split)
                if previous != split and split_profile == "object-held-out":
                    errors.append({"scene_id": row["id"], "asset_uid": uid, "error": "asset_uid_split_leakage", "splits": [previous, split]})
                base_scene_id = str(row.get("base_scene_id", ""))
                if base_scene_id:
                    previous_scene = seen_base_scenes.setdefault(base_scene_id, split)
                    if previous_scene != split and split_profile == "scene-held-out":
                        errors.append(
                            {
                                "scene_id": row["id"],
                                "base_scene_id": base_scene_id,
                                "error": "base_scene_split_leakage",
                                "splits": [previous_scene, split],
                            }
                        )
                try:
                    validate_scene(root, row, counts, require_composition=require_composition)
                except Exception as error:
                    errors.append({"scene_id": row.get("id"), "split": split, "error": type(error).__name__, "detail": str(error)})
                progress.update(1)
                progress.set_postfix(split=split, errors=len(errors))
    for task in data_config["tasks"]:
        if counts["task_eligible_scenes"][task] == 0:
            errors.append({"error": "configured_task_has_no_eligible_scene", "task": task})
    report_path = root / "validation_errors.jsonl"
    write_jsonl(report_path, errors)
    summary = {
        **counts,
        "split_profile": split_profile,
        "require_composition_contract": require_composition,
        "errors": len(errors),
        "error_log": str(report_path),
    }
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError(f"数据验证发现 {len(errors)} 个错误，详见 {report_path}")


def validate_scene(
    root: Path,
    scene: dict[str, Any],
    counts: dict[str, Any],
    *,
    require_composition: bool,
) -> None:
    for key in ("id", "asset", "ambient", "dark", "point_lights", "diffuse", "camera", "canonical"):
        if key not in scene:
            raise ValueError(f"manifest 缺少字段 {key}")
    if require_composition:
        validate_composition_contract(scene)
    ambient = read_linear(resolve(root, scene["ambient"]))
    dark = read_linear(resolve(root, scene["dark"]))
    if ambient.shape != dark.shape:
        raise ValueError(f"ambient/dark shape 不一致: {ambient.shape} vs {dark.shape}")
    if ambient.shape[2] < 3:
        raise ValueError(f"图像不是 RGB: {ambient.shape}")
    point_energies = []
    for light in scene["point_lights"]:
        require_vector(light.get("position"), 3, "point_light.position")
        if "renderer_position" in light:
            require_vector(light.get("renderer_position"), 3, "point_light.renderer_position")
        component = read_linear(resolve(root, light["path"]))
        require_shape(component, ambient, light["path"])
        point_energies.append(float(np.maximum(component - dark, 0).mean()))
        counts["point_lights"] += 1
    if point_energies and max(point_energies) <= 1e-8:
        raise ValueError("所有 point-light contribution 均为全黑")
    diffuse_means = []
    for item in scene["diffuse"]:
        component = read_linear(resolve(root, item["path"]))
        require_shape(component, ambient, item["path"])
        diffuse_means.append(float(component.mean()))
        counts["diffuse"] += 1
    if len(diffuse_means) >= 2 and max(diffuse_means) - min(diffuse_means) <= 1e-8:
        raise ValueError("diffuse levels 没有可测量差异")
    for fixture in scene.get("in_scene_lights", []):
        component = read_linear(resolve(root, fixture["path"]))
        require_shape(component, ambient, fixture["path"])
        contribution = np.maximum(component - dark, 0)
        if float(contribution.mean()) <= 1e-8:
            raise ValueError(f"in-scene light contribution 全黑: {fixture['path']}")
        mask = read_mask(resolve(root, fixture["mask"]))
        if mask.shape != ambient.shape[:2]:
            raise ValueError(f"fixture mask shape 不一致: {mask.shape} vs {ambient.shape[:2]}")
        if float(mask.max()) <= 0:
            raise ValueError(f"fixture mask 为空: {fixture['mask']}")
        source = fixture.get("fixture_source")
        if source not in {"scene_native", "procedural_fallback"}:
            raise ValueError(f"fixture_source 非法: {source}")
        entity_id = fixture.get("fixture_entity_id")
        if source == "scene_native" and not entity_id:
            raise ValueError("scene_native fixture 缺少 fixture_entity_id")
        if source == "procedural_fallback" and entity_id is not None:
            raise ValueError("procedural_fallback fixture 不得声明 scene entity")
        require_vector(fixture.get("position"), 3, "fixture.position")
        require_vector(fixture.get("renderer_position"), 3, "fixture.renderer_position")
        counts[f"{source}_fixtures"] += 1
        counts["in_scene_lights"] += 1
    supported = {
        "ambient_scale": bool(scene.get("ambient")),
        "global_diffuse": len(scene.get("diffuse", [])) >= 2,
        "add_light": bool(scene.get("point_lights")),
        "in_scene_light": bool(scene.get("in_scene_lights")),
    }
    for task in counts["task_eligible_scenes"]:
        counts["task_eligible_scenes"][task] += int(supported.get(task, False))


def validate_composition_contract(scene: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "base_scene_id",
        "base_scene_fingerprint",
        "composition",
        "lighting_profile",
        "lineage",
        "license",
        "in_scene_lights",
    )
    for key in required:
        if key not in scene:
            raise ValueError(f"composition manifest 缺少字段 {key}")
    require_digest(scene["base_scene_fingerprint"], "base_scene_fingerprint")
    camera = scene["camera"]
    if not isinstance(camera, dict):
        raise ValueError("camera 必须是 mapping")
    require_vector(camera.get("location"), 3, "camera.location")
    require_vector(camera.get("rotation_euler"), 3, "camera.rotation_euler")
    require_vector(camera.get("target"), 3, "camera.target")
    if camera.get("coordinate_space") != "blender_world_meter":
        raise ValueError("camera.coordinate_space 必须是 blender_world_meter")
    canonical = scene["canonical"]
    if not isinstance(canonical, dict):
        raise ValueError("canonical 必须是 mapping")
    require_vector(canonical.get("origin"), 3, "canonical.origin")
    if float(canonical.get("asset_size", 0)) <= 0:
        raise ValueError("canonical.asset_size 必须为正数")
    if canonical.get("position_axes") != "x=right,y=camera-forward,z=up":
        raise ValueError("canonical.position_axes 与 TokenLight reader 约定不一致")
    composition = scene["composition"]
    if composition.get("relation") not in {"place_on", "replace"}:
        raise ValueError("composition.relation 非法")
    if not composition.get("target_entity_id"):
        raise ValueError("composition.target_entity_id 为空")
    require_vector(composition.get("object_transform_world"), 16, "composition.object_transform_world")
    if int(composition.get("inserted_visible_pixels", 0)) <= 0:
        raise ValueError("composition 插入对象没有可见像素")
    lineage = scene["lineage"]
    for key in (
        "annotation_digest",
        "object_digest",
        "scene_digest",
        "prepared_geometry_digest",
        "config_digest",
        "generator_version",
    ):
        if key not in lineage:
            raise ValueError(f"lineage 缺少字段 {key}")
        if key.endswith("digest"):
            require_digest(lineage[key], f"lineage.{key}")
    license_record = scene["license"]
    if license_record.get("decision") != "allowed":
        raise ValueError("license decision 不是 allowed")
    if not license_record.get("policy_version") or license_record["policy_version"] == "unconfigured":
        raise ValueError("license policy_version 未配置")


def require_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} 必须包含 {length} 个数")
    result = [float(item) for item in value]
    if not np.isfinite(result).all():
        raise ValueError(f"{label} 包含 NaN/Inf")
    return result


def require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} 不是规范 sha256 digest")


def read_linear(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path).astype(np.float32)
    else:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV 无法读取 {path}")
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image[..., :3].astype(np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError(f"无效线性图像 {path}: shape={image.shape}, finite={np.isfinite(image).all()}")
    return image


def require_shape(image: np.ndarray, reference: np.ndarray, name: str) -> None:
    if image.shape != reference.shape:
        raise ValueError(f"component shape 不一致 {name}: {image.shape} vs {reference.shape}")


def read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"OpenCV 无法读取 fixture mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
