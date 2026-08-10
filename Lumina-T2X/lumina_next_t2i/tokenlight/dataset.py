from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokens import LightingSchema, PackedLighting, TASK_IDS


class TokenLightDataset(Dataset):
    """Build aligned source/target pairs from linear-RGB render components."""

    def __init__(self, config: dict[str, Any], split: str):
        self.config = config
        self.split = split
        self.root = Path(config["paths"]["dataset_root"]).expanduser()
        manifest_key = {"train": "train_manifest", "validation": "validation_manifest", "test": "test_manifest"}[split]
        manifest_value = config["paths"].get(manifest_key)
        if not manifest_value:
            raise ValueError(f"paths.{manifest_key} 未配置")
        self.manifest_path = Path(manifest_value).expanduser()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest 不存在: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.scenes = [json.loads(line) for line in handle if line.strip()]
        if not self.scenes:
            raise ValueError(f"manifest 中没有 scene: {self.manifest_path}")

        data = config["data"]
        self.resolution = int(data["resolution"])
        self.crop_mode = data["crop_mode"]
        self.exposure = float(data["exposure"])
        self.tasks = tuple(data["tasks"])
        samples_key = "samples_per_scene_train" if split == "train" else "samples_per_scene_validation"
        self.samples_per_scene = int(data[samples_key])
        self.deterministic = split != "train" and bool(data.get("deterministic_validation", True))
        self.seed = int(data.get("validation_seed", 0))
        self.task_probabilities = data.get("task_probabilities", {})
        self.schema = LightingSchema(int(config["model"]["max_lights"]))
        lighting = config["lighting"]
        self.ambient_range = tuple(float(value) for value in lighting["ambient_scale_range"])
        self.color_range = tuple(float(value) for value in lighting["light_color_range"])
        self.intensity_range = tuple(float(value) for value in lighting["light_intensity_range"])
        self.fixture_intensity_range = tuple(float(value) for value in lighting["fixture_intensity_range"])
        self.fixture_transition_range = tuple(float(value) for value in lighting["fixture_transition_range"])

    def __len__(self) -> int:
        return len(self.scenes) * self.samples_per_scene

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        sample_seed = None
        if isinstance(index, tuple):
            index, sample_seed = index
        scene = self.scenes[index % len(self.scenes)]
        if sample_seed is not None:
            rng = np.random.default_rng(sample_seed)
        elif self.deterministic:
            rng = np.random.default_rng(self.seed + index)
        else:
            rng = np.random.default_rng()
        available = [task for task in self.tasks if self._supports(scene, task)]
        if not available:
            raise ValueError(f"scene {scene.get('id')} 不支持配置中的任务 {self.tasks}")
        weights = np.asarray([float(self.task_probabilities.get(task, 1.0)) for task in available], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError(f"scene {scene.get('id')} 的可用任务概率之和为 0")
        task = str(rng.choice(available, p=weights / weights.sum()))

        ambient = self._read_linear(scene["ambient"])
        dark = self._read_linear(scene["dark"]) if scene.get("dark") else np.zeros_like(ambient)
        fixture_mask = np.zeros(ambient.shape[:2], dtype=np.float32)

        if task == "ambient_scale":
            scale = float(rng.uniform(*self.ambient_range))
            source = ambient
            target = dark + (ambient - dark) * scale
            lighting = self.schema.ambient(scale)
        elif task == "global_diffuse":
            source_index, target_index = rng.choice(len(scene["diffuse"]), size=2, replace=False)
            source_component = scene["diffuse"][int(source_index)]
            target_component = scene["diffuse"][int(target_index)]
            source = ambient + self._read_linear(source_component["path"]) - dark
            target = ambient + self._read_linear(target_component["path"]) - dark
            lighting = self.schema.global_diffuse(float(target_component["level"]) - float(source_component["level"]))
        elif task == "add_light":
            component_count = min(len(scene["point_lights"]), self.schema.max_lights)
            light_count = int(rng.integers(1, component_count + 1))
            selected = rng.choice(len(scene["point_lights"]), size=light_count, replace=False)
            source = ambient
            target = ambient.copy()
            light_values: list[dict[str, float]] = []
            for component_index in selected:
                component = scene["point_lights"][int(component_index)]
                color = rng.uniform(*self.color_range, size=3).astype(np.float32)
                intensity = float(rng.uniform(*self.intensity_range))
                contribution = np.maximum(self._read_linear(component["path"]) - dark, 0.0)
                target += contribution * color[None, None, :] * intensity
                position = component["position"]
                light_values.append(
                    {
                        "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
                        "r": float(color[0]), "g": float(color[1]), "b": float(color[2]),
                        "intensity": intensity, "diffuse": float(component.get("diffuse", 0.0)),
                    }
                )
            lighting = self.schema.add_lights(light_values)
        else:
            fixtures = self._fixtures(scene)
            fixture = fixtures[int(rng.integers(len(fixtures)))]
            color = rng.uniform(*self.color_range, size=3).astype(np.float32)
            intensity = float(rng.uniform(*self.fixture_intensity_range))
            transition = float(rng.uniform(*self.fixture_transition_range))
            source = ambient
            contribution = np.maximum(self._read_linear(fixture.get("path") or fixture.get("on")) - dark, 0.0)
            target = ambient + contribution * color[None, None, :] * intensity * transition
            fixture_mask = self._read_mask(fixture["mask"])
            lighting = self.schema.in_scene(
                {"r": float(color[0]), "g": float(color[1]), "b": float(color[2]),
                 "intensity": intensity, "transition": transition}
            )

        return self._format_sample(
            scene,
            task,
            source,
            target,
            fixture_mask,
            lighting,
            sample_index=int(index),
            sample_seed=int(sample_seed) if sample_seed is not None else None,
        )

    def _format_sample(
        self,
        scene: dict[str, Any],
        task: str,
        source: np.ndarray,
        target: np.ndarray,
        fixture_mask: np.ndarray,
        lighting: PackedLighting,
        sample_index: int,
        sample_seed: int | None,
    ) -> dict[str, Any]:
        return {
            "source_image": torch.from_numpy(self._prepare_image(source)).permute(2, 0, 1),
            "target_image": torch.from_numpy(self._prepare_image(target)).permute(2, 0, 1),
            "fixture_mask": torch.from_numpy(self._prepare_mask(fixture_mask))[None],
            "fixture_present": torch.tensor(task == "in_scene_light", dtype=torch.bool),
            "lighting_values": torch.from_numpy(lighting.values.copy()),
            "lighting_known": torch.from_numpy(lighting.known.copy()),
            "lighting_valid": torch.from_numpy(lighting.valid.copy()),
            "task": torch.tensor(TASK_IDS[task], dtype=torch.long),
            "task_name": task,
            "scene_id": str(scene["id"]),
            "asset_uid": self._asset_uid(scene),
            "sample_index": sample_index,
            "sample_seed": sample_seed,
        }

    def _supports(self, scene: dict[str, Any], task: str) -> bool:
        if task == "ambient_scale":
            return bool(scene.get("ambient"))
        if task == "global_diffuse":
            return len(scene.get("diffuse", [])) >= 2
        if task == "add_light":
            return bool(scene.get("point_lights"))
        fixtures = self._fixtures(scene)
        return any(item.get("mask") and (item.get("path") or item.get("on")) for item in fixtures)

    @staticmethod
    def _fixtures(scene: dict[str, Any]) -> list[dict[str, Any]]:
        return scene.get("in_scene_lights") or scene.get("fixtures") or []

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def _read_linear(self, value: str | Path) -> np.ndarray:
        path = self._resolve(value)
        if path.suffix.lower() == ".npy":
            image = np.load(path).astype(np.float32)
        else:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            import cv2

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"无法读取线性图像: {path}")
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"期望 HWC RGB 图像，实际 {path}: {image.shape}")
        if not np.isfinite(image[..., :3]).all():
            raise ValueError(f"线性图像包含 NaN/Inf: {path}")
        return image[..., :3]

    def _read_mask(self, value: str | Path) -> np.ndarray:
        path = self._resolve(value)
        if path.suffix.lower() == ".npy":
            mask = np.load(path).astype(np.float32)
        else:
            import cv2

            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"无法读取 fixture mask: {path}")
            mask = mask.astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        maximum = float(mask.max(initial=0.0))
        if maximum > 1.0:
            mask = mask / maximum
        return np.clip(mask, 0.0, 1.0)

    def _center_crop(self, array: np.ndarray) -> np.ndarray:
        if self.crop_mode != "center":
            raise ValueError(f"data.crop_mode 当前只支持 center，实际为 {self.crop_mode}")
        height, width = array.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        return array[top : top + side, left : left + side]

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        import cv2

        image = self._center_crop(np.maximum(image * self.exposure, 0.0))
        image = image / (1.0 + image)
        if image.shape[:2] != (self.resolution, self.resolution):
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image * 2.0 - 1.0, dtype=np.float32)

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        import cv2

        mask = self._center_crop(mask)
        if mask.shape != (self.resolution, self.resolution):
            mask = cv2.resize(mask, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)

    @staticmethod
    def _asset_uid(scene: dict[str, Any]) -> str:
        if scene.get("asset_uid"):
            return str(scene["asset_uid"])
        return Path(str(scene.get("asset", scene["id"]))).stem
