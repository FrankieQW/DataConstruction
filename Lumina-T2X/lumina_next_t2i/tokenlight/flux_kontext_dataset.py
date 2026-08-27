from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

from .config import load_config
from .dataset import TokenLightDataset


def load_flux_kontext_config(path: str | Path) -> dict[str, Any]:
    """Load the separate FLUX-Kontext LoRA configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("FLUX Kontext configuration root must be a mapping")

    required = {
        "paths": ("pretrained_model", "tokenlight_config", "output_root"),
        "data": ("resolution", "samples_per_scene_train", "samples_per_scene_validation", "seed"),
        "lora": ("rank", "alpha", "dropout", "target_modules"),
        "train": (
            "mixed_precision", "micro_batch_size", "gradient_accumulation_steps",
            "max_steps", "learning_rate", "weight_decay", "max_grad_norm",
            "checkpointing_steps", "seed",
        ),
    }
    for section, names in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")
        for name in names:
            if config[section].get(name) in (None, ""):
                raise ValueError(f"missing configuration value: {section}.{name}")

    resolution = int(config["data"]["resolution"])
    if resolution < 256 or resolution % 16:
        raise ValueError("data.resolution must be >= 256 and divisible by 16")
    if config["train"]["mixed_precision"] not in {"bf16", "fp16", "no"}:
        raise ValueError("train.mixed_precision must be bf16, fp16, or no")
    for key in ("rank", "alpha"):
        if int(config["lora"][key]) < 1:
            raise ValueError(f"lora.{key} must be positive")
    if not config["lora"]["target_modules"]:
        raise ValueError("lora.target_modules cannot be empty")

    config["_config_path"] = str(config_path)
    return config


class FluxKontextTokenLightDataset(Dataset):
    """Deterministic Kontext condition/target/instruction view of TokenLight."""

    def __init__(self, flux_config: dict[str, Any], split: str):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split}")
        tokenlight_path = Path(flux_config["paths"]["tokenlight_config"]).expanduser()
        tokenlight = deepcopy(load_config(tokenlight_path))
        data = flux_config["data"]
        tokenlight["data"]["resolution"] = int(data["resolution"])
        tokenlight["data"]["samples_per_scene_train"] = int(data["samples_per_scene_train"])
        tokenlight["data"]["samples_per_scene_validation"] = int(data["samples_per_scene_validation"])
        if data.get("tasks"):
            tokenlight["data"]["tasks"] = list(data["tasks"])
        if data.get("task_probabilities"):
            tokenlight["data"]["task_probabilities"] = dict(data["task_probabilities"])
        manifest_override = flux_config["paths"].get(f"{split}_manifest")
        if manifest_override:
            tokenlight["paths"][f"{split}_manifest"] = manifest_override

        self.base = TokenLightDataset(tokenlight, split)
        self.seed = int(data["seed"]) + {"train": 0, "validation": 10_000_000, "test": 20_000_000}[split]
        self.schema = self.base.schema

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Passing a tuple selects TokenLight's seeded path even for training.
        sample = self.base[(int(index), self.seed + int(index))]
        return {
            "condition_pixel_values": sample["source_image"],
            "target_pixel_values": sample["target_image"],
            "prompt": self._instruction(sample),
            "task_name": sample["task_name"],
            "scene_id": sample["scene_id"],
            "sample_index": int(index),
            "sample_seed": self.seed + int(index),
        }

    def _value(self, sample: dict[str, Any], name: str) -> float:
        return float(sample["lighting_values"][self.schema.index[name]])

    def _instruction(self, sample: dict[str, Any]) -> str:
        task = sample["task_name"]
        if task == "ambient_scale":
            scale = self._value(sample, "ambient")
            return (
                f"Relight this exact image by setting the ambient illumination to {scale:.3f} times "
                "its current level. Preserve the camera, geometry, identity, materials, and composition."
            )
        if task == "global_diffuse":
            delta = self._value(sample, "global_diffuse")
            return (
                f"Relight this exact image by changing global light softness by {delta:+.3f}. "
                "Preserve the camera, geometry, identity, materials, and composition."
            )
        if task == "add_light":
            descriptions: list[str] = []
            for slot in range(self.schema.max_lights):
                if self._value(sample, f"add_light.{slot}.valid") < 0.5:
                    continue
                values = {
                    field: self._value(sample, f"add_light.{slot}.{field}")
                    for field in ("x", "y", "z", "r", "g", "b", "intensity", "diffuse")
                }
                descriptions.append(
                    "light at canonical xyz "
                    f"({values['x']:.3f}, {values['y']:.3f}, {values['z']:.3f}), "
                    f"RGB ({values['r']:.3f}, {values['g']:.3f}, {values['b']:.3f}), "
                    f"intensity {values['intensity']:.3f}, softness {values['diffuse']:.3f}"
                )
            return (
                "Relight this exact image by adding " + "; ".join(descriptions) + ". "
                "Preserve the camera, geometry, identity, materials, and composition."
            )
        color = tuple(self._value(sample, f"in_scene.{channel}") for channel in "rgb")
        intensity = self._value(sample, "in_scene.intensity")
        transition = self._value(sample, "in_scene.transition")
        return (
            "Turn on the visible in-scene light fixture and relight this exact image with "
            f"RGB ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}), intensity {intensity:.3f}, "
            f"transition {transition:.3f}. Preserve the camera, geometry, identity, materials, and composition."
        )


def flux_kontext_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        "condition_pixel_values": torch.stack([item["condition_pixel_values"] for item in samples]),
        "target_pixel_values": torch.stack([item["target_pixel_values"] for item in samples]),
        "prompt": [item["prompt"] for item in samples],
        "task_name": [item["task_name"] for item in samples],
        "scene_id": [item["scene_id"] for item in samples],
        "sample_index": [item["sample_index"] for item in samples],
        "sample_seed": [item["sample_seed"] for item in samples],
    }
