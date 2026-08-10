from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .tokens import LightingSchema, PackedLighting


def load_source_image(path: str | Path, resolution: int, exposure: float) -> torch.Tensor:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"source image 不存在: {path}")
    if path.suffix.lower() in {".exr", ".npy"}:
        image = _read_linear(path)
        image = np.maximum(image * float(exposure), 0.0)
        image = image / (1.0 + image)
    else:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    image = _center_resize(image, resolution, is_mask=False)
    return torch.from_numpy(np.ascontiguousarray(image * 2.0 - 1.0)).permute(2, 0, 1)


def load_fixture_mask(path: str | Path | None, resolution: int) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, resolution, resolution, dtype=torch.float32)
    mask_path = Path(path).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(f"fixture mask 不存在: {mask_path}")
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path).astype(np.float32)
    else:
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = _center_resize(mask, resolution, is_mask=True)
    return torch.from_numpy(np.ascontiguousarray(mask))[None]


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().clamp(-1, 1).add(1).mul(0.5)
    image = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.uint8(np.round(image * 255.0))).save(output_path)


def parse_add_light(value: str) -> dict[str, float]:
    fields = ("x", "y", "z", "r", "g", "b", "intensity", "diffuse")
    try:
        numbers = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"--add-light 包含非数值内容: {value}") from error
    if len(numbers) != len(fields):
        raise ValueError("--add-light 必须是 8 个逗号分隔数值: x,y,z,r,g,b,intensity,diffuse")
    return dict(zip(fields, numbers))


def build_packed_lighting(args: Any, schema: LightingSchema) -> PackedLighting:
    if args.task == "ambient_scale":
        if args.ambient_scale is None:
            raise ValueError("ambient_scale 任务必须提供 --ambient-scale")
        return schema.ambient(args.ambient_scale)
    if args.task == "global_diffuse":
        if args.global_diffuse is None:
            raise ValueError("global_diffuse 任务必须提供 --global-diffuse")
        return schema.global_diffuse(args.global_diffuse)
    if args.task == "add_light":
        if not args.add_light:
            raise ValueError("add_light 任务至少提供一次 --add-light")
        lights = [parse_add_light(value) for value in args.add_light]
        return schema.add_lights(lights)
    required = {
        "fixture_r": args.fixture_r,
        "fixture_g": args.fixture_g,
        "fixture_b": args.fixture_b,
        "fixture_intensity": args.fixture_intensity,
        "fixture_transition": args.fixture_transition,
    }
    missing = [name.replace("_", "-") for name, value in required.items() if value is None]
    if not args.fixture_mask or missing:
        raise ValueError(f"in_scene_light 必须提供 --fixture-mask 及全部 fixture 参数；缺少: {missing}")
    return schema.in_scene(
        {
            "r": args.fixture_r,
            "g": args.fixture_g,
            "b": args.fixture_b,
            "intensity": args.fixture_intensity,
            "transition": args.fixture_transition,
        }
    )


def write_inference_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    metadata_path = Path(path).expanduser().with_suffix(Path(path).suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_linear(path: Path) -> np.ndarray:
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
    return image[..., :3]


def _center_resize(array: np.ndarray, resolution: int, is_mask: bool) -> np.ndarray:
    import cv2

    height, width = array.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    array = array[top : top + side, left : left + side]
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if array.shape[:2] != (resolution, resolution):
        array = cv2.resize(array, (resolution, resolution), interpolation=interpolation)
    return np.clip(array, 0.0, 1.0).astype(np.float32)

