#!/usr/bin/env python3
"""Run a local FLUX.1-Kontext-dev TokenLight LoRA on one source image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
from PIL import Image, ImageOps
import torch

from tokenlight.flux_kontext_dataset import load_flux_kontext_config
from train_flux_kontext_lora import validate_local_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora", required=True, help="checkpoint-N or final LoRA directory")
    parser.add_argument("--source", required=True, help="TokenLight EXR or ordinary RGB image")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt",
        help="Exact edit instruction. If omitted, --ambient-scale is required and builds the training-style prompt.",
    )
    parser.add_argument("--ambient-scale", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_source(path: Path, resolution: int, exposure: float) -> Image.Image:
    if path.suffix.lower() != ".exr":
        with Image.open(path) as image:
            return ImageOps.fit(image.convert("RGB"), (resolution, resolution), Image.Resampling.LANCZOS)

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read EXR: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    linear = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    if not np.isfinite(linear).all():
        raise ValueError(f"EXR contains NaN/Inf: {path}")
    height, width = linear.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    linear = np.maximum(linear[top : top + side, left : left + side] * exposure, 0.0)
    mapped = linear / (1.0 + linear)
    mapped = cv2.resize(mapped, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return Image.fromarray(np.clip(mapped * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    args = parse_args()
    config = load_flux_kontext_config(args.config)
    model_path = validate_local_model(config["paths"]["pretrained_model"], require_exists=True)
    lora_path = Path(args.lora).expanduser()
    if not lora_path.is_dir():
        raise FileNotFoundError(lora_path)
    source_path = Path(args.source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    prompt = args.prompt
    if not prompt:
        if args.ambient_scale is None:
            raise ValueError("provide either --prompt or --ambient-scale")
        prompt = (
            f"Relight this exact image by setting the ambient illumination to {args.ambient_scale:.3f} times "
            "its current level. Preserve the camera, geometry, identity, materials, and composition."
        )

    from diffusers import FluxKontextPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = FluxKontextPipeline.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True
    )
    pipeline.load_lora_weights(lora_path, local_files_only=True)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    resolution = int(config["data"]["resolution"])
    # This is deliberately the same linear-EXR -> Reinhard path used by TokenLightDataset.
    tokenlight_config_path = Path(config["paths"]["tokenlight_config"]).expanduser()
    import yaml

    with tokenlight_config_path.open("r", encoding="utf-8") as handle:
        tokenlight_config = yaml.safe_load(handle)
    exposure = float(tokenlight_config["data"]["exposure"])
    condition = load_source(source_path, resolution, exposure)
    seed = int(args.seed if args.seed is not None else config["infer"]["seed"])
    steps = int(args.steps if args.steps is not None else config["infer"]["steps"])
    guidance = float(
        args.guidance_scale if args.guidance_scale is not None else config["infer"]["guidance_scale"]
    )
    generator = torch.Generator(device=pipeline._execution_device).manual_seed(seed)
    result = pipeline(
        image=condition,
        prompt=prompt,
        width=resolution,
        height=resolution,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    metadata = {
        "base_model": str(model_path),
        "lora": str(lora_path),
        "source": str(source_path),
        "output": str(output),
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance,
        "resolution": resolution,
        "exposure": exposure,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
