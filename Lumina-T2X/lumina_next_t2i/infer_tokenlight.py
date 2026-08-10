from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tokenlight.checkpoint import load_tokenlight_weights
from tokenlight.config import load_config
from tokenlight.inference_io import (
    build_packed_lighting,
    load_fixture_mask,
    load_source_image,
    save_image,
    write_inference_metadata,
)
from tokenlight.model import build_tokenlight_model
from tokenlight.runtime import decode_latents, encode_images, initialize_runtime, load_frozen_vae, precision_dtype
from tokenlight.sampler import euler_sample
from tokenlight.tokens import LightingSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TokenLight-Lumina relighting inference. Persistent defaults come only from config.yaml."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True, help="Source RGB image, linear EXR, or NPY")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument(
        "--task", required=True,
        choices=("ambient_scale", "global_diffuse", "add_light", "in_scene_light"),
    )
    parser.add_argument("--ambient-scale", type=float)
    parser.add_argument("--global-diffuse", type=float, help="Target minus source diffuse level")
    parser.add_argument(
        "--add-light", action="append",
        help="Repeat for multiple lights: x,y,z,r,g,b,intensity,diffuse",
    )
    parser.add_argument("--fixture-mask")
    parser.add_argument("--fixture-r", type=float)
    parser.add_argument("--fixture-g", type=float)
    parser.add_argument("--fixture-b", type=float)
    parser.add_argument("--fixture-intensity", type=float)
    parser.add_argument("--fixture-transition", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = config["paths"].get("inference_checkpoint")
    if not checkpoint:
        raise ValueError("请先在 config.yaml 中设置 paths.inference_checkpoint")
    schema = LightingSchema(int(config["model"]["max_lights"]))
    packed = build_packed_lighting(args, schema)

    device = initialize_runtime(config)
    dtype = precision_dtype(config["infer"]["precision"])
    model = build_tokenlight_model(config)
    report = load_tokenlight_weights(model, checkpoint)
    print(f"加载 TokenLight 权重: {report.source}", flush=True)
    model.to(device=device, dtype=dtype).eval()
    vae = load_frozen_vae(config, device)

    resolution = int(config["data"]["resolution"])
    source_image = load_source_image(args.source, resolution, config["data"]["exposure"])[None].to(device)
    fixture_mask = load_fixture_mask(args.fixture_mask, resolution)[None].to(device)
    fixture_present = torch.tensor([args.task == "in_scene_light"], dtype=torch.bool, device=device)
    values = torch.from_numpy(packed.values)[None].to(device)
    known = torch.from_numpy(packed.known)[None].to(device)
    valid = torch.from_numpy(packed.valid)[None].to(device)

    source_latent = encode_images(
        vae, source_image, config["infer"]["vae_scale"], config["infer"]["vae_shift"], sample=False
    ).to(dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["infer"]["seed"]))
    noise = torch.randn(source_latent.shape, generator=generator, device=device, dtype=dtype)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
        result_latent = euler_sample(
            model, noise, source_latent, values, known, valid, fixture_mask, fixture_present,
            steps=int(config["infer"]["steps"]), cfg_scale=float(config["infer"]["cfg_scale"]),
        )
    result = decode_latents(
        vae, result_latent, config["infer"]["vae_scale"], config["infer"]["vae_shift"]
    )
    save_image(result[0], args.output)
    if config["infer"]["save_metadata"]:
        metadata = {
            "config": str(Path(args.config).expanduser().resolve()),
            "config_sha256": hashlib.sha256(Path(args.config).expanduser().read_bytes()).hexdigest(),
            "checkpoint": report.source,
            "source": str(Path(args.source).expanduser().resolve()),
            "output": str(Path(args.output).expanduser().resolve()),
            "task": args.task,
            "lighting_token_names": list(schema.names),
            "lighting_values": packed.values.tolist(),
            "lighting_known": packed.known.tolist(),
            "lighting_valid": packed.valid.tolist(),
            "fixture_mask": str(Path(args.fixture_mask).expanduser().resolve()) if args.fixture_mask else None,
            "seed": int(config["infer"]["seed"]),
            "steps": int(config["infer"]["steps"]),
            "cfg_scale": float(config["infer"]["cfg_scale"]),
            "solver": config["infer"]["solver"],
        }
        write_inference_metadata(args.output, metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
