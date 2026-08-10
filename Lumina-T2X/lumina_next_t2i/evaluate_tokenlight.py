from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenlight.checkpoint import load_tokenlight_weights
from tokenlight.collate import tokenlight_collate
from tokenlight.config import load_config
from tokenlight.dataset import TokenLightDataset
from tokenlight.inference_io import save_image
from tokenlight.metrics import psnr, ssim
from tokenlight.model import build_tokenlight_model
from tokenlight.runtime import decode_latents, encode_images, initialize_runtime, load_frozen_vae, precision_dtype
from tokenlight.sampler import euler_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TokenLight using config.yaml.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    config = load_config(parse_args().config)
    checkpoint = config["paths"].get("inference_checkpoint")
    if not checkpoint:
        raise ValueError("请先设置 paths.inference_checkpoint")
    evaluate = config["evaluate"]
    split = evaluate["split"]
    if split not in {"validation", "test"}:
        raise ValueError("evaluate.split 必须是 validation 或 test")

    device = initialize_runtime(config)
    dtype = precision_dtype(config["infer"]["precision"])
    model = build_tokenlight_model(config)
    report = load_tokenlight_weights(model, checkpoint)
    model.to(device=device, dtype=dtype).eval()
    vae = load_frozen_vae(config, device)
    dataset = TokenLightDataset(config, split)
    loader = DataLoader(
        dataset,
        batch_size=int(evaluate["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=tokenlight_collate,
    )
    output_directory = Path(config["paths"]["output_root"]) / config["logging"]["run_id"] / evaluate["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    metric_names = set(evaluate["metrics"])
    unknown = metric_names - {"psnr", "ssim", "lpips"}
    if unknown:
        raise ValueError(f"不支持的 evaluate.metrics: {sorted(unknown)}")
    lpips_model = load_lpips(evaluate["lpips_network"], device) if "lpips" in metric_names else None
    rows: list[dict[str, object]] = []
    max_samples = evaluate.get("max_samples")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(evaluate["seed"]))

    for batch in loader:
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        batch = move_batch(batch, device)
        source_latent = encode_images(
            vae, batch["source_image"], config["infer"]["vae_scale"], config["infer"]["vae_shift"], sample=False
        ).to(dtype)
        noise = torch.randn(source_latent.shape, generator=generator, device=device, dtype=dtype)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
            prediction_latent = euler_sample(
                model, noise, source_latent,
                batch["lighting_values"], batch["lighting_known"], batch["lighting_valid"],
                batch["fixture_mask"], batch["fixture_present"],
                int(config["infer"]["steps"]), float(config["infer"]["cfg_scale"]),
            )
        prediction = decode_latents(
            vae, prediction_latent, config["infer"]["vae_scale"], config["infer"]["vae_shift"]
        ).clamp(-1, 1)
        target = batch["target_image"].clamp(-1, 1)
        prediction_unit = (prediction + 1.0) * 0.5
        target_unit = (target + 1.0) * 0.5
        values: dict[str, torch.Tensor] = {}
        if "psnr" in metric_names:
            values["psnr"] = psnr(prediction_unit, target_unit)
        if "ssim" in metric_names:
            values["ssim"] = ssim(prediction_unit, target_unit)
        if lpips_model is not None:
            values["lpips"] = lpips_model(prediction, target).flatten()
        for index in range(prediction.shape[0]):
            if max_samples is not None and len(rows) >= int(max_samples):
                break
            row: dict[str, object] = {
                "index": len(rows),
                "scene_id": batch["scene_id"][index],
                "asset_uid": batch["asset_uid"][index],
                "task": batch["task_name"][index],
            }
            row.update({name: float(value[index]) for name, value in values.items()})
            rows.append(row)
            if evaluate["save_images"]:
                save_image(prediction[index], output_directory / "images" / f"{len(rows) - 1:06d}.png")

    if not rows:
        raise RuntimeError("评测没有产生样本")
    write_csv(output_directory / "metrics.csv", rows)
    summary = summarize(rows, metric_names)
    summary.update({"checkpoint": report.source, "split": split, "samples": len(rows)})
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def load_lpips(network: str, device: torch.device):
    try:
        import lpips
    except ImportError as error:
        raise ImportError("启用 LPIPS 前请安装 lpips；或从 evaluate.metrics 移除 lpips") from error
    return lpips.LPIPS(net=network).to(device).eval()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], metrics: set[str]) -> dict[str, object]:
    tasks = sorted({str(row["task"]) for row in rows})
    summary: dict[str, object] = {}
    for metric in sorted(metrics):
        summary[metric] = sum(float(row[metric]) for row in rows) / len(rows)
        summary[f"{metric}_by_task"] = {
            task: sum(float(row[metric]) for row in rows if row["task"] == task)
            / sum(1 for row in rows if row["task"] == task)
            for task in tasks
        }
    return summary


if __name__ == "__main__":
    main()

