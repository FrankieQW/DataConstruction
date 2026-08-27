#!/usr/bin/env python3
"""LoRA fine-tune FLUX.1-Kontext-dev on deterministic TokenLight pairs.

This entry point intentionally lives beside, rather than inside, the existing
Lumina/TokenLight trainer. It only loads local model files and never downloads
weights from the network.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.utils.data import DataLoader

from tokenlight.flux_kontext_dataset import (
    FluxKontextTokenLightDataset,
    flux_kontext_collate,
    load_flux_kontext_config,
)


REQUIRED_MODEL_ENTRIES = (
    "model_index.json",
    "scheduler",
    "transformer",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate YAML and print the local model contract without importing Diffusers.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Construct datasets and read their first deterministic sample, then exit.",
    )
    return parser.parse_args()


def validate_local_model(path_value: str | Path, require_exists: bool) -> Path:
    model_path = Path(path_value).expanduser()
    if not model_path.exists():
        if require_exists:
            raise FileNotFoundError(
                f"FLUX.1-Kontext-dev snapshot is missing: {model_path}. "
                "Download the complete Diffusers snapshot to this exact directory."
            )
        return model_path
    missing = [name for name in REQUIRED_MODEL_ENTRIES if not (model_path / name).exists()]
    if missing:
        raise RuntimeError(f"incomplete FLUX Diffusers snapshot at {model_path}; missing: {missing}")
    return model_path


def print_contract(config: dict[str, Any]) -> None:
    model_path = validate_local_model(config["paths"]["pretrained_model"], require_exists=False)
    contract = {
        "config": config["_config_path"],
        "pretrained_model": str(model_path),
        "required_model_entries": list(REQUIRED_MODEL_ENTRIES),
        "tokenlight_config": config["paths"]["tokenlight_config"],
        "resolution": int(config["data"]["resolution"]),
        "lora_rank": int(config["lora"]["rank"]),
        "lora_targets": list(config["lora"]["target_modules"]),
        "network_access": "disabled by local_files_only=True",
    }
    print(json.dumps(contract, ensure_ascii=False, indent=2))


def dtype_for(mixed_precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[mixed_precision]


def encode_vae(
    vae: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    encode_mode: str,
) -> torch.Tensor:
    images = images.to(device=vae.device, dtype=dtype)
    posterior = vae.encode(images).latent_dist
    if encode_mode == "mode":
        latents = posterior.mode()
    elif encode_mode == "sample":
        latents = posterior.sample()
    else:
        raise ValueError(f"unsupported train.vae_encode_mode: {encode_mode}")
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    return (latents - shift) * scale


def pack_latents(pipeline_class: type, latents: torch.Tensor) -> torch.Tensor:
    return pipeline_class._pack_latents(
        latents,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )


def make_image_ids(
    pipeline_class: type,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    condition: bool,
) -> torch.Tensor:
    image_ids = pipeline_class._prepare_latent_image_ids(
        latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2, device, dtype
    )
    if condition:
        image_ids = image_ids.clone()
        image_ids[..., 0] = 1
    return image_ids


def resolve_resume(config: dict[str, Any], output_dir: Path) -> Path | None:
    value = config["paths"].get("resume_checkpoint")
    if not value:
        return None
    if str(value) != "latest":
        path = Path(value).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None))[1]


def prune_checkpoints(output_dir: Path, limit: int) -> None:
    if limit < 1:
        return
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            pass
    for _, path in sorted(checkpoints)[: max(0, len(checkpoints) - limit)]:
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    config = load_flux_kontext_config(args.config)
    if args.check_config:
        print_contract(config)
        return

    if args.check_data:
        for split in ("train", "validation"):
            dataset = FluxKontextTokenLightDataset(config, split)
            sample = dataset[0]
            print(
                json.dumps(
                    {
                        "split": split,
                        "length": len(dataset),
                        "condition_shape": list(sample["condition_pixel_values"].shape),
                        "target_shape": list(sample["target_pixel_values"].shape),
                        "task": sample["task_name"],
                        "prompt": sample["prompt"],
                        "range": [
                            float(sample["condition_pixel_values"].min()),
                            float(sample["condition_pixel_values"].max()),
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return

    model_path = validate_local_model(config["paths"]["pretrained_model"], require_exists=True)

    # Heavy training-only dependencies are imported here so --check-config works
    # in the original Lumina environment without changing it.
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import FluxKontextPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

    train_config = config["train"]
    mixed_precision = str(train_config["mixed_precision"])
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        mixed_precision=None if mixed_precision == "no" else mixed_precision,
        log_with="tensorboard",
        project_dir=str(Path(config["paths"]["output_root"]).expanduser()),
    )
    set_seed(int(train_config["seed"]), device_specific=True)
    weight_dtype = dtype_for(mixed_precision)
    output_dir = Path(config["paths"]["output_root"]).expanduser() / str(train_config["run_name"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config["_config_path"], output_dir / "config.yaml")
    accelerator.wait_for_everyone()

    pipeline = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    for module in (transformer, vae, text_encoder, text_encoder_2):
        module.requires_grad_(False)
    if bool(train_config.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()

    lora_config = config["lora"]
    transformer.add_adapter(
        LoraConfig(
            r=int(lora_config["rank"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            init_lora_weights="gaussian",
            target_modules=list(lora_config["target_modules"]),
        )
    )
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced zero trainable parameters")
    if mixed_precision == "fp16":
        for parameter in trainable:
            parameter.data = parameter.data.float()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_config["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(train_config["weight_decay"]),
        eps=1.0e-8,
    )
    dataset = FluxKontextTokenLightDataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["micro_batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=True,
        collate_fn=flux_kontext_collate,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(train_config["seed"])),
    )

    def save_model_hook(models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], save_dir: str) -> None:
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(models[0])
            state = get_peft_model_state_dict(unwrapped)
            FluxKontextPipeline.save_lora_weights(
                save_dir, transformer_lora_layers=state, safe_serialization=True
            )
        while weights:
            weights.pop()

    def load_model_hook(models: list[torch.nn.Module], load_dir: str) -> None:
        unwrapped = accelerator.unwrap_model(models.pop())
        state = FluxKontextPipeline.lora_state_dict(load_dir, local_files_only=True)
        state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        state = convert_unet_state_dict_to_peft(state)
        incompatible = set_peft_model_state_dict(unwrapped, state, adapter_name="default")
        if getattr(incompatible, "unexpected_keys", None):
            raise RuntimeError(f"unexpected LoRA keys while resuming: {incompatible.unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    pipeline.vae = vae
    pipeline.text_encoder = text_encoder
    pipeline.text_encoder_2 = text_encoder_2

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    num_train_timesteps = int(getattr(noise_scheduler.config, "num_train_timesteps", 1000))
    noise_scheduler.set_timesteps(num_train_timesteps, device=accelerator.device)

    global_step = 0
    resume_path = resolve_resume(config, output_dir)
    if resume_path is not None:
        accelerator.load_state(str(resume_path))
        global_step = int(resume_path.name.rsplit("-", 1)[1])

    max_steps = int(train_config["max_steps"])
    checkpoint_every = int(train_config["checkpointing_steps"])
    max_grad_norm = float(train_config["max_grad_norm"])
    guidance_scale = float(train_config.get("guidance_scale", 1.0))
    max_sequence_length = int(train_config.get("max_sequence_length", 256))
    accelerator.init_trackers(str(train_config["run_name"]))
    transformer.train()

    while global_step < max_steps:
        for batch in loader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    vae_encode_mode = str(train_config.get("vae_encode_mode", "mode"))
                    target_latents = encode_vae(
                        vae, batch["target_pixel_values"], weight_dtype, vae_encode_mode
                    )
                    condition_latents = encode_vae(
                        vae, batch["condition_pixel_values"], weight_dtype, vae_encode_mode
                    )
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                        prompt=batch["prompt"],
                        prompt_2=None,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        max_sequence_length=max_sequence_length,
                    )

                target_ids = make_image_ids(
                    FluxKontextPipeline, target_latents, accelerator.device, weight_dtype, condition=False
                )
                condition_ids = make_image_ids(
                    FluxKontextPipeline, condition_latents, accelerator.device, weight_dtype, condition=True
                )
                target_packed = pack_latents(FluxKontextPipeline, target_latents)
                condition_packed = pack_latents(FluxKontextPipeline, condition_latents)
                noise = torch.randn_like(target_packed)
                indices = torch.randint(
                    0, num_train_timesteps, (target_packed.shape[0],), device=accelerator.device
                )
                timesteps = noise_scheduler.timesteps[indices].to(dtype=weight_dtype)
                sigmas = noise_scheduler.sigmas[indices].to(
                    device=accelerator.device, dtype=target_packed.dtype
                ).view(-1, *([1] * (target_packed.ndim - 1)))
                noisy_target = (1.0 - sigmas) * target_packed + sigmas * noise
                model_input = torch.cat((noisy_target, condition_packed), dim=1)
                image_ids = torch.cat((target_ids, condition_ids), dim=-2)
                guidance = None
                raw_transformer = accelerator.unwrap_model(transformer)
                if bool(getattr(raw_transformer.config, "guidance_embeds", False)):
                    guidance = torch.full(
                        (model_input.shape[0],), guidance_scale,
                        device=accelerator.device, dtype=weight_dtype,
                    )

                autocast = (
                    torch.autocast("cuda", dtype=weight_dtype)
                    if accelerator.device.type == "cuda" and mixed_precision != "no"
                    else nullcontext()
                )
                with autocast:
                    prediction = transformer(
                        hidden_states=model_input,
                        timestep=timesteps / 1000.0,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=image_ids,
                        return_dict=False,
                    )[0]
                    prediction = prediction[:, : target_packed.shape[1]]
                    flow_target = noise - target_packed
                    loss = torch.mean((prediction.float() - flow_target.float()) ** 2)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                reduced_loss = accelerator.gather(loss.detach().repeat(model_input.shape[0])).mean().item()
                accelerator.log({"train/loss": reduced_loss}, step=global_step)
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step={global_step}/{max_steps} loss={reduced_loss:.6f}", flush=True)
                if global_step % checkpoint_every == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(checkpoint_dir))
                    if accelerator.is_main_process:
                        prune_checkpoints(output_dir, int(train_config.get("checkpoints_total_limit", 0)))
                if global_step >= max_steps:
                    break

    accelerator.wait_for_everyone()
    final_dir = output_dir / "final"
    accelerator.save_state(str(final_dir))
    if accelerator.is_main_process:
        (final_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "base_model": str(model_path),
                    "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                    "world_size": accelerator.num_processes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
