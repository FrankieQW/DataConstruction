from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL


def precision_dtype(name: str) -> torch.dtype:
    try:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
    except KeyError as error:
        raise ValueError(f"不支持的 precision: {name}") from error


def initialize_runtime(config: dict[str, Any], distributed_training: bool = False) -> torch.device:
    runtime = config["runtime"]
    if runtime["device"] != "cuda":
        raise ValueError("Next-DiT 当前只支持 runtime.device=cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，TokenLight Next-DiT 无法初始化")
    gpu_ids = [int(gpu_id) for gpu_id in runtime["gpu_ids"]]
    if int(runtime["model_parallel_size"]) != 1:
        raise NotImplementedError("TokenLight DDP 训练要求 runtime.model_parallel_size=1")
    if distributed_training and len(gpu_ids) > 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != len(gpu_ids) or "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "多卡训练必须通过 train.sh 或 torchrun 启动，且 WORLD_SIZE 必须等于 runtime.gpu_ids 数量"
            )
        device_index = int(os.environ["LOCAL_RANK"])
    else:
        if distributed_training and int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("单卡配置不能在多进程 torchrun 环境中启动")
        device_index = int(gpu_ids[0])
    torch.cuda.set_device(device_index)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("nccl")
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)
    return torch.device("cuda", device_index)


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def load_frozen_vae(config: dict[str, Any], device: torch.device) -> AutoencoderKL:
    path = Path(config["paths"]["vae"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"VAE 路径不存在: {path}")
    vae = AutoencoderKL.from_pretrained(str(path), torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def encode_images(
    vae: AutoencoderKL,
    images: torch.Tensor,
    scale: float,
    shift: float,
    sample: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        distribution = vae.encode(images.float()).latent_dist
        latent = distribution.sample() if sample else distribution.mode()
    return (latent - float(shift)) * float(scale)


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    with torch.no_grad():
        images = vae.decode(latents.float() / float(scale) + float(shift)).sample
    return images


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
