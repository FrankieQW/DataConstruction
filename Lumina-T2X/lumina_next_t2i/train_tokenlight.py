from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from tokenlight.checkpoint import (
    append_checkpoint_index,
    capture_rng_state,
    load_tokenlight_weights,
    load_upstream_weights,
    restore_training_state,
    save_training_checkpoint,
)
from tokenlight.collate import tokenlight_collate
from tokenlight.config import load_config
from tokenlight.dataset import TokenLightDataset
from tokenlight.flow import sample_linear_path, velocity_mse
from tokenlight.logging_utils import JsonlLogger
from tokenlight.model import build_tokenlight_model
from tokenlight.runtime import (
    encode_images,
    get_rank,
    get_world_size,
    initialize_runtime,
    is_main_process,
    load_frozen_vae,
    precision_dtype,
)
from tokenlight.sampler_state import ResumableRandomSampler
from tokenlight.training_smoke import (
    build_smoke_gate,
    finalize_smoke_gate,
    inspect_smoke_batch,
    optimizer_step_marker,
    record_smoke_checkpoint,
    record_smoke_step,
    require_finite_smoke_tensor,
)
from tokenlight.tokens import TASK_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TokenLight-Lumina from the single YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to lumina_next_t2i/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = initialize_runtime(config, distributed_training=True)
    rank = get_rank()
    world_size = get_world_size()
    main_process = is_main_process()
    smoke_enabled = bool(config["train"].get("smoke", {}).get("enabled", False))
    if smoke_enabled and world_size != 1:
        raise ValueError("training smoke must run with a single process/GPU for exact resume verification")
    seed_everything(int(config["train"]["seed"]) + rank)
    dtype = precision_dtype(config["train"]["precision"])

    run_directory = Path(config["paths"]["output_root"]).expanduser() / config["logging"]["run_id"]
    if main_process:
        run_directory.mkdir(parents=True, exist_ok=True)
        write_run_metadata(config, run_directory)
    if world_size > 1:
        dist.barrier()
    train_logger = JsonlLogger(run_directory / config["logging"]["train_jsonl"]) if main_process else None
    validation_logger = (
        JsonlLogger(run_directory / config["logging"]["validation_jsonl"]) if main_process else None
    )
    writer = (
        SummaryWriter(run_directory / "tensorboard")
        if main_process and config["logging"]["tensorboard"] else None
    )

    raw_model = build_tokenlight_model(config)
    resume_path = config["paths"].get("resume_checkpoint")
    if resume_path:
        report = load_tokenlight_weights(raw_model, resume_path)
    else:
        report = load_upstream_weights(raw_model, config["paths"]["upstream_checkpoint"])
    if main_process:
        print_load_report(report)
    raw_model.to(device=device, dtype=dtype)
    model: torch.nn.Module = raw_model
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
    model.train()

    train_config = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=tuple(float(value) for value in train_config["betas"]),
    )
    global_step = 0
    samples_seen = 0
    restored_dataloader_state = None
    vae = load_frozen_vae(config, device)
    if resume_path:
        restored = restore_training_state(optimizer, resume_path, rank=rank, world_size=world_size)
        global_step = restored["global_step"]
        samples_seen = restored["samples_seen"]
        restored_dataloader_state = restored["dataloader_state"]

    accumulation_steps = int(train_config["gradient_accumulation_steps"])
    max_steps = int(train_config["max_steps"])
    remaining_samples = (
        max(max_steps - global_step, 0)
        * accumulation_steps
        * int(train_config["micro_batch_size"])
        * world_size
    )
    train_loader, train_sampler = make_train_loader(config, samples_seen, remaining_samples)
    validate_dataloader_resume(restored_dataloader_state, train_sampler, samples_seen, config)
    smoke_gate = build_smoke_gate(
        config,
        run_directory,
        resume_path,
        {
            "global_step": global_step,
            "samples_seen": samples_seen,
            "dataloader_state": restored_dataloader_state,
        }
        if resume_path
        else None,
        train_sampler,
        global_step,
        samples_seen,
        resume_signature(config),
    )
    validation_loader = make_validation_loader(config)
    train_iterator = iter(train_loader)
    log_every = int(config["logging"]["log_every_steps"])
    start_time = time.monotonic()
    best_validation = float("inf")
    latest_validation: float | None = None
    progress = tqdm(
        total=max_steps,
        initial=global_step,
        desc="train",
        unit="step",
        dynamic_ncols=True,
        disable=not main_process,
    )

    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        task_loss_sum = {name: 0.0 for name in TASK_NAMES}
        task_count = {name: 0 for name in TASK_NAMES}
        optimizer_samples = 0

        for accumulation_index in range(accumulation_steps):
            batch = move_batch(next(train_iterator), device)
            inspect_smoke_batch(smoke_gate, batch)
            source_latent = encode_images(
                vae, batch["source_image"], train_config["vae_scale"], train_config["vae_shift"]
            ).to(dtype)
            target_latent = encode_images(
                vae, batch["target_image"], train_config["vae_scale"], train_config["vae_shift"]
            ).to(dtype)
            noisy_target, flow_time, velocity_target = sample_linear_path(
                target_latent, config["flow"]["time_min"], config["flow"]["time_max"]
            )
            drop_condition = torch.rand(target_latent.shape[0], device=device) < float(config["lighting"]["cfg_dropout"])
            synchronize = world_size == 1 or accumulation_index == accumulation_steps - 1
            sync_context = nullcontext() if synchronize else model.no_sync()
            with sync_context:
                with autocast_context(device, dtype):
                    prediction = model(
                        noisy_target,
                        flow_time,
                        source_latent,
                        batch["lighting_values"],
                        batch["lighting_known"],
                        batch["lighting_valid"],
                        fixture_mask=batch["fixture_mask"],
                        fixture_present=batch["fixture_present"],
                        drop_condition=drop_condition,
                    )
                    per_sample_loss = velocity_mse(prediction, velocity_target)
                    require_finite_smoke_tensor(smoke_gate, per_sample_loss, "per_sample_loss")
                    loss = per_sample_loss.mean() / accumulation_steps
                loss.backward()

            batch_size = target_latent.shape[0]
            optimizer_samples += batch_size
            loss_sum += float(per_sample_loss.detach().sum())
            for task_id, task_name in enumerate(TASK_NAMES):
                selected = batch["task"] == task_id
                if selected.any():
                    task_loss_sum[task_name] += float(per_sample_loss.detach()[selected].sum())
                    task_count[task_name] += int(selected.sum())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["gradient_clip_norm"]))
        optimizer_step_before = optimizer_step_marker(optimizer) if smoke_gate is not None else None
        optimizer.step()
        global_step += 1
        statistics = torch.tensor(
            [loss_sum, optimizer_samples]
            + [value for name in TASK_NAMES for value in (task_loss_sum[name], task_count[name])],
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        values = statistics.tolist()
        loss_sum, optimizer_samples = values[:2]
        for index, name in enumerate(TASK_NAMES):
            task_loss_sum[name] = values[2 + index * 2]
            task_count[name] = int(values[3 + index * 2])
        optimizer_samples = int(optimizer_samples)
        samples_seen += optimizer_samples
        step_loss = loss_sum / optimizer_samples
        record_smoke_step(
            smoke_gate,
            step_loss,
            float(grad_norm),
            task_loss_sum,
            task_count,
            optimizer,
            optimizer_step_before,
        )
        step_task_losses = {
            short_task_name(name): task_loss_sum[name] / task_count[name]
            for name in TASK_NAMES
            if task_count[name]
        }
        progress.update(1)
        progress.set_postfix(
            loss=f"{step_loss:.4f}",
            **{name: f"{value:.4f}" for name, value in step_task_losses.items()},
            val="-" if latest_validation is None else f"{latest_validation:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            mem=f"{torch.cuda.max_memory_allocated(device) / 1024**3:.1f}G",
        )

        if main_process and (global_step % log_every == 0 or global_step == 1):
            record: dict[str, Any] = {
                "run_id": config["logging"]["run_id"],
                "global_step": global_step,
                "epoch": samples_seen / max(len(train_loader.dataset), 1),
                "samples_seen": samples_seen,
                "wall_time_sec": time.monotonic() - start_time,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss_total": step_loss,
                "grad_norm": float(grad_norm),
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            }
            for name in TASK_NAMES:
                record[f"loss_{short_task_name(name)}"] = step_task_losses.get(short_task_name(name))
            assert train_logger is not None
            train_logger.write(record)
            if writer:
                writer.add_scalar("train/loss_total", record["loss_total"], global_step)
                writer.add_scalar("train/learning_rate", record["learning_rate"], global_step)
                for name in TASK_NAMES:
                    value = record[f"loss_{short_task_name(name)}"]
                    if value is not None:
                        writer.add_scalar(f"train/loss_{short_task_name(name)}", value, global_step)

        if global_step % int(train_config["validation_every_steps"]) == 0:
            validation_record = validate(model, vae, validation_loader, config, device, dtype, global_step)
            latest_validation = float(validation_record["val/loss_total"])
            if main_process:
                assert validation_logger is not None
                validation_logger.write(validation_record)
            progress.set_postfix(
                loss=f"{step_loss:.4f}",
                **{name: f"{value:.4f}" for name, value in step_task_losses.items()},
                val=f"{latest_validation:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                mem=f"{torch.cuda.max_memory_allocated(device) / 1024**3:.1f}G",
            )
            if writer:
                for key, value in validation_record.items():
                    if key.startswith("val/") and value is not None:
                        writer.add_scalar(key, value, global_step)

        if global_step % int(train_config["checkpoint_every_steps"]) == 0 or global_step == max_steps:
            rng_states = gather_rng_states()
            is_best = latest_validation is not None and latest_validation < best_validation
            if is_best:
                best_validation = latest_validation
            if main_process:
                checkpoint_path = save_training_checkpoint(
                    raw_model,
                    optimizer,
                    config,
                    global_step,
                    samples_seen,
                    current_dataloader_state(train_sampler, samples_seen, config),
                    run_directory,
                    rng_states=rng_states,
                )
                record_smoke_checkpoint(smoke_gate, checkpoint_path)
                append_checkpoint_index(
                    run_directory / config["logging"]["checkpoint_index_jsonl"],
                    global_step,
                    checkpoint_path,
                    latest_validation,
                    is_best,
                    datetime.now(timezone.utc).isoformat(),
                )
            if world_size > 1:
                dist.barrier()

    finalize_smoke_gate(smoke_gate, config, global_step, samples_seen)
    progress.close()
    if writer:
        writer.close()


@torch.no_grad()
def validate(model, vae, loader, config, device, dtype, global_step: int) -> dict[str, Any]:
    model.eval()
    total_sum = 0.0
    total_count = 0
    task_sum = {name: 0.0 for name in TASK_NAMES}
    task_count = {name: 0 for name in TASK_NAMES}
    generator_state = torch.cuda.get_rng_state(device)
    torch.cuda.manual_seed(int(config["data"]["validation_seed"]) + get_rank())
    validation_batches = int(config["train"]["validation_batches"])
    progress = tqdm(
        total=min(len(loader), validation_batches),
        desc=f"val@{global_step}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not is_main_process(),
    )
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= validation_batches:
                break
            batch = move_batch(batch, device)
            source_latent = encode_images(
                vae, batch["source_image"], config["train"]["vae_scale"], config["train"]["vae_shift"], sample=False
            ).to(dtype)
            target_latent = encode_images(
                vae, batch["target_image"], config["train"]["vae_scale"], config["train"]["vae_shift"], sample=False
            ).to(dtype)
            noisy_target, flow_time, target_velocity = sample_linear_path(
                target_latent, config["flow"]["time_min"], config["flow"]["time_max"]
            )
            with autocast_context(device, dtype):
                prediction = model(
                    noisy_target, flow_time, source_latent,
                    batch["lighting_values"], batch["lighting_known"], batch["lighting_valid"],
                    fixture_mask=batch["fixture_mask"], fixture_present=batch["fixture_present"],
                    drop_condition=torch.zeros(target_latent.shape[0], device=device, dtype=torch.bool),
                )
                losses = velocity_mse(prediction, target_velocity)
            total_sum += float(losses.sum())
            total_count += len(losses)
            progress.update(1)
            progress.set_postfix(loss=f"{total_sum / total_count:.4f}")
            for task_id, task_name in enumerate(TASK_NAMES):
                selected = batch["task"] == task_id
                if selected.any():
                    task_sum[task_name] += float(losses[selected].sum())
                    task_count[task_name] += int(selected.sum())
    finally:
        progress.close()
        torch.cuda.set_rng_state(generator_state, device)
        model.train()
    statistics = torch.tensor(
        [total_sum, total_count]
        + [value for name in TASK_NAMES for value in (task_sum[name], task_count[name])],
        dtype=torch.float64,
        device=device,
    )
    if get_world_size() > 1:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    values = statistics.tolist()
    total_sum, total_count = values[:2]
    total_count = int(total_count)
    for index, name in enumerate(TASK_NAMES):
        task_sum[name] = values[2 + index * 2]
        task_count[name] = int(values[3 + index * 2])
    if not total_count:
        raise RuntimeError("validation loader 未产生任何样本")
    record: dict[str, Any] = {"run_id": config["logging"]["run_id"], "global_step": global_step}
    record["val/loss_total"] = total_sum / total_count
    for name in TASK_NAMES:
        record[f"val/loss_{short_task_name(name)}"] = task_sum[name] / task_count[name] if task_count[name] else None
    return record


def make_train_loader(
    config: dict[str, Any], start_offset: int, sample_count: int
) -> tuple[DataLoader, ResumableRandomSampler]:
    dataset = TokenLightDataset(config, "train")
    sampler = ResumableRandomSampler(
        len(dataset),
        start_offset,
        sample_count,
        int(config["train"]["seed"]),
        rank=get_rank(),
        world_size=get_world_size(),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["micro_batch_size"]),
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        collate_fn=tokenlight_collate,
        drop_last=False,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )
    return loader, sampler


def make_validation_loader(config: dict[str, Any]) -> DataLoader:
    dataset = TokenLightDataset(config, "validation")
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )
        if get_world_size() > 1 else None
    )
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["micro_batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        collate_fn=tokenlight_collate,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )


def validate_dataloader_resume(
    restored: dict[str, Any] | None,
    sampler: ResumableRandomSampler,
    samples_seen: int,
    config: dict[str, Any],
) -> None:
    if restored is None:
        return
    if "world_size" not in restored:
        restored = {**restored, "world_size": 1}
    expected = current_dataloader_state(sampler, samples_seen, config)
    if restored != expected:
        raise ValueError(
            "checkpoint dataloader state 与当前数据或配置不一致。"
            f" checkpoint={restored}, current={expected}"
        )


def current_dataloader_state(
    sampler: ResumableRandomSampler,
    samples_seen: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    state: dict[str, Any] = sampler.state_dict(samples_seen)
    state["train_manifest_sha256"] = file_sha256(config["paths"]["train_manifest"])
    state["resume_signature"] = resume_signature(config)
    return state


def resume_signature(config: dict[str, Any]) -> str:
    signature = {
        "data": {
            key: config["data"][key]
            for key in (
                "resolution",
                "crop_mode",
                "exposure",
                "tone_mapping",
                "samples_per_scene_train",
                "tasks",
                "task_probabilities",
            )
        },
        "lighting": config["lighting"],
        "flow": config["flow"],
        "train": {
            key: config["train"][key]
            for key in (
                "seed",
                "precision",
                "micro_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "betas",
                "gradient_clip_norm",
                "vae_scale",
                "vae_shift",
            )
        },
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def short_task_name(name: str) -> str:
    return {"ambient_scale": "ambient", "global_diffuse": "diffuse", "add_light": "add_light", "in_scene_light": "in_scene"}[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gather_rng_states() -> list[dict[str, Any]] | None:
    local_state = capture_rng_state()
    if get_world_size() == 1:
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * get_world_size()
    dist.all_gather_object(states, local_state)
    if not is_main_process():
        return None
    return [state for state in states if state is not None]


def print_load_report(report) -> None:
    print(f"加载权重: {report.source}", flush=True)
    print(f"missing keys ({len(report.missing_keys)}): {list(report.missing_keys)}", flush=True)
    print(f"unexpected keys ({len(report.unexpected_keys)}): {list(report.unexpected_keys)}", flush=True)


def write_run_metadata(config: dict[str, Any], run_directory: Path) -> None:
    config_bytes = Path(config["_config_path"]).read_bytes()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config["_config_path"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "train_manifest_sha256": file_sha256(config["paths"]["train_manifest"]),
        "validation_manifest_sha256": file_sha256(config["paths"]["validation_manifest"]),
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "world_size": get_world_size(),
        "resumed_from_checkpoint": config["paths"].get("resume_checkpoint"),
        "parent_run_id": parent_run_id(config["paths"].get("resume_checkpoint")),
    }
    path = run_directory / "run_metadata.json"
    if not path.exists():
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parent_run_id(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    path = Path(checkpoint).expanduser()
    directory = path if path.is_dir() else path.parent
    return directory.parent.parent.name if directory.parent.name == "checkpoints" else None


if __name__ == "__main__":
    main()
