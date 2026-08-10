from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
from safetensors.torch import load_file, save_file
import torch


@dataclass(frozen=True)
class LoadReport:
    source: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


NEW_PARAMETER_PREFIXES = ("lighting_encoder.", "fixture_mask_embedder.")


def load_upstream_weights(model: torch.nn.Module, checkpoint: str | Path) -> LoadReport:
    source = _find_weight_file(Path(checkpoint).expanduser(), upstream=True)
    state = _load_state_dict(source)
    result = model.load_state_dict(state, strict=False)
    missing = tuple(result.missing_keys)
    unexpected = tuple(result.unexpected_keys)
    invalid_missing = [key for key in missing if not key.startswith(NEW_PARAMETER_PREFIXES)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Lumina 权重与 TokenLight backbone 不兼容。"
            f"\n不允许缺失的 keys: {invalid_missing}\nunexpected keys: {list(unexpected)}"
        )
    return LoadReport(str(source), missing, unexpected)


def load_tokenlight_weights(model: torch.nn.Module, checkpoint: str | Path) -> LoadReport:
    source = _find_weight_file(Path(checkpoint).expanduser(), upstream=False)
    state = _load_state_dict(source)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"TokenLight checkpoint 必须严格匹配。missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return LoadReport(str(source), (), ())


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    global_step: int,
    samples_seen: int,
    dataloader_state: dict[str, Any],
    run_directory: Path,
    rng_states: list[dict[str, Any]] | None = None,
) -> Path:
    checkpoint_dir = run_directory / "checkpoints" / f"step_{global_step:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(model_state, str(checkpoint_dir / "model.safetensors"))
    if rng_states is None:
        rng_states = [capture_rng_state()]
    training_state = {
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "samples_seen": int(samples_seen),
        "dataloader_state": dataloader_state,
        "rng_states": rng_states,
    }
    torch.save(training_state, checkpoint_dir / "training_state.pth")
    shutil.copyfile(config["_config_path"], checkpoint_dir / "config.yaml")
    return checkpoint_dir


def restore_training_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: str | Path,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint).expanduser()
    state_path = checkpoint_dir / "training_state.pth" if checkpoint_dir.is_dir() else checkpoint_dir.parent / "training_state.pth"
    if not state_path.is_file():
        raise FileNotFoundError(f"断点训练状态不存在: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if "rng_states" in state:
        rng_states = state["rng_states"]
        if len(rng_states) != world_size:
            raise ValueError(
                f"checkpoint world size 为 {len(rng_states)}，当前为 {world_size}，无法精确续训"
            )
        restore_rng_state(rng_states[rank])
    else:
        if world_size != 1:
            raise ValueError("旧版单卡 checkpoint 不包含多 rank 随机状态，无法用于多卡精确续训")
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    if "dataloader_state" not in state:
        raise ValueError(
            "checkpoint 缺少 dataloader_state，无法保证精确续训；请从新版 checkpoint 开始训练"
        )
    return {
        "global_step": int(state["global_step"]),
        "samples_seen": int(state["samples_seen"]),
        "dataloader_state": state["dataloader_state"],
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def append_checkpoint_index(
    index_path: Path,
    global_step: int,
    checkpoint: Path,
    validation_loss: float | None,
    is_best: bool,
    created_at: str,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "global_step": int(global_step),
        "checkpoint": str(checkpoint),
        "val_loss_total": validation_loss,
        "is_best": bool(is_best),
        "created_at": created_at,
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _find_weight_file(path: Path, upstream: bool) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint 路径不存在: {path}")
    names = ["model.safetensors"]
    if upstream:
        names.extend(["consolidated.00-of-01.safetensors", "consolidated.00-of-01.pth", "consolidated.pth"])
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"在 {path} 中找不到可加载的模型权重，检查过: {names}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        state: Any = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"checkpoint 不是有效 state_dict: {path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state
