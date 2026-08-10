from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .tokens import TASK_NAMES


SUMMARY_VERSION = 1


def build_smoke_gate(
    config: dict[str, Any],
    run_directory: Path,
    resume_path: str | None,
    restored_state: dict[str, Any] | None,
    sampler,
    global_step: int,
    samples_seen: int,
    resume_signature: str,
) -> dict[str, Any] | None:
    settings = config["train"].get("smoke", {})
    if not bool(settings.get("enabled", False)):
        return None

    required_tasks = tuple(settings.get("required_tasks", config["data"]["tasks"]))
    summary_path = run_directory / str(settings.get("summary_json", "smoke_summary.json"))
    manifest_sha256 = _file_sha256(config["paths"]["train_manifest"])
    expected_next = next(iter(sampler), None)
    state: dict[str, Any] = {
        "enabled": True,
        "summary_path": summary_path,
        "required_tasks": required_tasks,
        "task_counts": {name: 0 for name in required_tasks},
        "task_loss_sums": {name: 0.0 for name in required_tasks},
        "lighting_batches": 0,
        "fixture_batches": 0,
        "steps_completed": 0,
        "initial_global_step": int(global_step),
        "initial_samples_seen": int(samples_seen),
        "expected_next_sample": list(expected_next) if expected_next is not None else None,
        "first_batch_checked": False,
        "manifest_sha256": manifest_sha256,
        "resume_signature": resume_signature,
        "resumed": resume_path is not None,
        "resume_verified": False,
        "resume_checkpoint": str(Path(resume_path).expanduser().resolve()) if resume_path else None,
        "checkpoint": None,
    }

    if resume_path is None:
        if summary_path.exists():
            raise RuntimeError(
                f"smoke summary already exists; use a new logging.run_id instead of overwriting it: {summary_path}"
            )
        return state

    if restored_state is None:
        raise RuntimeError("smoke resume requires restored training state")
    if not summary_path.is_file():
        raise RuntimeError(f"smoke resume summary does not exist: {summary_path}")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    if previous.get("status") != "awaiting-resume" or previous.get("train_ready") is not False:
        raise RuntimeError("smoke resume requires an awaiting-resume summary from the first phase")
    if previous.get("train_manifest_sha256") != manifest_sha256:
        raise RuntimeError("smoke resume manifest digest does not match the first phase")
    if previous.get("resume_signature") != resume_signature:
        raise RuntimeError("smoke resume signature does not match the first phase")
    expected_checkpoint = previous.get("checkpoint")
    actual_checkpoint = Path(resume_path).expanduser().resolve()
    if (
        not expected_checkpoint
        or Path(expected_checkpoint).expanduser().resolve() != actual_checkpoint
    ):
        raise RuntimeError("paths.resume_checkpoint is not the checkpoint recorded by the first smoke phase")
    if int(previous.get("final_global_step", -1)) != int(restored_state["global_step"]):
        raise RuntimeError("restored global_step does not match the first smoke phase")
    if int(previous.get("final_samples_seen", -1)) != int(restored_state["samples_seen"]):
        raise RuntimeError("restored samples_seen does not match the first smoke phase")

    for name in required_tasks:
        state["task_counts"][name] = int(previous.get("task_counts", {}).get(name, 0))
        average = previous.get("task_losses", {}).get(name)
        if average is not None:
            state["task_loss_sums"][name] = float(average) * state["task_counts"][name]
    state["lighting_batches"] = int(previous.get("lighting_batches", 0))
    state["fixture_batches"] = int(previous.get("fixture_batches", 0))
    state["resume_verified"] = True
    return state


def inspect_smoke_batch(state: dict[str, Any] | None, batch: dict[str, Any]) -> None:
    if state is None:
        return

    required_tensor_keys = (
        "lighting_values",
        "lighting_known",
        "lighting_valid",
        "fixture_mask",
        "fixture_present",
        "task",
    )
    missing = [
        key
        for key in required_tensor_keys
        if key not in batch or not isinstance(batch[key], torch.Tensor)
    ]
    if missing:
        smoke_fail(state, f"training batch is missing smoke tensors: {missing}")

    if not state["first_batch_checked"]:
        actual = None
        if batch.get("sample_index") and batch.get("sample_seed"):
            actual = [int(batch["sample_index"][0]), int(batch["sample_seed"][0])]
        if actual != state["expected_next_sample"]:
            mismatch = (
                f"expected={state['expected_next_sample']}, actual={actual}"
            )
            smoke_fail(
                state,
                f"first resumed sample does not match sampler state: {mismatch}",
            )
        state["first_batch_checked"] = True

    values = batch["lighting_values"]
    known = batch["lighting_known"].bool()
    valid = batch["lighting_valid"].bool()
    if not bool(torch.isfinite(values).all().item()):
        smoke_fail(state, "lighting_values contains NaN or Inf")
    if values.shape != known.shape or values.shape != valid.shape:
        smoke_fail(state, "lighting value/known/valid tensor shapes differ")
    if not bool((known & valid).any().item()):
        smoke_fail(state, "batch has no effective lighting token")

    task = batch["task"]
    fixture_present = batch["fixture_present"].bool()
    fixture_mask = batch["fixture_mask"]
    in_scene_id = TASK_NAMES.index("in_scene_light")
    in_scene = task == in_scene_id
    if not torch.equal(fixture_present, in_scene):
        smoke_fail(state, "fixture_present does not match in_scene_light task selection")
    if bool(in_scene.any().item()):
        selected_masks = fixture_mask[in_scene]
        nonempty = selected_masks.flatten(1).abs().sum(dim=1) > 0
        if not bool(nonempty.all().item()):
            smoke_fail(state, "in_scene_light batch contains an empty fixture mask")
        state["fixture_batches"] += 1
    if bool((~in_scene).any().item()):
        other_masks = fixture_mask[~in_scene]
        if bool((other_masks.abs() > 0).any().item()):
            smoke_fail(state, "non-in_scene_light batch contains a non-empty fixture mask")
    state["lighting_batches"] += 1


def require_finite_smoke_tensor(
    state: dict[str, Any] | None, value: torch.Tensor, name: str
) -> None:
    if state is not None and not bool(torch.isfinite(value).all().item()):
        smoke_fail(state, f"{name} contains NaN or Inf")


def record_smoke_step(
    state: dict[str, Any] | None,
    step_loss: float,
    grad_norm: float,
    task_loss_sums: dict[str, float],
    task_counts: dict[str, int],
    optimizer: torch.optim.Optimizer,
    optimizer_step_before: float | None = None,
) -> None:
    if state is None:
        return
    if not math.isfinite(step_loss) or not math.isfinite(grad_norm):
        smoke_fail(state, f"non-finite optimizer metrics: loss={step_loss}, grad_norm={grad_norm}")
    if not optimizer.state:
        smoke_fail(state, "optimizer state is empty after optimizer.step; no update was observed")
    if optimizer_step_before is not None:
        optimizer_step_after = optimizer_step_marker(optimizer)
        if optimizer_step_after <= optimizer_step_before:
            smoke_fail(
                state,
                "optimizer state step did not advance after optimizer.step",
            )
    for name in state["required_tasks"]:
        count = int(task_counts.get(name, 0))
        loss_sum = float(task_loss_sums.get(name, 0.0))
        if count and not math.isfinite(loss_sum / count):
            smoke_fail(state, f"task {name} produced a non-finite loss")
        state["task_counts"][name] += count
        state["task_loss_sums"][name] += loss_sum
    state["steps_completed"] += 1


def optimizer_step_marker(optimizer: torch.optim.Optimizer) -> float:
    """Return the greatest optimizer-state step, or -1 before state is initialized."""
    marker = -1.0
    for values in optimizer.state.values():
        step = values.get("step")
        if step is None:
            continue
        marker = max(marker, float(step.item() if isinstance(step, torch.Tensor) else step))
    return marker


def record_smoke_checkpoint(state: dict[str, Any] | None, checkpoint: Path) -> None:
    if state is not None:
        state["checkpoint"] = str(checkpoint.resolve())


def finalize_smoke_gate(
    state: dict[str, Any] | None,
    config: dict[str, Any],
    global_step: int,
    samples_seen: int,
) -> None:
    if state is None:
        return
    if state["steps_completed"] < 1:
        smoke_fail(state, "smoke phase completed no optimizer update")
    if not state["first_batch_checked"]:
        smoke_fail(state, "smoke phase consumed no training batch")
    if state["lighting_batches"] < 1:
        smoke_fail(state, "smoke phase observed no lighting-conditioned model input")
    missing_tasks = [name for name in state["required_tasks"] if state["task_counts"][name] < 1]
    if missing_tasks:
        smoke_fail(state, f"fixed smoke schedule did not cover required tasks: {missing_tasks}")
    if "in_scene_light" in state["required_tasks"] and state["fixture_batches"] < 1:
        smoke_fail(state, "smoke phase observed no in_scene_light fixture batch")
    if not state["checkpoint"]:
        smoke_fail(state, "smoke phase did not save a checkpoint")

    train_ready = bool(state["resumed"] and state["resume_verified"])
    status = "pass" if train_ready else "awaiting-resume"
    task_losses = {
        name: state["task_loss_sums"][name] / state["task_counts"][name]
        for name in state["required_tasks"]
    }
    summary = _base_summary(state)
    summary.update(
        {
            "status": status,
            "train_ready": train_ready,
            "run_id": config["logging"]["run_id"],
            "config_sha256": _file_sha256(config["_config_path"]),
            "final_global_step": int(global_step),
            "final_samples_seen": int(samples_seen),
            "task_counts": state["task_counts"],
            "task_losses": task_losses,
            "lighting_batches": state["lighting_batches"],
            "fixture_batches": state["fixture_batches"],
            "checkpoint": state["checkpoint"],
            "failure": None,
        }
    )
    _write_summary(state["summary_path"], summary)


def smoke_fail(state: dict[str, Any], reason: str) -> None:
    summary = _base_summary(state)
    summary.update(
        {
            "status": "failed",
            "train_ready": False,
            "failure": reason,
            "task_counts": state["task_counts"],
            "lighting_batches": state["lighting_batches"],
            "fixture_batches": state["fixture_batches"],
            "checkpoint": state["checkpoint"],
        }
    )
    _write_summary(state["summary_path"], summary)
    raise RuntimeError(f"TokenLight training smoke failed: {reason}")


def _base_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_VERSION,
        "train_manifest_sha256": state["manifest_sha256"],
        "resume_signature": state["resume_signature"],
        "resumed": state["resumed"],
        "resume_verified": state["resume_verified"],
        "resume_checkpoint": state["resume_checkpoint"],
        "initial_global_step": state["initial_global_step"],
        "initial_samples_seen": state["initial_samples_seen"],
        "expected_next_sample": state["expected_next_sample"],
        "steps_completed_this_phase": state["steps_completed"],
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
