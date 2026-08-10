from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when the TokenLight YAML configuration is incomplete."""


REQUIRED_FIELDS = (
    "paths.upstream_checkpoint",
    "paths.vae",
    "paths.dataset_root",
    "paths.train_manifest",
    "paths.validation_manifest",
    "paths.output_root",
    "data.resolution",
    "data.crop_mode",
    "data.exposure",
    "data.tone_mapping",
    "data.num_workers",
    "data.pin_memory",
    "data.samples_per_scene_train",
    "data.samples_per_scene_validation",
    "data.deterministic_validation",
    "data.validation_seed",
    "data.tasks",
    "data.task_probabilities",
    "model.name",
    "model.latent_channels",
    "model.patch_size",
    "model.dim",
    "model.layers",
    "model.heads",
    "model.multiple_of",
    "model.norm_epsilon",
    "model.qk_norm",
    "model.learn_sigma",
    "model.cap_feat_dim",
    "model.rope_max_size",
    "model.max_lights",
    "model.fixture_mask_enabled",
    "model.fixture_mask_patch_size",
    "lighting.fourier_features",
    "lighting.fourier_sigma",
    "lighting.fourier_seed",
    "lighting.cfg_dropout",
    "lighting.ambient_scale_range",
    "lighting.light_color_range",
    "lighting.light_intensity_range",
    "lighting.fixture_intensity_range",
    "lighting.fixture_transition_range",
    "flow.path",
    "flow.prediction",
    "flow.time_min",
    "flow.time_max",
    "flow.loss",
    "train.seed",
    "train.precision",
    "train.micro_batch_size",
    "train.gradient_accumulation_steps",
    "train.max_steps",
    "train.learning_rate",
    "train.weight_decay",
    "train.betas",
    "train.gradient_clip_norm",
    "train.vae_scale",
    "train.vae_shift",
    "train.validation_every_steps",
    "train.validation_batches",
    "train.checkpoint_every_steps",
    "infer.seed",
    "infer.steps",
    "infer.solver",
    "infer.cfg_scale",
    "infer.precision",
    "infer.vae_scale",
    "infer.vae_shift",
    "infer.save_metadata",
    "evaluate.split",
    "evaluate.batch_size",
    "evaluate.seed",
    "evaluate.metrics",
    "evaluate.lpips_network",
    "evaluate.output_directory",
    "evaluate.save_images",
    "logging.run_id",
    "logging.log_every_steps",
    "logging.tensorboard",
    "logging.train_jsonl",
    "logging.validation_jsonl",
    "logging.checkpoint_index_jsonl",
    "runtime.device",
    "runtime.gpu_ids",
    "runtime.model_parallel_size",
    "runtime.flash_attention",
    "runtime.activation_checkpointing",
)


def get_config_value(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"配置缺少必填字段: {dotted_path}")
        value = value[part]
    if value is None or value == "":
        raise ConfigError(f"配置字段不能为空: {dotted_path}")
    return value


def require_fields(config: dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        get_config_value(config, field)


def load_config(path: str | Path, required_fields: Iterable[str] = REQUIRED_FIELDS) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"TokenLight 配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"配置根节点必须是 mapping: {config_path}")

    config = deepcopy(loaded)
    require_fields(config, required_fields)
    config["_config_path"] = str(config_path)
    _validate_values(config)
    return config


def _validate_values(config: dict[str, Any]) -> None:
    if config["flow"]["path"] != "linear":
        raise ConfigError("flow.path 当前只支持 linear")
    if config["flow"]["prediction"] != "velocity":
        raise ConfigError("flow.prediction 当前只支持 velocity")
    if config["flow"]["loss"] != "mse":
        raise ConfigError("flow.loss 当前只支持 mse")
    if config["infer"]["solver"] != "euler":
        raise ConfigError("infer.solver 当前只支持 euler")
    if not config["runtime"]["flash_attention"]:
        raise ConfigError("当前 Next-DiT 实现要求 runtime.flash_attention=true")
    if config["data"]["tone_mapping"] != "reinhard":
        raise ConfigError("data.tone_mapping 当前只支持 reinhard")
    resolution = int(config["data"]["resolution"])
    patch_multiple = 8 * int(config["model"]["patch_size"])
    if resolution <= 0 or resolution % patch_multiple:
        raise ConfigError(f"data.resolution 必须是 {patch_multiple} 的正整数倍")
    if int(config["model"]["max_lights"]) < 1:
        raise ConfigError("model.max_lights 必须大于等于 1")
    allowed_tasks = {"ambient_scale", "global_diffuse", "add_light", "in_scene_light"}
    unknown = set(config["data"]["tasks"]) - allowed_tasks
    if unknown:
        raise ConfigError(f"data.tasks 包含未知任务: {sorted(unknown)}")
    probabilities = config["data"].get("task_probabilities", {})
    if any(float(probabilities.get(task, 0.0)) < 0 for task in allowed_tasks):
        raise ConfigError("data.task_probabilities 不能包含负数")
    if sum(float(probabilities.get(task, 0.0)) for task in config["data"]["tasks"]) <= 0:
        raise ConfigError("data.tasks 对应的 task_probabilities 之和必须大于 0")
    if not 0.0 <= float(config["lighting"]["cfg_dropout"]) <= 1.0:
        raise ConfigError("lighting.cfg_dropout 必须在 [0, 1] 内")
    if not 0.0 <= float(config["flow"]["time_min"]) < float(config["flow"]["time_max"]) <= 1.0:
        raise ConfigError("flow.time_min/time_max 必须满足 0 <= min < max <= 1")
    positive_fields = (
        "data.samples_per_scene_train",
        "data.samples_per_scene_validation",
        "model.latent_channels",
        "model.dim",
        "model.layers",
        "model.heads",
        "model.rope_max_size",
        "lighting.fourier_features",
        "train.micro_batch_size",
        "train.gradient_accumulation_steps",
        "train.max_steps",
        "train.validation_every_steps",
        "train.validation_batches",
        "train.checkpoint_every_steps",
        "infer.steps",
        "evaluate.batch_size",
        "logging.log_every_steps",
    )
    for field in positive_fields:
        if int(get_config_value(config, field)) < 1:
            raise ConfigError(f"{field} 必须大于等于 1")
    if len(config["train"]["betas"]) != 2:
        raise ConfigError("train.betas 必须包含两个数值")
    smoke = config["train"].get("smoke", {})
    if not isinstance(smoke, dict):
        raise ConfigError("train.smoke must be a mapping")
    if bool(smoke.get("enabled", False)):
        required_tasks = smoke.get("required_tasks", config["data"]["tasks"])
        if not isinstance(required_tasks, list) or not required_tasks:
            raise ConfigError("train.smoke.required_tasks must be a non-empty list")
        if set(required_tasks) != set(config["data"]["tasks"]):
            raise ConfigError("train.smoke.required_tasks must match all enabled data.tasks")
        if len(required_tasks) != len(set(required_tasks)):
            raise ConfigError("train.smoke.required_tasks cannot contain duplicates")
        if int(config["data"]["num_workers"]) != 0:
            raise ConfigError("training smoke requires data.num_workers=0")
        summary_value = str(smoke.get("summary_json", "smoke_summary.json"))
        summary_json = Path(summary_value)
        if not summary_value or summary_json == Path("."):
            raise ConfigError("train.smoke.summary_json must name a JSON file")
        if summary_json.is_absolute() or ".." in summary_json.parts:
            raise ConfigError("train.smoke.summary_json must stay inside the run directory")
    gpu_ids = config["runtime"]["gpu_ids"]
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ConfigError("runtime.gpu_ids 必须是非空 GPU 编号列表")
    parsed_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    if bool(smoke.get("enabled", False)) and len(parsed_gpu_ids) != 1:
        raise ConfigError("training smoke requires exactly one runtime.gpu_ids entry")
    if len(set(parsed_gpu_ids)) != len(parsed_gpu_ids) or any(gpu_id < 0 for gpu_id in parsed_gpu_ids):
        raise ConfigError("runtime.gpu_ids 必须包含互不重复的非负 GPU 编号")
    if int(config["runtime"]["model_parallel_size"]) != 1:
        raise ConfigError("TokenLight DDP 训练要求 runtime.model_parallel_size=1")
