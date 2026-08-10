from __future__ import annotations

from typing import Any

import torch


def tokenlight_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("TokenLight batch 不能为空")
    tensor_keys = (
        "source_image", "target_image", "fixture_mask", "fixture_present",
        "lighting_values", "lighting_known", "lighting_valid", "task",
    )
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in tensor_keys}
    for key in ("task_name", "scene_id", "asset_uid", "sample_index", "sample_seed"):
        batch[key] = [sample[key] for sample in samples]
    return batch
