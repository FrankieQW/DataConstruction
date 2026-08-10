from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


TASK_NAMES = ("ambient_scale", "global_diffuse", "add_light", "in_scene_light")
TASK_IDS = {name: index for index, name in enumerate(TASK_NAMES)}
ADD_LIGHT_FIELDS = ("x", "y", "z", "r", "g", "b", "intensity", "diffuse")
FIXTURE_FIELDS = ("r", "g", "b", "intensity", "transition")


@dataclass(frozen=True)
class PackedLighting:
    values: np.ndarray
    known: np.ndarray
    valid: np.ndarray


class LightingSchema:
    """Canonical scalar-token ordering shared by data, training and inference."""

    def __init__(self, max_lights: int):
        if max_lights < 1:
            raise ValueError("max_lights must be at least 1")
        names = ["ambient", "global_diffuse"]
        for slot in range(max_lights):
            names.append(f"add_light.{slot}.valid")
            names.extend(f"add_light.{slot}.{field}" for field in ADD_LIGHT_FIELDS)
        names.extend(f"in_scene.{field}" for field in FIXTURE_FIELDS)
        self.max_lights = int(max_lights)
        self.names = tuple(names)
        self.index = {name: index for index, name in enumerate(self.names)}

    def empty(self) -> PackedLighting:
        count = len(self.names)
        return PackedLighting(
            values=np.zeros(count, dtype=np.float32),
            known=np.zeros(count, dtype=np.float32),
            valid=np.zeros(count, dtype=np.float32),
        )

    def ambient(self, scale: float) -> PackedLighting:
        packed = self.empty()
        self._set(packed, "ambient", scale)
        return packed

    def global_diffuse(self, delta: float) -> PackedLighting:
        packed = self.empty()
        self._set(packed, "global_diffuse", delta)
        return packed

    def add_lights(self, lights: Iterable[dict[str, float]]) -> PackedLighting:
        packed = self.empty()
        lights = list(lights)
        if len(lights) > self.max_lights:
            raise ValueError(f"Received {len(lights)} lights, max_lights={self.max_lights}")
        for slot in range(self.max_lights):
            slot_valid = slot < len(lights)
            self._set(packed, f"add_light.{slot}.valid", float(slot_valid))
            if not slot_valid:
                continue
            light = lights[slot]
            missing = [field for field in ADD_LIGHT_FIELDS if field not in light]
            if missing:
                raise ValueError(f"add_light slot {slot} 缺少字段: {missing}")
            for field in ADD_LIGHT_FIELDS:
                self._set(packed, f"add_light.{slot}.{field}", float(light[field]))
        return packed

    def in_scene(self, values: dict[str, float]) -> PackedLighting:
        packed = self.empty()
        missing = [field for field in FIXTURE_FIELDS if field not in values]
        if missing:
            raise ValueError(f"in_scene_light 缺少字段: {missing}")
        for field in FIXTURE_FIELDS:
            self._set(packed, f"in_scene.{field}", float(values[field]))
        return packed

    def _set(self, packed: PackedLighting, name: str, value: float) -> None:
        index = self.index[name]
        packed.values[index] = np.float32(value)
        packed.known[index] = 1.0
        packed.valid[index] = 1.0


class LightingTokenEncoder(nn.Module):
    """Encode every scalar with its own Gaussian-Fourier projection."""

    def __init__(self, schema: LightingSchema, hidden_size: int, feature_count: int, sigma: float, seed: int):
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be at least 1")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        frequencies = torch.randn(len(schema.names), feature_count, generator=generator) * float(sigma)
        self.register_buffer("frequencies", frequencies, persistent=True)
        input_size = feature_count * 2 + 2
        self.projections = nn.ModuleList(nn.Linear(input_size, hidden_size) for _ in schema.names)
        self.schema_names = schema.names
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        values: torch.Tensor,
        known: torch.Tensor,
        valid: torch.Tensor,
        drop_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or values.shape[1] != len(self.projections):
            raise ValueError(f"lighting values shape 应为 [B, {len(self.projections)}]，实际为 {tuple(values.shape)}")
        if known.shape != values.shape or valid.shape != values.shape:
            raise ValueError("lighting known/valid shape 必须与 values 相同")
        frequencies = self.frequencies.to(device=values.device, dtype=torch.float32)
        phase = values.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        features = torch.cat(
            [torch.sin(phase), torch.cos(phase), known.float().unsqueeze(-1), valid.float().unsqueeze(-1)],
            dim=-1,
        )
        tokens = torch.stack(
            [projection(features[:, index].to(projection.weight.dtype)) for index, projection in enumerate(self.projections)],
            dim=1,
        )
        token_mask = valid.bool()
        if drop_condition is not None:
            if drop_condition.shape != (values.shape[0],):
                raise ValueError("drop_condition shape 必须为 [B]")
            token_mask = token_mask & ~drop_condition.bool().unsqueeze(1)
        return tokens, token_mask

