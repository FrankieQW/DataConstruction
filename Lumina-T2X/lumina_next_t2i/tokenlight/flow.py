from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_linear_path(target: torch.Tensor, time_min: float = 0.0, time_max: float = 1.0):
    batch_size = target.shape[0]
    time = torch.rand(batch_size, device=target.device, dtype=torch.float32)
    time = time * (float(time_max) - float(time_min)) + float(time_min)
    noise = torch.randn_like(target)
    view_shape = (batch_size,) + (1,) * (target.ndim - 1)
    time_view = time.view(view_shape).to(target.dtype)
    noisy_target = (1.0 - time_view) * noise + time_view * target
    velocity = target - noise
    return noisy_target, time, velocity


def velocity_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)

