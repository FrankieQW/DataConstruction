from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def euler_sample(
    model: Callable[..., torch.Tensor],
    initial_noise: torch.Tensor,
    source: torch.Tensor,
    lighting_values: torch.Tensor,
    lighting_known: torch.Tensor,
    lighting_valid: torch.Tensor,
    fixture_mask: torch.Tensor,
    fixture_present: torch.Tensor,
    steps: int,
    cfg_scale: float,
    step_callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("infer.steps 必须大于等于 1")
    state = initial_noise
    batch_size = state.shape[0]
    dt = 1.0 / float(steps)
    for step in range(steps):
        time = torch.full((batch_size,), step / float(steps), device=state.device, dtype=torch.float32)
        conditional = model(
            state, time, source, lighting_values, lighting_known, lighting_valid,
            fixture_mask=fixture_mask, fixture_present=fixture_present,
            drop_condition=torch.zeros(batch_size, device=state.device, dtype=torch.bool),
        )
        if cfg_scale == 1.0:
            velocity = conditional
        else:
            unconditional = model(
                state, time, source, lighting_values, lighting_known, lighting_valid,
                fixture_mask=fixture_mask, fixture_present=fixture_present,
                drop_condition=torch.ones(batch_size, device=state.device, dtype=torch.bool),
            )
            velocity = unconditional + float(cfg_scale) * (conditional - unconditional)
        state = state + dt * velocity
        if step_callback is not None:
            step_callback(step + 1, state)
    return state

