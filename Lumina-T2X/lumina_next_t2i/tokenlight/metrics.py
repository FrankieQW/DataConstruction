from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(1)

