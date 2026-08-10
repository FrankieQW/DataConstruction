from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.model import NextDiT

from .tokens import LightingSchema, LightingTokenEncoder


class TokenLightNextDiT(NextDiT):
    """Next-DiT variant with joint target, source, lighting and fixture-mask tokens."""

    def __init__(self, config: dict[str, Any]):
        model = config["model"]
        super().__init__(
            patch_size=int(model["patch_size"]),
            in_channels=int(model["latent_channels"]),
            dim=int(model["dim"]),
            n_layers=int(model["layers"]),
            n_heads=int(model["heads"]),
            n_kv_heads=model.get("kv_heads"),
            multiple_of=int(model["multiple_of"]),
            ffn_dim_multiplier=model.get("ffn_dim_multiplier"),
            norm_eps=float(model["norm_epsilon"]),
            learn_sigma=bool(model["learn_sigma"]),
            qk_norm=bool(model["qk_norm"]),
            cap_feat_dim=int(model["cap_feat_dim"]),
            rope_max_size=int(model["rope_max_size"]),
        )
        self.schema = LightingSchema(int(model["max_lights"]))
        lighting = config["lighting"]
        self.lighting_encoder = LightingTokenEncoder(
            self.schema,
            hidden_size=self.dim,
            feature_count=int(lighting["fourier_features"]),
            sigma=float(lighting["fourier_sigma"]),
            seed=int(lighting["fourier_seed"]),
        )
        self.fixture_mask_enabled = bool(model["fixture_mask_enabled"])
        self.fixture_mask_patch_size = int(model["fixture_mask_patch_size"])
        if self.fixture_mask_enabled:
            if self.fixture_mask_patch_size != self.patch_size:
                raise ValueError("第一版要求 model.fixture_mask_patch_size 与 model.patch_size 相同")
            self.fixture_mask_embedder = nn.Linear(self.fixture_mask_patch_size**2, self.dim)
            nn.init.xavier_uniform_(self.fixture_mask_embedder.weight)
            nn.init.zeros_(self.fixture_mask_embedder.bias)
        self.activation_checkpointing = bool(config["runtime"]["activation_checkpointing"])

    def forward(
        self,
        noisy_target: torch.Tensor,
        time: torch.Tensor,
        source: torch.Tensor,
        lighting_values: torch.Tensor,
        lighting_known: torch.Tensor,
        lighting_valid: torch.Tensor,
        fixture_mask: torch.Tensor | None = None,
        fixture_present: torch.Tensor | None = None,
        drop_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_target.shape != source.shape:
            raise ValueError(
                f"noisy target 与 source latent shape 必须相同: {tuple(noisy_target.shape)} != {tuple(source.shape)}"
            )
        target_tokens, target_mask, image_sizes, target_freqs = self.patchify_and_embed(noisy_target)
        source_tokens, source_mask, source_sizes, source_freqs = self.patchify_and_embed(source)
        if image_sizes != source_sizes:
            raise ValueError("target/source latent 空间尺寸必须相同")

        batch_size = noisy_target.shape[0]
        target_freqs = target_freqs.expand(batch_size, -1, -1)
        source_freqs = source_freqs.expand(batch_size, -1, -1)
        light_tokens, light_mask = self.lighting_encoder(
            lighting_values, lighting_known, lighting_valid, drop_condition=drop_condition
        )
        light_tokens = light_tokens.to(target_tokens.dtype)
        rope_width = target_freqs.shape[-1]
        light_freqs = torch.ones(
            batch_size,
            light_tokens.shape[1],
            rope_width,
            dtype=target_freqs.dtype,
            device=target_freqs.device,
        )

        token_groups = [target_tokens, source_tokens, light_tokens]
        mask_groups = [target_mask.bool(), source_mask.bool(), light_mask]
        frequency_groups = [target_freqs, source_freqs, light_freqs]

        if self.fixture_mask_enabled:
            mask_tokens, mask_valid, mask_freqs = self._embed_fixture_mask(
                fixture_mask,
                fixture_present,
                drop_condition,
                noisy_target.shape[-2:],
                target_freqs,
                target_tokens.dtype,
            )
            token_groups.append(mask_tokens)
            mask_groups.append(mask_valid)
            frequency_groups.append(mask_freqs)

        hidden = torch.cat(token_groups, dim=1)
        joint_mask = torch.cat(mask_groups, dim=1)
        joint_freqs = torch.cat(frequency_groups, dim=1)
        adaln_input = self.t_embedder(time)

        for layer in self.layers:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    lambda value, block=layer: block(
                        value, joint_mask, joint_freqs, None, None, adaln_input=adaln_input
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, joint_mask, joint_freqs, None, None, adaln_input=adaln_input)

        target_length = target_tokens.shape[1]
        output = self.final_layer(hidden[:, :target_length], adaln_input)
        output = self.unpatchify(output, image_sizes, return_tensor=True)
        if self.learn_sigma:
            output, _ = output.chunk(2, dim=1)
        return output

    def _embed_fixture_mask(
        self,
        fixture_mask: torch.Tensor | None,
        fixture_present: torch.Tensor | None,
        drop_condition: torch.Tensor | None,
        latent_size: tuple[int, int],
        spatial_freqs: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = spatial_freqs.shape[0]
        if fixture_mask is None:
            fixture_mask = torch.zeros(batch_size, 1, *latent_size, device=spatial_freqs.device)
        if fixture_present is None:
            fixture_present = torch.zeros(batch_size, dtype=torch.bool, device=spatial_freqs.device)
        fixture_mask = F.interpolate(fixture_mask.float(), size=latent_size, mode="area")
        patch_size = self.fixture_mask_patch_size
        height, width = latent_size
        patches = fixture_mask.view(
            batch_size, 1, height // patch_size, patch_size, width // patch_size, patch_size
        ).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)
        tokens = self.fixture_mask_embedder(patches.to(self.fixture_mask_embedder.weight.dtype)).to(dtype)
        valid = fixture_present.bool().unsqueeze(1).expand(-1, tokens.shape[1])
        if drop_condition is not None:
            valid = valid & ~drop_condition.bool().unsqueeze(1)
        if tokens.shape[1] != spatial_freqs.shape[1]:
            raise ValueError("fixture mask patch 数必须与 target patch 数一致")
        return tokens, valid, spatial_freqs


def build_tokenlight_model(config: dict[str, Any]) -> TokenLightNextDiT:
    supported = {"NextDiT_2B_patch2", "NextDiT_2B_GQA_patch2"}
    name = config["model"]["name"]
    if name not in supported:
        raise ValueError(f"TokenLight 当前支持的 model.name: {sorted(supported)}，实际为 {name}")
    return TokenLightNextDiT(config)
