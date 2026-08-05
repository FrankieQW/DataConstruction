from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

from .artifacts import save_npz_atomic
from .config import SegmentationConfig
from .model_lifecycle import release_cuda_model
from .recap_clip import load_local_recap_clip


@contextmanager
def _repository_import(root: Path) -> Iterator[None]:
    original = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = original


def _voxelize(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minimum = points.min(axis=0)
    grid = np.floor((points - minimum) / voxel_size).astype(np.int32)
    unique_grid, unique_indices, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    voxel_colors = np.add.reduceat(colors[order].astype(np.float32), starts, axis=0) / counts[:, None]
    return unique_grid, points[unique_indices], voxel_colors, inverse.astype(np.int64)


class Mosaic3DAdapter:
    def __init__(self, project_root: Path, config: SegmentationConfig, device: str) -> None:
        self.project_root = project_root
        self.config = config
        self.device = device
        self.repository = (project_root / config.mosaic3d.repository).resolve()
        self.checkpoint = (project_root / config.mosaic3d.checkpoint).resolve()
        self.text_model_path = (project_root / config.mosaic3d.text_model_path).resolve()
        self._torch = None
        self._module = None
        self._text_encoder = None

    def release(self) -> None:
        release_cuda_model(self, "_module", "_text_encoder")
        self._torch = None

    def _load(self) -> None:
        if self._module is not None:
            return
        with _repository_import(self.repository):
            import torch
            self._torch = torch
            from hydra import compose, initialize_config_dir
            from hydra.utils import instantiate
            with initialize_config_dir(version_base="1.3", config_dir=str(self.repository / "configs")):
                cfg = compose(config_name="eval", overrides=[f"experiment={self.config.mosaic3d.hydra_config}"])
            module = instantiate(cfg.model)
            module.net = module.hparams.net()
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            missing, _ = module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
            hard_missing = [name for name in missing if name.startswith("net.")]
            if hard_missing:
                raise RuntimeError(f"Mosaic3D checkpoint is missing network keys: {hard_missing[:8]}")
            module.net.to(self.device).eval()
            text_encoder = load_local_recap_clip(self.text_model_path, self.device)
        self._module = module
        self._text_encoder = text_encoder

    def _text_features(self):
        self._load()
        torch = self._torch
        encoder = self._text_encoder
        class_features = []
        with torch.inference_mode():
            for item in self.config.vocabulary:
                prompts = [item.name, *item.synonyms]
                tokens = encoder.text_tokenizer(prompts).to(self.device)
                features = encoder.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                combined = features.mean(dim=0)
                class_features.append(combined / combined.norm().clamp_min(1e-8))
        return torch.stack(class_features)

    def infer(self, samples_path: Path, output_dir: Path) -> list[Path]:
        self._load()
        torch = self._torch
        with np.load(samples_path, allow_pickle=False) as archive:
            points = archive["points"].astype(np.float32)
            colors = archive["colors"].astype(np.float32)
        if len(points) > self.config.mosaic3d.chunk_points:
            raise RuntimeError(
                f"Mosaic3D input has {len(points)} points, exceeding chunk_points="
                f"{self.config.mosaic3d.chunk_points}; reduce geometry.sample_count"
            )
        grid, voxel_points, voxel_colors, inverse = _voxelize(points, colors, self.config.mosaic3d.voxel_size_m)
        voxel_points = voxel_points - voxel_points.mean(axis=0, keepdims=True)
        batch = {
            "coord": torch.from_numpy(voxel_points).to(self.device),
            "grid_coord": torch.from_numpy(grid).to(self.device),
            "feat": torch.from_numpy(voxel_colors / 127.5 - 1.0).to(self.device),
            "offset": torch.tensor([len(grid)], dtype=torch.long, device=self.device),
            "condition": [self.config.mosaic3d.condition],
        }
        with torch.inference_mode():
            output = self._module.net(batch)
            voxel_features = output.sparse_conv_feat.features[output.v2p_map]
            voxel_features = voxel_features / voxel_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            point_features = voxel_features[torch.from_numpy(inverse).to(self.device)]
            logits = point_features @ self._text_features().T
            probabilities = logits.softmax(dim=-1)
            k = min(self.config.mosaic3d.top_k, probabilities.shape[1])
            confidence, class_indices = probabilities.topk(k, dim=-1)
            margin = confidence[:, 0] - (confidence[:, 1] if k > 1 else 0.0)
            unknown = (confidence[:, 0] < self.config.mosaic3d.unknown_confidence) | (margin < self.config.mosaic3d.unknown_margin)
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_path = output_dir / "point_features.pt"
        temporary = feature_path.with_suffix(".pt.tmp")
        torch.save(point_features.half().cpu(), temporary)
        temporary.replace(feature_path)
        score_path = output_dir / "semantic_scores.npz"
        canonical_ids = np.asarray([item.id for item in self.config.vocabulary], dtype=np.int32)
        save_npz_atomic(
            score_path,
            class_ids=canonical_ids[class_indices.cpu().numpy()],
            confidence=confidence.float().cpu().numpy(),
            unknown=unknown.cpu().numpy(),
        )
        return [feature_path, score_path]
