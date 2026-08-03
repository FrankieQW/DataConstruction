from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

from .artifacts import MaskObservation, encode_mask_rle, save_json_atomic
from .config import SegmentationConfig


@contextmanager
def _repository_import(root: Path) -> Iterator[None]:
    original = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = original


class Sam3Adapter:
    def __init__(self, project_root: Path, config: SegmentationConfig, device: str) -> None:
        self.project_root = project_root
        self.config = config
        self.device = device
        self.repository = (project_root / config.sam3.repository).resolve()
        self.checkpoint = (project_root / config.sam3.checkpoint).resolve()
        self._processor = None

    def _load(self) -> None:
        if self._processor is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        with _repository_import(self.repository):
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            kwargs = {
                "checkpoint_path": str(self.checkpoint),
                "device": self.device,
                "load_from_HF": False,
            }
            if self.config.sam3.bpe_path:
                kwargs["bpe_path"] = str((self.project_root / self.config.sam3.bpe_path).resolve())
            model = build_sam3_image_model(**kwargs)
            self._processor = Sam3Processor(model, device=self.device, confidence_threshold=self.config.sam3.score_threshold)

    def infer(self, rgb_dir: Path, output_dir: Path) -> list[Path]:
        self._load()
        from PIL import Image
        output_dir.mkdir(parents=True, exist_ok=True)
        records: list[MaskObservation] = []
        for image_path in sorted(rgb_dir.glob("*.png")):
            view_id = image_path.stem
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                state = self._processor.set_image(image)
                for item in self.config.vocabulary:
                    for prompt in (item.name, *item.synonyms):
                        state = self._processor.set_text_prompt(prompt=prompt, state=state)
                        masks = state["masks"].detach().cpu().numpy()
                        scores = state["scores"].detach().float().cpu().numpy()
                        boxes = state["boxes"].detach().float().cpu().numpy()
                        for index, (mask_tensor, score, box) in enumerate(zip(masks, scores, boxes)):
                            mask = np.asarray(mask_tensor).squeeze().astype(bool)
                            area = int(mask.sum())
                            if area < self.config.sam3.min_mask_area_px:
                                continue
                            border = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
                            perimeter_scale = max(1, 2 * (mask.shape[0] + mask.shape[1]))
                            if border / perimeter_scale > self.config.sam3.max_border_fraction:
                                continue
                            key = f"{view_id}|{item.id}|{prompt}|{index}"
                            records.append(MaskObservation(
                                hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], view_id,
                                item.id, prompt, float(score), tuple(float(value) for value in box),
                                mask.shape[0], mask.shape[1], tuple(encode_mask_rle(mask)),
                            ))
        records.sort(key=lambda item: (item.view_id, item.class_id, item.mask_id))
        index_path = output_dir / "masks.json"
        save_json_atomic(index_path, {"schema_version": 1, "masks": [item.to_dict() for item in records]})
        return [index_path]
