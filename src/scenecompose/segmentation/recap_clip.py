from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RECAP_CLIP_FILES = (
    "open_clip_config.json",
    "open_clip_pytorch_model.bin",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def required_recap_clip_paths(model_dir: Path) -> dict[str, Path]:
    return {name: model_dir / name for name in REQUIRED_RECAP_CLIP_FILES}


def load_local_recap_clip(model_dir: Path, device: str):
    """Construct ReCap-CLIP from a flat local OpenCLIP model directory."""
    model_dir = model_dir.resolve()
    config_path = model_dir / "open_clip_config.json"
    checkpoint_path = model_dir / "open_clip_pytorch_model.bin"

    try:
        raw_config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read local ReCap-CLIP config: {config_path}") from exc
    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("model_cfg"), dict):
        raise RuntimeError(f"ReCap-CLIP config has no object-valued model_cfg: {config_path}")

    model_cfg = dict(raw_config["model_cfg"])
    text_cfg = model_cfg.get("text_cfg")
    if not isinstance(text_cfg, dict):
        raise RuntimeError(f"ReCap-CLIP model_cfg has no object-valued text_cfg: {config_path}")

    from open_clip.factory import load_checkpoint
    from open_clip.model import CLIP
    from open_clip.tokenizer import HFTokenizer

    model = CLIP(**model_cfg)
    load_checkpoint(model, str(checkpoint_path), strict=True)

    tokenizer_kwargs = dict(text_cfg.get("tokenizer_kwargs", {}))
    tokenizer_kwargs["local_files_only"] = True
    tokenizer = HFTokenizer(
        str(model_dir),
        context_length=int(text_cfg.get("context_length", 77)),
        **tokenizer_kwargs,
    )
    model.text_tokenizer = tokenizer
    return model.to(device).eval()
