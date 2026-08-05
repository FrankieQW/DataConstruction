from __future__ import annotations

import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_version(name: str, actual: str, prefix: str) -> None:
    if not actual.startswith(prefix):
        raise RuntimeError(f"{name} must be {prefix}.*, found {actual}")


def main() -> None:
    import numpy
    import torch
    import torchvision

    require_version("Python", f"{sys.version_info.major}.{sys.version_info.minor}", "3.12")
    require_version("PyTorch", torch.__version__.split("+")[0], "2.7")
    require_version("torchvision", torchvision.__version__.split("+")[0], "0.22")
    require_version("NumPy", numpy.__version__, "1.26")
    if torch.version.cuda != "12.6":
        raise RuntimeError(f"PyTorch must use CUDA 12.6, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU")

    for module_name in ("spconv.pytorch", "torch_scatter", "torch_cluster", "open_clip"):
        importlib.import_module(module_name)

    for repository in (PROJECT_ROOT / "Mosaic3D", PROJECT_ROOT / "sam3"):
        sys.path.insert(0, str(repository))
    try:
        importlib.import_module("src.models.networks.spunet.spconv_unet_v1m3_pdnorm")
        importlib.import_module("sam3.model_builder")
        importlib.import_module("sam3.model.sam3_image_processor")
    finally:
        del sys.path[:2]

    device = torch.cuda.get_device_name(0)
    print(f"Unified segmentation environment OK: torch={torch.__version__}, "
          f"torchvision={torchvision.__version__}, numpy={numpy.__version__}, gpu={device}")
    print("Model checkpoints were not loaded; run one observation for the model-backed check.")


if __name__ == "__main__":
    main()
