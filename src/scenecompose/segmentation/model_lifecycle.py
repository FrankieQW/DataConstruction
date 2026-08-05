from __future__ import annotations

import gc
from typing import Any


def release_cuda_model(owner: Any, *attribute_names: str) -> None:
    """Drop model references and return unused CUDA allocations to PyTorch."""
    torch_module = getattr(owner, "_torch", None)
    for name in attribute_names:
        if hasattr(owner, name):
            setattr(owner, name, None)
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()

