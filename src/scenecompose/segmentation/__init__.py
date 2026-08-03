"""Whole-scene semantic and instance segmentation."""

from .config import SegmentationConfig
from .pipeline import run_segmentation_batch

__all__ = ["SegmentationConfig", "run_segmentation_batch"]
