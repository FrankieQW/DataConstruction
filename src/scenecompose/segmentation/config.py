from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def _strict_kwargs(cls: type[T], raw: dict[str, Any], section: str) -> dict[str, Any]:
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown {section} config keys: {', '.join(unknown)}")
    return raw


@dataclass(frozen=True)
class VocabularyClass:
    id: int
    name: str
    synonyms: tuple[str, ...] = ()
    kind: str = "thing"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VocabularyClass":
        data = _strict_kwargs(cls, dict(raw), "vocabulary class")
        data["synonyms"] = tuple(data.get("synonyms", ()))
        item = cls(**data)
        if item.id < 0 or not item.name.strip():
            raise ValueError("Vocabulary id must be non-negative and name must be non-empty")
        if item.kind not in {"thing", "stuff"}:
            raise ValueError(f"Vocabulary kind must be 'thing' or 'stuff': {item.name}")
        prompts = (item.name, *item.synonyms)
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError(f"Vocabulary prompts must be non-empty: {item.name}")
        return item


@dataclass(frozen=True)
class GeometryConfig:
    sample_count: int = 500_000
    seed: int = 20260803
    meters_per_blender_unit: float | None = None
    neutral_rgb: tuple[int, int, int] = (128, 128, 128)


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1024
    height: int = 1024
    engine: str = "BLENDER_EEVEE_NEXT"
    horizontal_fov_deg: float = 75.0
    azimuths_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    elevations_deg: tuple[float, ...] = (-15.0, 10.0, 30.0)
    cell_size_m: float = 5.0
    min_depth_coverage: float = 0.08
    target_surface_coverage: float = 0.95
    min_point_observations: int = 2
    max_views: int = 256
    observation_camera_count: int = 24
    observation_yaw_offsets_deg: tuple[float, ...] = (-35.0, -17.5, 0.0, 17.5, 35.0)
    observation_pitch_offsets_deg: tuple[float, ...] = (-12.0, 0.0, 12.0)
    camera_min_clearance_m: float = 0.25


@dataclass(frozen=True)
class MosaicConfig:
    repository: str = "Mosaic3D"
    checkpoint: str = "weights/mosaic3d.ckpt"
    hydra_config: str = "train_spunet_multidata_ppt"
    condition: str = "ScanNet"
    text_model_id: str = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"
    voxel_size_m: float = 0.02
    chunk_points: int = 1_000_000
    chunk_overlap_m: float = 0.25
    top_k: int = 5
    unknown_confidence: float = 0.25
    unknown_margin: float = 0.05


@dataclass(frozen=True)
class Sam3Config:
    repository: str = "sam3"
    checkpoint: str = "weights/sam3.pt"
    bpe_path: str | None = None
    image_batch_size: int = 4
    score_threshold: float = 0.25
    min_mask_area_px: int = 64
    max_border_fraction: float = 0.95


@dataclass(frozen=True)
class FusionConfig:
    lift_radius_m: float = 0.035
    depth_tolerance_m: float = 0.05
    association_grid_m: float = 0.5
    association_threshold: float = 0.55
    min_instance_points: int = 50
    min_instance_views: int = 2
    mosaic_weight: float = 0.45
    sam3_weight: float = 0.35
    view_weight: float = 0.10
    visibility_weight: float = 0.10
    face_semantic_vote: float = 0.45
    face_instance_vote: float = 0.55
    visibility_pixel_stride: int = 2


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int = 1
    gpus: tuple[int, ...] = (0,)
    workers: int = 1
    supported_suffixes: tuple[str, ...] = (".fbx",)


@dataclass(frozen=True)
class SegmentationConfig:
    vocabulary: tuple[VocabularyClass, ...]
    geometry: GeometryConfig = GeometryConfig()
    render: RenderConfig = RenderConfig()
    mosaic3d: MosaicConfig = MosaicConfig()
    sam3: Sam3Config = Sam3Config()
    fusion: FusionConfig = FusionConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_json(cls, path: Path) -> "SegmentationConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _strict_kwargs(cls, raw, "segmentation")
        config = cls(
            vocabulary=tuple(VocabularyClass.from_dict(item) for item in raw["vocabulary"]),
            geometry=_section(GeometryConfig, raw.get("geometry", {}), "geometry"),
            render=_section(
                RenderConfig, raw.get("render", {}), "render",
                tuple_keys={
                    "azimuths_deg", "elevations_deg", "observation_yaw_offsets_deg",
                    "observation_pitch_offsets_deg",
                },
            ),
            mosaic3d=_section(MosaicConfig, raw.get("mosaic3d", {}), "mosaic3d"),
            sam3=_section(Sam3Config, raw.get("sam3", {}), "sam3"),
            fusion=_section(FusionConfig, raw.get("fusion", {}), "fusion"),
            runtime=_section(RuntimeConfig, raw.get("runtime", {}), "runtime", tuple_keys={"gpus", "supported_suffixes"}),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")
        ids = [item.id for item in self.vocabulary]
        names = [item.name.casefold() for item in self.vocabulary]
        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary ids must be unique")
        if len(names) != len(set(names)):
            raise ValueError("Vocabulary names must be unique")
        if self.geometry.sample_count <= 0 or self.runtime.workers <= 0:
            raise ValueError("sample_count and workers must be positive")
        if self.runtime.schema_version != 1:
            raise ValueError(f"Unsupported runtime schema_version: {self.runtime.schema_version}")
        if self.geometry.meters_per_blender_unit is not None and self.geometry.meters_per_blender_unit <= 0:
            raise ValueError("geometry.meters_per_blender_unit must be positive when set")
        if not self.runtime.gpus or any(gpu < 0 for gpu in self.runtime.gpus):
            raise ValueError("runtime.gpus must contain non-negative GPU ids")
        if (
            self.render.width <= 0 or self.render.height <= 0 or self.render.max_views <= 0
            or self.render.observation_camera_count <= 0
            or self.render.camera_min_clearance_m <= 0
        ):
            raise ValueError("render dimensions and max_views must be positive")
        if not self.render.observation_yaw_offsets_deg or not self.render.observation_pitch_offsets_deg:
            raise ValueError("observation camera yaw/pitch offsets must not be empty")
        if not 1.0 < self.render.horizontal_fov_deg < 179.0:
            raise ValueError("render.horizontal_fov_deg must be between 1 and 179 degrees")
        for name, value in {
            "min_depth_coverage": self.render.min_depth_coverage,
            "target_surface_coverage": self.render.target_surface_coverage,
            "unknown_confidence": self.mosaic3d.unknown_confidence,
            "unknown_margin": self.mosaic3d.unknown_margin,
            "sam3.score_threshold": self.sam3.score_threshold,
            "max_border_fraction": self.sam3.max_border_fraction,
            "association_threshold": self.fusion.association_threshold,
            "face_semantic_vote": self.fusion.face_semantic_vote,
            "face_instance_vote": self.fusion.face_instance_vote,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        weights = (
            self.fusion.mosaic_weight + self.fusion.sam3_weight
            + self.fusion.view_weight + self.fusion.visibility_weight
        )
        if abs(weights - 1.0) > 1e-6:
            raise ValueError("fusion evidence weights must sum to 1")
        if self.fusion.visibility_pixel_stride <= 0:
            raise ValueError("fusion.visibility_pixel_stride must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def vocabulary_digest(self) -> str:
        return _digest([asdict(item) for item in self.vocabulary])

    def stage_digest(self, stage: str) -> str:
        dependencies = {
            "geometry": {"geometry": asdict(self.geometry)},
            "views": {"geometry": asdict(self.geometry), "render": asdict(self.render)},
            "mosaic3d": {"geometry": asdict(self.geometry), "mosaic3d": asdict(self.mosaic3d), "vocabulary": self.vocabulary_digest()},
            "sam3": {"render": asdict(self.render), "sam3": asdict(self.sam3), "vocabulary": self.vocabulary_digest()},
            "fusion": {"fusion": asdict(self.fusion), "vocabulary": self.vocabulary_digest()},
            "export": {"vocabulary": self.vocabulary_digest()},
        }
        if stage not in dependencies:
            raise ValueError(f"Unknown segmentation stage: {stage}")
        return _digest(dependencies[stage])


def _section(cls: type[T], raw: dict[str, Any], name: str, tuple_keys: set[str] | None = None) -> T:
    data = dict(_strict_kwargs(cls, dict(raw), name))
    for key in tuple_keys or set():
        if key in data:
            data[key] = tuple(data[key])
    if cls is GeometryConfig and "neutral_rgb" in data:
        data["neutral_rgb"] = tuple(data["neutral_rgb"])
    return cls(**data)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
