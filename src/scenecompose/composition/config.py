from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def _strict(cls: type[T], raw: dict[str, Any], section: str) -> dict[str, Any]:
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown {section} config keys: {', '.join(unknown)}")
    return raw


@dataclass(frozen=True)
class CanonicalClassRule:
    canonical_class: str
    placement_type: str
    support_classes: tuple[str, ...]
    terms: tuple[str, ...]
    target_dimension: str
    target_min_m: float
    target_max_m: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CanonicalClassRule":
        data = dict(_strict(cls, raw, "canonical class rule"))
        data["support_classes"] = tuple(data.get("support_classes", ()))
        data["terms"] = tuple(data.get("terms", ()))
        return cls(**data)


@dataclass(frozen=True)
class CatalogConfig:
    seed: int = 20260804
    blender_batch_size: int = 16
    workers: int = 2
    minimum_faces: int = 20
    maximum_faces: int = 2_000_000
    maximum_components: int = 64
    minimum_primary_component_ratio: float = 0.65
    allowed_licenses: tuple[str, ...] = ("cc0", "by", "by-sa", "by-nc", "by-nc-sa")
    reject_metadata_categories: tuple[str, ...] = (
        "architecture", "characters-creatures", "cultural-heritage-history",
        "art-abstract", "weapons-military", "cars-vehicles", "animals-pets", "people",
    )
    class_rules: tuple[CanonicalClassRule, ...] = ()


@dataclass(frozen=True)
class LlmConfig:
    enabled: bool = False
    backend: str = "transformers"
    model_path: str = "weights/Qwen3"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    batch_size: int = 8
    max_new_tokens: int = 256
    trust_remote_code: bool = False
    local_files_only: bool = True
    temperature: float = 0.0
    max_retries: int = 2
    acceptance_threshold: float = 0.75


@dataclass(frozen=True)
class NormalizationConfig:
    component_distance_ratio: float = 0.10
    upright_axis_candidates: tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    bottom_band_ratio: float = 0.03
    collision_decimation_faces: int = 20_000
    minimum_footprint_ratio: float = 0.01
    maximum_scale_factor: float = 100.0
    minimum_scale_factor: float = 0.001
    uniform_scale_only: bool = True


@dataclass(frozen=True)
class SupportConfig:
    allowed_support_classes: tuple[str, ...] = (
        "table", "desk", "counter", "shelf", "cabinet", "floor",
    )
    minimum_visible_views: int = 1
    maximum_slope_deg: float = 12.0
    plane_height_tolerance_m: float = 0.04
    adjacency_tolerance_m: float = 0.03
    minimum_patch_area_m2: float = 0.04
    boundary_margin_m: float = 0.03
    occupancy_cell_m: float = 0.02


@dataclass(frozen=True)
class PlacementConfig:
    seed: int = 20260804
    objects_per_observation: int = 1
    max_object_trials: int = 12
    positions_per_surface: int = 24
    yaw_trials: int = 12
    minimum_support_coverage: float = 0.92
    minimum_clearance_m: float = 0.005
    maximum_penetration_m: float = 0.003
    minimum_camera_visible_ratio: float = 0.20
    camera_samples: int = 24
    object_visibility_samples: int = 256
    horizontal_fov_deg: float = 60.0


@dataclass(frozen=True)
class CompositionConfig:
    schema_version: int
    catalog: CatalogConfig
    llm: LlmConfig
    object_normalization: NormalizationConfig
    support: SupportConfig
    placement: PlacementConfig

    @classmethod
    def from_json(cls, path: Path) -> "CompositionConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _strict(cls, raw, "composition")
        catalog_raw = dict(raw.get("catalog", {}))
        class_rules = tuple(
            CanonicalClassRule.from_dict(value)
            for value in catalog_raw.pop("class_rules", ())
        )
        for key in ("allowed_licenses", "reject_metadata_categories"):
            if key in catalog_raw:
                catalog_raw[key] = tuple(catalog_raw[key])
        catalog = CatalogConfig(
            **_strict(CatalogConfig, catalog_raw, "catalog"), class_rules=class_rules
        )
        normalization_raw = dict(raw.get("object_normalization", {}))
        if "upright_axis_candidates" in normalization_raw:
            normalization_raw["upright_axis_candidates"] = tuple(
                normalization_raw["upright_axis_candidates"]
            )
        support_raw = dict(raw.get("support", {}))
        if "allowed_support_classes" in support_raw:
            support_raw["allowed_support_classes"] = tuple(
                support_raw["allowed_support_classes"]
            )
        config = cls(
            schema_version=int(raw.get("schema_version", 1)),
            catalog=catalog,
            llm=LlmConfig(**_strict(LlmConfig, dict(raw.get("llm", {})), "llm")),
            object_normalization=NormalizationConfig(
                **_strict(NormalizationConfig, normalization_raw, "object_normalization")
            ),
            support=SupportConfig(**_strict(SupportConfig, support_raw, "support")),
            placement=PlacementConfig(
                **_strict(PlacementConfig, dict(raw.get("placement", {})), "placement")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported composition schema_version: {self.schema_version}")
        if self.catalog.workers < 1 or self.catalog.blender_batch_size < 1:
            raise ValueError("catalog workers and blender_batch_size must be positive")
        if not 0 < self.catalog.minimum_primary_component_ratio <= 1:
            raise ValueError("minimum_primary_component_ratio must be in (0, 1]")
        if self.catalog.minimum_faces < 1 or self.catalog.maximum_faces < self.catalog.minimum_faces:
            raise ValueError("catalog face limits are invalid")
        if not self.catalog.class_rules:
            raise ValueError("catalog.class_rules must not be empty")
        names = [rule.canonical_class.casefold() for rule in self.catalog.class_rules]
        if len(names) != len(set(names)):
            raise ValueError("canonical class names must be unique")
        for rule in self.catalog.class_rules:
            if rule.placement_type not in {"surface", "floor"}:
                raise ValueError(f"Unsupported placement type: {rule.placement_type}")
            if rule.target_dimension not in {"height", "longest"}:
                raise ValueError(f"Unsupported target dimension: {rule.target_dimension}")
            if not rule.terms or not rule.support_classes:
                raise ValueError(f"Class rule is incomplete: {rule.canonical_class}")
            if not 0 < rule.target_min_m <= rule.target_max_m:
                raise ValueError(f"Invalid target size: {rule.canonical_class}")
        if self.llm.backend != "transformers":
            raise ValueError("The first implementation supports only llm.backend='transformers'")
        if not self.llm.local_files_only:
            raise ValueError("llm.local_files_only must remain true")
        if self.llm.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("llm.dtype must be bfloat16, float16, or float32")
        if self.llm.batch_size < 1 or self.llm.max_new_tokens < 1 or self.llm.max_retries < 1:
            raise ValueError("LLM batch/token/retry values must be positive")
        if not 0 <= self.llm.temperature <= 2 or not 0 <= self.llm.acceptance_threshold <= 1:
            raise ValueError("LLM temperature/acceptance values are out of range")
        if not self.object_normalization.uniform_scale_only:
            raise ValueError("object_normalization.uniform_scale_only must remain true")
        if self.placement.objects_per_observation != 1:
            raise ValueError("The first implementation supports one Object per observation")
        positive = {
            "support.occupancy_cell_m": self.support.occupancy_cell_m,
            "support.minimum_patch_area_m2": self.support.minimum_patch_area_m2,
            "placement.max_object_trials": self.placement.max_object_trials,
            "placement.positions_per_surface": self.placement.positions_per_surface,
            "placement.yaw_trials": self.placement.yaw_trials,
            "placement.camera_samples": self.placement.camera_samples,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Composition values must be positive: {', '.join(invalid)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

