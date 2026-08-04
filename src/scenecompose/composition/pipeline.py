from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import subprocess

from scenecompose.segmentation.observation_discovery import discover_observations

from .config import CompositionConfig
from .contracts import stable_digest, write_json_atomic
from .support import extract_support_patches


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def compose_observations(
    observation_root: Path, catalog_path: Path, output_root: Path,
    config_path: Path, segmentation_config: Path, blender: str,
    force: bool = False, dry_run: bool = False,
) -> int:
    config = CompositionConfig.from_json(config_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    usable = []
    for record in catalog.get("records", []):
        profile_path = Path(record["profile"])
        classification = record["classification"]
        if classification["decision"] != "accepted" or not profile_path.is_file():
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        triangle_count = int(profile.get("triangle_count", -1))
        mesh_count = int(profile.get("mesh_count", -1))
        if (profile.get("status") == "complete" and
                config.catalog.minimum_faces <= triangle_count <= config.catalog.maximum_faces and
                0 < mesh_count <= config.catalog.maximum_components):
            usable.append({**record, "profile_data": profile})
    observations = discover_observations(observation_root)
    jobs = []
    observation_root = observation_root.resolve()
    for observation in observations:
        relative_observation = observation.directory.relative_to(observation_root)
        final_dir = output_root / relative_observation
        if (final_dir / "combined_scene.glb").is_file() and not force:
            continue
        support_path = final_dir / "support_patches.json"
        patches = extract_support_patches(
            observation.directory, segmentation_config, config, support_path
        )
        compatible = []
        support_names = {str(patch["semantic_class"]) for patch in patches}
        for record in usable:
            if support_names & set(record["classification"]["support_classes"]):
                compatible.append(record)
        seed = config.placement.seed ^ int(stable_digest({
            "observation": observation.observation_id,
        })[:16], 16)
        random.Random(seed).shuffle(compatible)
        manifest = json.loads(observation.manifest.read_text(encoding="utf-8"))
        placement_config = config.to_dict()["placement"]
        placement_config["support_boundary_margin_m"] = config.support.boundary_margin_m
        jobs.append({
            "observation_id": observation.observation_id,
            "partition_glb": str(observation.geometry), "observation": manifest,
            "supports": patches, "objects": compatible[:config.placement.max_object_trials],
            "output": str(final_dir), "seed": seed,
            "placement": placement_config,
            "normalization": config.to_dict()["object_normalization"],
        })
    print(json.dumps({"observations": len(observations), "jobs": len(jobs),
                      "usable_objects": len(usable)}, indent=2))
    if dry_run:
        return 0
    executable = shutil.which(blender) or (blender if Path(blender).is_file() else None)
    if not executable:
        raise ValueError(f"Blender executable not found: {blender}")
    failures = 0
    script = PROJECT_ROOT / "scripts" / "blender" / "compose_observation.py"
    job_root = output_root / "jobs"
    for index, job in enumerate(jobs):
        job_path = job_root / f"compose_{index:08d}.json"
        write_json_atomic(job_path, job)
        completed = subprocess.run([
            str(executable), "--background", "--factory-startup", "--python", str(script),
            "--", "--job", str(job_path),
        ], cwd=PROJECT_ROOT, check=False)
        failures += completed.returncode != 0
    write_json_atomic(output_root / "composition_summary.json", {
        "schema_version": 1, "job_count": len(jobs), "failed_count": failures,
        "config_digest": config.digest(),
    })
    return 1 if failures else 0
