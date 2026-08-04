from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .config import CompositionConfig
from .contracts import Classification, file_signature, write_json_atomic
from .discovery import discover_objects
from .llm import classify_with_local_llm
from .rules import classify_with_rules


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_object_catalog(
    object_root: Path, metadata: Path, output: Path, config_path: Path,
    blender: str, force: bool = False, dry_run: bool = False,
) -> int:
    config = CompositionConfig.from_json(config_path)
    assets = discover_objects(object_root, metadata)
    output.mkdir(parents=True, exist_ok=True)
    rule_results = {}
    for asset in assets:
        archive = asset.annotation.get("archives", {}).get("glb", {})
        face_count = archive.get("faceCount") if isinstance(archive, dict) else None
        if isinstance(face_count, int) and not (
            config.catalog.minimum_faces <= face_count <= config.catalog.maximum_faces
        ):
            rule_results[asset.uid] = Classification(
                "rejected", None, None, (), None, None, None, 1.0,
                "geometry_metadata", f"GLB face count outside configured limits: {face_count}",
            )
        else:
            rule_results[asset.uid] = classify_with_rules(asset, config)
    unresolved = [asset for asset in assets if rule_results[asset.uid].decision == "needs_review"]
    llm_results = classify_with_local_llm(
        unresolved, config, output / "llm_classification_cache.json"
    ) if not dry_run else {}
    records = []
    profile_jobs = []
    for asset in assets:
        classification = llm_results.get(asset.uid, rule_results[asset.uid])
        profile_path = output / "profiles" / f"{asset.uid}.json"
        records.append({
            "uid": asset.uid, "glb": str(asset.glb_path),
            "relative_path": asset.relative_path,
            "classification": classification.to_dict(),
            "profile": str(profile_path), "file_signature": file_signature(asset.glb_path),
        })
        if classification.decision == "accepted" and (force or not profile_path.is_file()):
            profile_jobs.append({"uid": asset.uid, "glb": str(asset.glb_path),
                                 "output": str(profile_path)})
    write_json_atomic(output / "catalog.pending.json", {
        "schema_version": 1, "config_digest": config.digest(), "records": records,
        "profile_job_count": len(profile_jobs),
    })
    if dry_run:
        print(json.dumps({"objects": len(records), "profiles_pending": len(profile_jobs)}, indent=2))
        return 0
    executable = shutil.which(blender) or (blender if Path(blender).is_file() else None)
    if not executable:
        raise ValueError(f"Blender executable not found: {blender}")
    script = PROJECT_ROOT / "scripts" / "blender" / "profile_objaverse_batch.py"
    commands = []
    for start in range(0, len(profile_jobs), config.catalog.blender_batch_size):
        job_path = output / "jobs" / f"profile_{start:08d}.json"
        write_json_atomic(job_path, {"jobs": profile_jobs[start:start + config.catalog.blender_batch_size]})
        commands.append([
            str(executable), "--background", "--factory-startup", "--python", str(script),
            "--", "--job", str(job_path),
        ])
    def run(command: list[str]) -> int:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    with ThreadPoolExecutor(max_workers=config.catalog.workers) as executor:
        return_codes = list(executor.map(run, commands))
    if any(return_codes):
        return next(code for code in return_codes if code)
    for record in records:
        profile = Path(record["profile"])
        record["profile_status"] = "complete" if profile.is_file() else "not_required"
    write_json_atomic(output / "catalog.json", {
        "schema_version": 1, "config_digest": config.digest(), "records": records,
        "accepted_count": sum(r["classification"]["decision"] == "accepted" for r in records),
    })
    return 0
