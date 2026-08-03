from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


STAGES = ("geometry", "views", "mosaic3d", "sam3", "fusion", "export")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "missing": True}
    return file_fingerprint(path)


class SegmentationManifest:
    def __init__(self, path: Path, scene_id: str, source: Path, relative_source: str) -> None:
        self.path = path
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            current_source = file_fingerprint(source)
            if self.data.get("source_fingerprint") != current_source:
                self.data["source_fingerprint"] = current_source
                self.data["stages"] = {name: {"status": "pending"} for name in STAGES}
        else:
            self.data = {
                "schema_version": 1,
                "scene_id": scene_id,
                "source": relative_source,
                "source_fingerprint": file_fingerprint(source),
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "warnings": [],
                "stages": {name: {"status": "pending"} for name in STAGES},
            }

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = utc_now()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def can_resume(self, stage: str, config_digest: str, upstream_digest: str | None = None) -> bool:
        state = self.data["stages"][stage]
        return (
            state.get("status") == "complete"
            and state.get("config_digest") == config_digest
            and state.get("upstream_digest") == upstream_digest
        )

    def invalidate_from(self, stage: str) -> None:
        index = STAGES.index(stage)
        for name in STAGES[index:]:
            self.data["stages"][name] = {"status": "pending"}
        self.write()

    @contextmanager
    def running(self, stage: str, config_digest: str, upstream_digest: str | None, log_path: Path) -> Iterator[dict[str, Any]]:
        state = {
            "status": "running", "started_at": utc_now(),
            "config_digest": config_digest, "upstream_digest": upstream_digest,
            "log": str(log_path),
        }
        self.data["stages"][stage] = state
        self.write()
        try:
            yield state
        except Exception as error:
            state.update(status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
            self.write()
            raise
        else:
            state.update(status="complete", finished_at=utc_now())
            self.write()

    def complete_artifacts(self, stage: str, paths: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in sorted(paths, key=lambda item: item.as_posix()):
            stat = path.stat()
            digest.update(path.name.encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
        value = digest.hexdigest()
        self.data["stages"][stage]["artifact_digest"] = value
        return value
