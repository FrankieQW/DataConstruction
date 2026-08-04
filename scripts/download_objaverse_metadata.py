#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gzip
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
SHARD_PATTERN = re.compile(
    r"^(?P<shard>[0-9]{3}-[0-9]{3})/(?P<uid>[^/\\]+)\.(?P<extension>glb)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class ObjectRecord:
    glb_relative_path: str
    shard: str
    uid: str


@dataclass
class CacheStats:
    cache_hits: int = 0
    downloaded: int = 0


def parse_manifest(path: Path) -> tuple[ObjectRecord, ...]:
    if not path.is_file():
        raise ValueError(f"Manifest not found: {path}")
    records_by_uid: dict[str, ObjectRecord] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        candidate = raw_line.strip().strip("`").replace("\\", "/")
        match = SHARD_PATTERN.fullmatch(candidate)
        if match is None:
            continue
        shard = match.group("shard")
        uid = match.group("uid")
        relative = PurePosixPath(shard, f"{uid}.glb").as_posix()
        record = ObjectRecord(relative, shard, uid)
        previous = records_by_uid.get(uid)
        if previous is not None and previous.shard != shard:
            raise ValueError(
                f"UID appears in multiple shards: {uid} ({previous.shard}, {shard})"
            )
        records_by_uid[uid] = record
    if not records_by_uid:
        raise ValueError(f"Manifest contains no '<shard>/<uid>.glb' records: {path}")
    return tuple(sorted(records_by_uid.values()))


def group_by_shard(records: Iterable[ObjectRecord]) -> dict[str, tuple[ObjectRecord, ...]]:
    grouped: dict[str, list[ObjectRecord]] = {}
    for record in records:
        grouped.setdefault(record.shard, []).append(record)
    return {
        shard: tuple(sorted(items))
        for shard, items in sorted(grouped.items())
    }


def load_gzip_json(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Invalid gzip JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Gzip JSON root must be an object: {path}")
    return payload


def _download(url: str, destination: Path, timeout: float) -> None:
    request = Request(url, headers={"User-Agent": "SceneCompose-Objaverse-Metadata/1"})
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def ensure_cached_gzip_json(
    *,
    url: str,
    cache_path: Path,
    timeout: float,
    retries: int,
    force_download: bool,
    stats: CacheStats,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_download:
        try:
            payload = load_gzip_json(cache_path)
        except ValueError:
            invalid = cache_path.with_name(
                f"{cache_path.name}.invalid-{uuid.uuid4().hex}"
            )
            cache_path.replace(invalid)
            print(f"[INVALID] {cache_path.name} -> {invalid.name}", flush=True)
        else:
            stats.cache_hits += 1
            return payload

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary = cache_path.with_name(
            f".{cache_path.name}.part-{uuid.uuid4().hex}"
        )
        try:
            _download(url, temporary, timeout)
            payload = load_gzip_json(temporary)
            temporary.replace(cache_path)
            stats.downloaded += 1
            return payload
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                delay = min(8.0, float(2 ** (attempt - 1)))
                print(
                    f"[RETRY {attempt}/{retries}] {cache_path.name}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Failed to download valid metadata after {retries} attempts: {url}: {last_error}"
    )


def _join_url(base_url: str, relative: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(relative, safe='/')}"


def extract_annotations(
    grouped: dict[str, tuple[ObjectRecord, ...]],
    *,
    output: Path,
    base_url: str,
    timeout: float,
    retries: int,
    force_download: bool,
    stats: CacheStats,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    annotations: dict[str, dict[str, Any]] = {}
    missing: set[str] = set()
    shard_count = len(grouped)
    for index, (shard, records) in enumerate(grouped.items(), start=1):
        cache_path = output / "cache" / "metadata" / f"{shard}.json.gz"
        cache_hits_before = stats.cache_hits
        payload = ensure_cached_gzip_json(
            url=_join_url(base_url, f"metadata/{shard}.json.gz"),
            cache_path=cache_path,
            timeout=timeout,
            retries=retries,
            force_download=force_download,
            stats=stats,
        )
        state = "CACHE" if stats.cache_hits > cache_hits_before else "DOWNLOAD"
        print(f"[{index:03d}/{shard_count:03d}] [{state}] {shard}", flush=True)
        for record in records:
            raw_annotation = payload.get(record.uid)
            if raw_annotation is None:
                missing.add(record.uid)
                continue
            if not isinstance(raw_annotation, dict):
                raise ValueError(
                    f"Annotation must be a JSON object: shard={shard}, uid={record.uid}"
                )
            if "_scenecompose" in raw_annotation:
                raise ValueError(
                    f"Official annotation uses reserved key '_scenecompose': {record.uid}"
                )
            annotations[record.uid] = copy.deepcopy(raw_annotation)
    return annotations, missing


def invert_lvis_annotations(
    payload: dict[str, Any], requested_uids: set[str]
) -> dict[str, tuple[str, ...]]:
    categories: dict[str, set[str]] = {}
    for category, values in payload.items():
        if not isinstance(category, str) or not isinstance(values, list):
            raise ValueError("LVIS annotations must map category strings to UID arrays")
        for uid in values:
            if isinstance(uid, str) and uid in requested_uids:
                categories.setdefault(uid, set()).add(category)
    return {
        uid: tuple(sorted(values, key=str.casefold))
        for uid, values in sorted(categories.items())
    }


def build_outputs(
    records: tuple[ObjectRecord, ...],
    annotations: dict[str, dict[str, Any]],
    missing: set[str],
    lvis_categories: dict[str, tuple[str, ...]],
    *,
    manifest: Path,
    output: Path,
    sample_size: int,
    stats: CacheStats,
    lvis_skipped: bool,
) -> dict[str, object]:
    enriched: dict[str, dict[str, Any]] = {}
    jsonl_records: list[dict[str, Any]] = []
    for record in records:
        if record.uid not in annotations:
            continue
        annotation = copy.deepcopy(annotations[record.uid])
        annotation["_scenecompose"] = {
            "uid": record.uid,
            "shard": record.shard,
            "glb_relative_path": record.glb_relative_path,
            "lvis_categories": list(lvis_categories.get(record.uid, ())),
        }
        enriched[record.uid] = annotation
        jsonl_record = copy.deepcopy(annotation)
        jsonl_record["uid"] = record.uid
        jsonl_records.append(jsonl_record)

    lvis_output = {
        uid: list(lvis_categories[uid])
        for uid in sorted(lvis_categories)
        if uid in enriched
    }
    artifact_names = (
        "annotations.json", "annotations.jsonl", "sample.json",
        "lvis_categories.json", "missing_uids.txt", "summary.json",
    )
    summary = {
        "schema_version": 1,
        "manifest": str(manifest.resolve()),
        "output": str(output.resolve()),
        "requested_object_count": len(records),
        "shard_count": len({record.shard for record in records}),
        "annotation_count": len(enriched),
        "missing_count": len(missing),
        "lvis_match_count": len(lvis_output),
        "lvis_skipped": lvis_skipped,
        "cache_hit_count": stats.cache_hits,
        "downloaded_file_count": stats.downloaded,
        "artifacts": list(artifact_names),
        "warnings": (
            [f"{len(missing)} requested UIDs were not found in official metadata"]
            if missing else []
        ),
    }
    return {
        "annotations": enriched,
        "jsonl_records": jsonl_records,
        "sample": jsonl_records[:sample_size],
        "lvis_categories": lvis_output,
        "missing": [record.uid for record in records if record.uid in missing],
        "summary": summary,
    }


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_outputs_atomic(output: Path, payloads: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    serialized = {
        "annotations.json": _json_bytes(payloads["annotations"]),
        "annotations.jsonl": _jsonl_bytes(payloads["jsonl_records"]),
        "sample.json": _json_bytes(payloads["sample"]),
        "lvis_categories.json": _json_bytes(payloads["lvis_categories"]),
        "missing_uids.txt": (
            ("\n".join(payloads["missing"]) + "\n") if payloads["missing"] else ""
        ).encode("utf-8"),
        "summary.json": _json_bytes(payloads["summary"]),
    }
    temporary_paths: dict[str, Path] = {}
    try:
        for name, content in serialized.items():
            temporary = output / f".{name}.tmp-{uuid.uuid4().hex}"
            temporary.write_bytes(content)
            temporary_paths[name] = temporary
        for name in serialized:
            temporary_paths[name].replace(output / name)
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Objaverse metadata for GLBs listed in a manifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "obj" / "metadata"
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--skip-lvis", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if not str(args.base_url).strip():
        parser.error("--base-url must not be empty")
    return args


def run(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    records = parse_manifest(manifest)
    grouped = group_by_shard(records)
    stats = CacheStats()
    print(
        f"Manifest: {len(records)} objects across {len(grouped)} metadata shards",
        flush=True,
    )
    annotations, missing = extract_annotations(
        grouped,
        output=output,
        base_url=args.base_url,
        timeout=args.timeout,
        retries=args.retries,
        force_download=args.force_download,
        stats=stats,
    )
    requested_uids = {record.uid for record in records}
    if args.skip_lvis:
        lvis_categories: dict[str, tuple[str, ...]] = {}
    else:
        lvis_payload = ensure_cached_gzip_json(
            url=_join_url(args.base_url, "lvis-annotations.json.gz"),
            cache_path=output / "cache" / "lvis-annotations.json.gz",
            timeout=args.timeout,
            retries=args.retries,
            force_download=args.force_download,
            stats=stats,
        )
        lvis_categories = invert_lvis_annotations(lvis_payload, requested_uids)
    payloads = build_outputs(
        records, annotations, missing, lvis_categories,
        manifest=manifest, output=output, sample_size=args.sample_size,
        stats=stats, lvis_skipped=args.skip_lvis,
    )
    write_outputs_atomic(output, payloads)
    print(
        "Complete: "
        f"annotations={len(annotations)}, missing={len(missing)}, "
        f"lvis_matches={len(lvis_categories)}, cache_hits={stats.cache_hits}, "
        f"downloads={stats.downloaded}",
        flush=True,
    )
    if missing and not args.allow_missing:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_arguments(argv))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
