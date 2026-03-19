#!/usr/bin/env python3
"""
Folder-based surveillance automation built on top of CodeProject.AI.

What it does:
  - watches a drop folder for new video files
  - waits until a file has finished copying
  - processes one video at a time through the existing analyzer
  - writes clip-first outputs with a small summary instead of a large HTML report
  - maintains a persistent cross-video people library using face similarity
  - moves originals into processed or failed folders

The watcher implementation borrows the same debounce/watch approach used by the
watchfiles project, while the people-first browsing model is intentionally
inspired by Immich's "person with thumbnail and sightings" workflow.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import requests
from watchfiles import Change, watch

from analyze_videos import (
    CPAI_URL,
    FACE_CLUSTER_THRESHOLD,
    VIDEO_EXTENSIONS,
    FaceClusterer,
    face_similarity,
    process_video,
)

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
DRIVE_REMOTE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a folder and auto-process dropped surveillance videos"
    )
    parser.add_argument(
        "watch_dir",
        help="Folder to watch for new MP4/MOV/AVI/MKV videos",
    )
    parser.add_argument(
        "--workspace",
        default="automation_workspace",
        help="Workspace folder for reports, people library, processing, and archives",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Watch subfolders recursively",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process files currently in the watch folder and exit",
    )
    parser.add_argument(
        "--settle-seconds",
        type=int,
        default=12,
        help="Require file size and mtime to remain unchanged for this many seconds before processing",
    )
    parser.add_argument(
        "--settle-timeout",
        type=int,
        default=1800,
        help="Maximum seconds to wait for a newly dropped file to finish copying",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=90,
        help="Analyze every N frames",
    )
    parser.add_argument(
        "--scene-stride",
        type=int,
        default=0,
        help="Classify scene every N frames (0 uses stride)",
    )
    parser.add_argument(
        "--min-face",
        type=int,
        default=40,
        help="Skip faces smaller than this size in pixels",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum confidence threshold for faces and objects",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=FACE_CLUSTER_THRESHOLD,
        help="Similarity threshold used both for local clustering and persistent people matching",
    )
    parser.add_argument(
        "--activity-gap",
        type=int,
        default=20,
        help="Seconds of inactivity that split clips",
    )
    parser.add_argument(
        "--min-activity",
        type=int,
        default=3,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--no-objects",
        action="store_true",
        help="Skip object detection",
    )
    parser.add_argument(
        "--no-alpr",
        action="store_true",
        help="Skip license plate recognition",
    )
    parser.add_argument(
        "--no-scene",
        action="store_true",
        help="Skip scene classification",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        help="Skip clip extraction",
    )
    parser.add_argument(
        "--force-polling",
        action="store_true",
        help="Force polling in watchfiles instead of native notifications",
    )
    parser.add_argument(
        "--poll-delay-ms",
        type=int,
        default=500,
        help="Polling delay in milliseconds if force-polling is enabled",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "item"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "workspace": root,
        "processing": root / "processing",
        "processed_originals": root / "processed_originals",
        "failed": root / "failed",
        "reports": root / "reports",
        "people": root / "people",
        "state": root / "state",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def is_network_share_path(path: Path) -> bool:
    path_str = str(path)
    if path_str.startswith("\\\\"):
        return True

    anchor = path.anchor or os.path.splitdrive(path_str)[0]
    if not anchor:
        return False

    try:
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(anchor)
        return drive_type == DRIVE_REMOTE
    except Exception:
        return False


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def collect_candidates(watch_dir: Path, recursive: bool) -> list[Path]:
    iterator = watch_dir.rglob("*") if recursive else watch_dir.iterdir()
    return sorted(path for path in iterator if is_video_file(path))


def collect_processing_candidates(processing_dir: Path) -> list[Path]:
    if not processing_dir.exists():
        return []
    return sorted(path for path in processing_dir.iterdir() if is_video_file(path))


def wait_for_file_ready(path: Path, stable_seconds: int, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    last_sig: tuple[int, int] | None = None
    stable_since: float | None = None

    while time.time() < deadline:
        if not path.exists() or not path.is_file():
            return False

        try:
            stat = path.stat()
        except OSError:
            time.sleep(1)
            continue

        # Existing backlog files that have not changed for longer than the
        # settle window can be accepted immediately.
        if stat.st_size > 0 and (time.time() - stat.st_mtime) >= stable_seconds:
            try:
                with open(path, "rb"):
                    return True
            except OSError:
                pass

        signature = (stat.st_size, stat.st_mtime_ns)
        if stat.st_size > 0 and signature == last_sig:
            stable_since = stable_since or time.time()
            if (time.time() - stable_since) >= stable_seconds:
                try:
                    with open(path, "rb"):
                        return True
                except OSError:
                    pass
        else:
            last_sig = signature
            stable_since = None

        time.sleep(2)

    return False


def ensure_cpai_available() -> None:
    response = requests.get(f"{CPAI_URL}/v1/server/status/ping", timeout=10)
    payload = response.json()
    if response.status_code != 200 or not payload.get("success"):
        raise RuntimeError("CodeProject.AI did not respond successfully")


def ffprobe_creation_time(video_path: Path) -> datetime | None:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=creation_time:stream_tags=creation_time",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        payload = json.loads(result.stdout)
    except Exception:
        return None

    candidates: list[str] = []
    fmt = payload.get("format") or {}
    tags = fmt.get("tags") or {}
    if tags.get("creation_time"):
        candidates.append(tags["creation_time"])
    for stream in payload.get("streams", []):
        stream_tags = stream.get("tags") or {}
        if stream_tags.get("creation_time"):
            candidates.append(stream_tags["creation_time"])

    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
        except ValueError:
            continue
    return None


def derive_video_time(video_path: Path) -> datetime:
    return ffprobe_creation_time(video_path) or datetime.fromtimestamp(video_path.stat().st_mtime)


def build_run_name(video_path: Path, video_time: datetime) -> str:
    return f"{video_time.strftime(TIME_FORMAT)}__{slugify(video_path.stem)}"


def source_fingerprint(video_path: Path) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "name": video_path.name,
        "path": str(video_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_failure_fingerprint(failure_dir: Path) -> dict[str, Any] | None:
    meta_path = failure_dir / "source.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def should_skip_failed_source(video_path: Path, failed_root: Path) -> str | None:
    current = source_fingerprint(video_path)
    slug = slugify(video_path.stem)
    for failure_dir in sorted(failed_root.glob(f"*__{slug}")):
        if not failure_dir.is_dir():
            continue
        previous = load_failure_fingerprint(failure_dir)
        if previous is not None:
            if (
                previous.get("name") == current["name"]
                and previous.get("size") == current["size"]
                and previous.get("mtime_ns") == current["mtime_ns"]
            ):
                return failure_dir.name
            continue

        failed_copy = failure_dir / video_path.name
        if failed_copy.exists():
            try:
                failed_stat = failed_copy.stat()
            except OSError:
                continue
            if failed_stat.st_size == current["size"]:
                return failure_dir.name
        for candidate in failure_dir.iterdir():
            if not is_video_file(candidate):
                continue
            try:
                failed_stat = candidate.stat()
            except OSError:
                continue
            if failed_stat.st_size == current["size"]:
                return failure_dir.name
    return None


def analyzer_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=str(output_dir),
        stride=args.stride,
        min_face=args.min_face,
        confidence=args.confidence,
        face_threshold=args.face_threshold,
        no_objects=args.no_objects,
        no_alpr=args.no_alpr,
        no_scene=args.no_scene,
        no_clips=args.no_clips,
        activity_gap=args.activity_gap,
        min_activity=args.min_activity,
        scene_stride=args.scene_stride,
    )


def best_crop(cluster_dir: Path) -> Path | None:
    crops = sorted(cluster_dir.glob("*.jpg"))
    if not crops:
        return None
    return max(crops, key=lambda path: path.stat().st_size)


def load_people_index(people_root: Path) -> dict[str, Any]:
    index_path = people_root / "index.json"
    if not index_path.exists():
        return {"next_person_id": 1, "people": {}}
    with open(index_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_people_index(people_root: Path, index: dict[str, Any]) -> None:
    with open(people_root / "index.json", "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)


def match_person(representative: Path, people_root: Path, index: dict[str, Any], threshold: float) -> tuple[str | None, float]:
    crop = cv2.imread(str(representative))
    if crop is None:
        return None, 0.0

    best_person_id: str | None = None
    best_score = 0.0
    for person_id, person in index["people"].items():
        thumbnail_rel = person.get("thumbnail")
        if not thumbnail_rel:
            continue
        thumbnail_path = people_root / thumbnail_rel
        thumb = cv2.imread(str(thumbnail_path)) if thumbnail_path.exists() else None
        if thumb is None:
            continue
        score = face_similarity(crop, thumb)
        if score > best_score:
            best_score = score
            best_person_id = person_id

    if best_score >= threshold:
        return best_person_id, best_score
    return None, best_score


def sync_people_library(
    local_faces_dir: Path,
    people_root: Path,
    source_video: str,
    source_time: datetime,
    threshold: float,
) -> dict[str, str]:
    index = load_people_index(people_root)
    mapping: dict[str, str] = {}

    if not local_faces_dir.exists():
        return mapping

    for cluster_dir in sorted(path for path in local_faces_dir.iterdir() if path.is_dir()):
        representative = best_crop(cluster_dir)
        if representative is None:
            continue

        person_id, similarity = match_person(representative, people_root, index, threshold)
        now_iso = source_time.isoformat(timespec="seconds")
        if person_id is None:
            person_id = f"person_{index['next_person_id']:04d}"
            index["next_person_id"] += 1
            index["people"][person_id] = {
                "id": person_id,
                "created_at": now_iso,
                "first_seen": now_iso,
                "last_seen": now_iso,
                "face_count": 0,
                "videos": [],
                "thumbnail": f"{person_id}/thumbnail.jpg",
                "samples": [],
                "best_similarity": 1.0,
            }

        person = index["people"][person_id]
        person_dir = people_root / person_id
        samples_dir = person_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        thumbnail_path = people_root / person["thumbnail"]
        if not thumbnail_path.exists() or similarity > float(person.get("best_similarity", 0.0)):
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(representative, thumbnail_path)
            person["best_similarity"] = similarity

        crop_files = sorted(cluster_dir.glob("*.jpg"))
        copied_count = 0
        for crop in crop_files:
            if copied_count >= 5:
                break
            dest = unique_path(samples_dir / f"{slugify(Path(source_video).stem)}__{crop.name}")
            shutil.copy2(crop, dest)
            person["samples"].append(str(dest.relative_to(people_root)).replace("\\", "/"))
            copied_count += 1

        person["face_count"] += len(crop_files)
        person["last_seen"] = now_iso
        person["first_seen"] = min(person.get("first_seen", now_iso), now_iso)
        if source_video not in person["videos"]:
            person["videos"].append(source_video)

        mapping[cluster_dir.name] = person_id

    save_people_index(people_root, index)
    return mapping


def save_detections_csv(csv_path: Path, detections: list[dict[str, Any]]) -> None:
    if not detections:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detections[0].keys()))
        writer.writeheader()
        writer.writerows(detections)


def summarize_clip(
    clip: dict[str, Any],
    detections: list[dict[str, Any]],
    local_to_global: dict[str, str],
) -> dict[str, Any]:
    start_sec = float(clip["start_sec"])
    end_sec = float(clip["end_sec"])
    relevant = [
        detection
        for detection in detections
        if start_sec <= float(detection["timestamp_sec"]) <= end_sec
    ]

    object_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    people: list[str] = []
    plates: list[str] = []

    for detection in relevant:
        dtype = detection["type"]
        if dtype == "face":
            people.append(local_to_global.get(detection["label"], detection["label"]))
        elif dtype == "object":
            object_counts[slugify(str(detection["label"]))] += 1
        elif dtype == "plate":
            plates.append(slugify(str(detection["label"])))
        elif dtype == "scene":
            scene_counts[slugify(str(detection["label"]))] += 1

    tags: list[str] = []
    unique_people = sorted(set(people))
    if unique_people:
        tags.extend(unique_people[:2])
        object_counts.pop("person", None)
    elif object_counts.get("person"):
        tags.append("person")
        object_counts.pop("person", None)

    for label, _count in object_counts.most_common(2):
        if label not in tags:
            tags.append(label)

    if plates:
        tags.append("plate")

    if not tags and scene_counts:
        tags.append(scene_counts.most_common(1)[0][0])

    if not tags:
        tags.append("activity")

    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "tags": tags,
        "people": unique_people,
        "plates": sorted(set(plates)),
        "objects": [label for label, _count in object_counts.most_common(5)],
    }


def rename_clips(
    report_dir: Path,
    clip_entries: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    local_to_global: dict[str, str],
    source_time: datetime,
) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for clip in clip_entries:
        clip_path = Path(clip["path"])
        if not clip_path.exists():
            continue

        clip_summary = summarize_clip(clip, detections, local_to_global)
        clip_time = source_time + timedelta(seconds=float(clip["start_sec"]))
        base_name = f"{clip_time.strftime(TIME_FORMAT)}__{'__'.join(clip_summary['tags'][:4])}"
        new_path = unique_path(clip_path.with_name(f"{base_name}{clip_path.suffix}"))
        clip_path.rename(new_path)

        renamed.append({
            "path": str(new_path),
            "relative_path": str(new_path.relative_to(report_dir)).replace("\\", "/"),
            "video": clip["video"],
            **clip_summary,
        })
    return renamed


def write_summary(
    report_dir: Path,
    source_video: Path,
    processed_original: Path,
    detections: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    local_to_global: dict[str, str],
    source_time: datetime,
    elapsed_sec: float,
) -> None:
    face_labels = []
    for detection in detections:
        if detection["type"] == "face":
            detection["label"] = local_to_global.get(detection["label"], detection["label"])
            face_labels.append(detection["label"])

    object_counts = Counter(d["label"] for d in detections if d["type"] == "object")
    plate_values = sorted({d["label"] for d in detections if d["type"] == "plate"})
    person_ids = sorted(set(face_labels))

    summary = {
        "source_video": source_video.name,
        "source_video_path": str(source_video),
        "processed_original_path": str(processed_original),
        "source_time": source_time.isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed_sec, 1),
        "counts": {
            "detections": len(detections),
            "faces": sum(1 for d in detections if d["type"] == "face"),
            "people": len(person_ids),
            "objects": sum(1 for d in detections if d["type"] == "object"),
            "plates": sum(1 for d in detections if d["type"] == "plate"),
            "clips": len(clips),
        },
        "people": person_ids,
        "plates": plate_values,
        "top_objects": object_counts.most_common(10),
        "clips": clips,
    }

    summary_json = report_dir / "summary.json"
    summary_txt = report_dir / "summary.txt"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        f"video: {source_video.name}",
        f"source time: {source_time.strftime(TIME_FORMAT)}",
        f"processed original: {processed_original}",
        f"elapsed sec: {round(elapsed_sec, 1)}",
        f"people: {', '.join(person_ids) if person_ids else 'none'}",
        f"plates: {', '.join(plate_values) if plate_values else 'none'}",
        f"top objects: {', '.join(f'{name} x{count}' for name, count in object_counts.most_common(5)) or 'none'}",
        f"clips: {len(clips)}",
    ]
    for clip in clips:
        lines.append(
            f"  - {clip['relative_path']} [{clip['start_sec']:.1f}s - {clip['end_sec']:.1f}s] tags={', '.join(clip['tags'])}"
        )

    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    save_detections_csv(report_dir / "detections.csv", detections)


def move_with_unique_name(source: Path, destination_dir: Path, new_name: str | None = None) -> Path:
    destination = destination_dir / (new_name or source.name)
    destination = unique_path(destination)
    shutil.move(str(source), str(destination))
    return destination


def ensure_video_readable(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total_frames <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Unreadable or incomplete video metadata: {video_path}")
    finally:
        cap.release()


def process_video_file(video_path: Path, args: argparse.Namespace, paths: dict[str, Path]) -> Path:
    ensure_cpai_available()

    source_time = derive_video_time(video_path)
    run_name = build_run_name(video_path, source_time)
    if video_path.parent == paths["processing"]:
        processing_path = video_path
    else:
        processing_path = move_with_unique_name(
            video_path,
            paths["processing"],
            f"{run_name}{video_path.suffix.lower()}",
        )
    report_dir = paths["reports"] / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    analyzer = analyzer_args(args, report_dir)
    detections: list[dict[str, Any]] = []
    clusterer = FaceClusterer(threshold=args.face_threshold)

    try:
        ensure_video_readable(processing_path)
        clips = process_video(processing_path, analyzer, report_dir, clusterer, detections)
        local_to_global = sync_people_library(
            report_dir / "faces",
            paths["people"],
            processing_path.name,
            source_time,
            args.face_threshold,
        )
        clips = rename_clips(report_dir, clips, detections, local_to_global, source_time)
        processed_original = move_with_unique_name(processing_path, paths["processed_originals"])
        elapsed_sec = time.time() - t0
        write_summary(
            report_dir,
            video_path,
            processed_original,
            detections,
            clips,
            local_to_global,
            source_time,
            elapsed_sec,
        )
        return report_dir
    except Exception:
        failure_dir = paths["failed"] / run_name
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_target = unique_path(failure_dir / processing_path.name)
        if processing_path.exists():
            shutil.move(str(processing_path), str(failure_target))
        with open(failure_dir / "source.json", "w", encoding="utf-8") as handle:
            json.dump(source_fingerprint(video_path), handle, indent=2)
        with open(failure_dir / "error.txt", "w", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise


def drain_pending(pending: set[Path], args: argparse.Namespace, paths: dict[str, Path]) -> None:
    ready: list[Path] = []
    for path in sorted(pending):
        if not path.exists() or not is_video_file(path):
            ready.append(path)
            continue
        if path.parent != paths["processing"]:
            existing_failure = should_skip_failed_source(path, paths["failed"])
            if existing_failure is not None:
                print(f"\nSkipping previously failed source {path} -> {existing_failure}")
                ready.append(path)
                continue
        if wait_for_file_ready(path, args.settle_seconds, args.settle_timeout):
            ready.append(path)

    for path in ready:
        pending.discard(path)
        if not path.exists() or not is_video_file(path):
            continue
        print(f"\nProcessing {path}")
        try:
            report_dir = process_video_file(path, args, paths)
            print(f"  Done -> {report_dir}")
        except Exception as exc:
            print(f"  Failed -> {path}: {exc}")


def main() -> int:
    args = parse_args()
    watch_dir = Path(args.watch_dir).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    paths = ensure_dirs(workspace)

    if not watch_dir.exists() or not watch_dir.is_dir():
        print(f"Watch folder does not exist: {watch_dir}")
        return 1

    if is_network_share_path(watch_dir):
        if not args.force_polling:
            args.force_polling = True
            print(f"Network share detected at {watch_dir}; enabling polling watcher")
        if args.settle_seconds < 30:
            args.settle_seconds = 30
            print(f"Network share detected at {watch_dir}; using settle_seconds={args.settle_seconds}")

    try:
        ensure_cpai_available()
    except Exception as exc:
        print(f"Cannot reach CodeProject.AI at {CPAI_URL}: {exc}")
        return 1

    processing_pending = set(collect_processing_candidates(paths["processing"]))
    if processing_pending:
        print(f"Resuming {len(processing_pending)} processing file(s)")
        drain_pending(processing_pending, args, paths)

    pending = set(collect_candidates(watch_dir, args.recursive))
    if pending:
        print(f"Found {len(pending)} existing video(s) to process")
        drain_pending(pending, args, paths)

    if args.once:
        return 0

    stop_event = threading.Event()
    print(f"Watching {watch_dir}")
    try:
        for changes in watch(
            str(watch_dir),
            recursive=args.recursive,
            debounce=2000,
            step=250,
            stop_event=stop_event,
            yield_on_timeout=True,
            force_polling=args.force_polling,
            poll_delay_ms=args.poll_delay_ms,
            watch_filter=None,
        ):
            for change, changed_path in changes:
                if change in {Change.added, Change.modified}:
                    candidate = Path(changed_path)
                    if is_video_file(candidate):
                        pending.add(candidate)
            if pending:
                drain_pending(pending, args, paths)
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping watcher")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())