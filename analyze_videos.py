#!/usr/bin/env python3
"""
Surveillance Video Analysis Pipeline
=====================================
Uses CodeProject.AI Server (http://localhost:32168) for:
  - Face detection + greedy similarity clustering (person re-identification)
  - Person / vehicle / object detection
  - License plate recognition (ALPR)
  - Scene classification

Produces per-run in --output directory:
  detections.csv         all per-frame detections with timestamps
  faces/person_NNN/      cropped face images per identity
  clips/                 condensed mp4 clips of activity (requires ffmpeg)
  report.json            machine-readable summary
  report.html            browsable HTML summary with face grid + stats

Usage examples:
  python analyze_videos.py D:\\PRIVATE\\M4ROOT\\CLIP --output C:\\CoRoot\\analysis
  python analyze_videos.py myvideo.mp4 --output out --stride 90 --no-alpr --no-scene
  python analyze_videos.py D:\\clips --output out --no-clips   # skip clip extraction
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
CPAI_URL = "http://localhost:32168"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".m2ts", ".ts"}
FACE_CLUSTER_THRESHOLD = 0.75   # CPAI similarity >= this → same person
ACTIVITY_LABELS = {             # object labels that mark "active" segments
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
}


# ── CLI ─────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Surveillance video analysis using CodeProject.AI"
    )
    p.add_argument("input", nargs="+",
                   help="Video file(s) or directory of videos")
    p.add_argument("--output", "-o", default="output",
                   help="Output folder (created if absent)")
    p.add_argument("--stride", type=int, default=60,
                   help="Analyse every N frames (default 60 ≈ 2s @ 30fps)")
    p.add_argument("--min-face", type=int, default=40,
                   help="Skip faces smaller than this (pixels)")
    p.add_argument("--confidence", type=float, default=0.45,
                   help="Minimum detection confidence 0-1")
    p.add_argument("--face-threshold", type=float, default=FACE_CLUSTER_THRESHOLD,
                   help="Similarity threshold for same-person clustering")
    p.add_argument("--no-objects", action="store_true",
                   help="Skip object detection")
    p.add_argument("--no-alpr", action="store_true",
                   help="Skip license-plate recognition")
    p.add_argument("--no-scene", action="store_true",
                   help="Skip scene classification")
    p.add_argument("--no-clips", action="store_true",
                   help="Skip condensed-clip extraction (ffmpeg not needed)")
    p.add_argument("--activity-gap", type=int, default=30,
                   help="Seconds of inactivity that split clips (default 30)")
    p.add_argument("--min-activity", type=int, default=5,
                   help="Minimum clip duration in seconds (default 5)")
    p.add_argument("--scene-stride", type=int, default=0,
                   help="Classify scene every N frames (0=same as --stride)")
    return p.parse_args()


# ── CPAI helpers ────────────────────────────────────────────────────────────────
def _post(path, files=None, data=None, timeout=30):
    """POST to CodeProject.AI v1 endpoint, return parsed JSON or None."""
    try:
        r = requests.post(
            f"{CPAI_URL}/v1/{path}",
            files=files, data=data,
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _encode_jpg(frame_bgr, quality=85):
    _, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def detect_faces(frame_bgr, min_confidence):
    result = _post(
        "vision/face",
        files={"image": ("f.jpg", _encode_jpg(frame_bgr), "image/jpeg")},
        data={"min_confidence": min_confidence},
    )
    return result.get("predictions", []) if result and result.get("success") else []


def detect_objects(frame_bgr, min_confidence):
    result = _post(
        "vision/detection",
        files={"image": ("f.jpg", _encode_jpg(frame_bgr), "image/jpeg")},
        data={"min_confidence": min_confidence},
    )
    return result.get("predictions", []) if result and result.get("success") else []


def detect_plates(frame_bgr):
    result = _post(
        "vision/alpr",
        files={"upload": ("f.jpg", _encode_jpg(frame_bgr), "image/jpeg")},
    )
    return result.get("predictions", []) if result and result.get("success") else []


def classify_scene(frame_bgr):
    result = _post(
        "vision/scene",
        files={"image": ("f.jpg", _encode_jpg(frame_bgr), "image/jpeg")},
    )
    if result and result.get("success"):
        return result.get("label"), result.get("confidence", 0.0)
    return None, 0.0


def face_similarity(crop1_bgr, crop2_bgr):
    result = _post(
        "vision/face/match",
        files={
            "image1": ("i1.jpg", _encode_jpg(crop1_bgr), "image/jpeg"),
            "image2": ("i2.jpg", _encode_jpg(crop2_bgr), "image/jpeg"),
        },
        timeout=20,
    )
    if result and result.get("success"):
        return float(result.get("similarity", 0.0))
    return 0.0


# ── Face clustering ─────────────────────────────────────────────────────────────
class FaceClusterer:
    """
    Greedy incremental clustering using CPAI vision/face/match.
    Each cluster is represented by one face crop (its first/clearest example).
    O(N × C) calls where C = number of unique identities — fast in practice.
    """
    def __init__(self, threshold: float = FACE_CLUSTER_THRESHOLD):
        self.threshold = threshold
        self._reps: list[tuple[int, bytes]] = []  # (cluster_id, jpg bytes)
        self.next_id = 0

    def assign(self, face_crop_bgr) -> int:
        crop_bytes = _encode_jpg(face_crop_bgr, quality=90)
        # Compare against stored cluster representatives
        best_sim = 0.0
        best_cid = -1
        for cid, rep_bytes in self._reps:
            try:
                result = _post(
                    "vision/face/match",
                    files={
                        "image1": ("i1.jpg", crop_bytes, "image/jpeg"),
                        "image2": ("i2.jpg", rep_bytes, "image/jpeg"),
                    },
                    timeout=20,
                )
                sim = float(result.get("similarity", 0.0)) if result and result.get("success") else 0.0
            except Exception:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_sim >= self.threshold:
            return best_cid

        # New identity
        cid = self.next_id
        self.next_id += 1
        self._reps.append((cid, crop_bytes))
        return cid


# ── Video helpers ────────────────────────────────────────────────────────────────
def collect_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
        elif p.is_dir():
            for f in p.iterdir():
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(f)
    return sorted(set(videos))


def get_video_info(cap) -> tuple[float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return fps, total


def activity_windows(
    active_frames: list[int],
    fps: float,
    gap_sec: int = 30,
    min_duration_sec: int = 5,
) -> list[tuple[float, float]]:
    """Merge nearby active frames into continuous time windows."""
    if not active_frames:
        return []
    frames = sorted(set(active_frames))
    gap = gap_sec * fps
    windows = []
    start = end = frames[0]
    for f in frames[1:]:
        if f - end <= gap:
            end = f
        else:
            windows.append((start / fps, end / fps))
            start = end = f
    windows.append((start / fps, end / fps))
    return [(s, e) for s, e in windows if (e - s) >= min_duration_sec]


def extract_clips_ffmpeg(
    video_path: Path,
    windows: list[tuple[float, float]],
    clips_dir: Path,
) -> list[dict[str, Any]]:
    clips = []
    for i, (start, end) in enumerate(windows):
        out = clips_dir / f"{video_path.stem}_clip{i + 1:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-i", str(video_path),
            "-t", f"{end - start:.2f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and out.exists():
            clips.append({
                "path": str(out),
                "video": video_path.name,
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
            })
    return clips


# ── Main processing ──────────────────────────────────────────────────────────────
def process_video(
    video_path: Path,
    args,
    output_dir: Path,
    clusterer: FaceClusterer,
    all_detections: list[dict],
) -> list[dict[str, Any]]:
    print(f"\n{'─'*60}")
    print(f"  {video_path.name}")
    print(f"{'─'*60}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping")
        return []

    fps, total_frames = get_video_info(cap)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(exist_ok=True)

    active_frames: list[int] = []
    scene_stride = args.scene_stride if args.scene_stride > 0 else args.stride
    # Seek through video rather than decoding every frame between strides
    analyse_frames = list(range(0, total_frames, args.stride)) if total_frames else []
    # For scene classification use its own stride
    scene_frames  = set(range(0, total_frames, scene_stride)) if (total_frames and not args.no_scene) else set()
    total_to_scan = len(analyse_frames) or 1
    report_every  = max(1, total_to_scan // 20)

    print(f"  fps={fps:.1f}  frames={total_frames}  stride={args.stride}  "
          f"conf={args.confidence}  frames-to-scan={total_to_scan}")

    for scan_n, frame_idx in enumerate(analyse_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps

        # ── Face detection ─────────────────────────────────────────────────
        faces = detect_faces(frame, args.confidence)
        for face in faces:
            x1 = int(face.get("x_min", 0))
            y1 = int(face.get("y_min", 0))
            x2 = int(face.get("x_max", 0))
            y2 = int(face.get("y_max", 0))
            if (x2 - x1) < args.min_face or (y2 - y1) < args.min_face:
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            cid = clusterer.assign(crop)
            cluster_dir = faces_dir / f"person_{cid:03d}"
            cluster_dir.mkdir(exist_ok=True)
            crop_name = f"{video_path.stem}_f{frame_idx:08d}.jpg"
            cv2.imwrite(str(cluster_dir / crop_name), crop)
            active_frames.append(frame_idx)

            all_detections.append({
                "video":          video_path.name,
                "frame_idx":      frame_idx,
                "timestamp_sec":  round(ts, 2),
                "type":           "face",
                "label":          f"person_{cid:03d}",
                "confidence":     round(float(face.get("confidence", 0)), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cluster_id":     cid,
            })

        # ── Object detection ──────────────────────────────────────────────
        if not args.no_objects:
            for obj in detect_objects(frame, args.confidence):
                lbl = obj.get("label", "")
                all_detections.append({
                    "video":         video_path.name,
                    "frame_idx":     frame_idx,
                    "timestamp_sec": round(ts, 2),
                    "type":          "object",
                    "label":         lbl,
                    "confidence":    round(float(obj.get("confidence", 0)), 3),
                    "x1": int(obj.get("x_min", 0)),
                    "y1": int(obj.get("y_min", 0)),
                    "x2": int(obj.get("x_max", 0)),
                    "y2": int(obj.get("y_max", 0)),
                    "cluster_id":    -1,
                })
                if lbl in ACTIVITY_LABELS:
                    active_frames.append(frame_idx)

        # ── ALPR ──────────────────────────────────────────────────────────
        if not args.no_alpr:
            for plate in detect_plates(frame):
                txt = plate.get("plate", plate.get("label", ""))
                all_detections.append({
                    "video":         video_path.name,
                    "frame_idx":     frame_idx,
                    "timestamp_sec": round(ts, 2),
                    "type":          "plate",
                    "label":         txt,
                    "confidence":    round(float(plate.get("confidence", 0)), 3),
                    "x1": int(plate.get("x_min", 0)),
                    "y1": int(plate.get("y_min", 0)),
                    "x2": int(plate.get("x_max", 0)),
                    "y2": int(plate.get("y_max", 0)),
                    "cluster_id":    -1,
                })

        # ── Scene classification (less frequent) ───────────────────────────
        if not args.no_scene and frame_idx in scene_frames:
            label, conf = classify_scene(frame)
            if label:
                all_detections.append({
                    "video":         video_path.name,
                    "frame_idx":     frame_idx,
                    "timestamp_sec": round(ts, 2),
                    "type":          "scene",
                    "label":         label,
                    "confidence":    round(float(conf), 3),
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                    "cluster_id":    -1,
                })

        if scan_n > 0 and scan_n % report_every == 0:
            pct = scan_n / total_to_scan * 100
            faces_so_far = sum(1 for d in all_detections
                               if d["video"] == video_path.name and d["type"] == "face")
            print(f"  [{pct:3.0f}%] frame {frame_idx:,}  ts={ts:.0f}s  "
                  f"faces={faces_so_far}  identities={clusterer.next_id}")

    cap.release()
    print(f"  Done: {total_to_scan} frames scanned (stride={args.stride})")
    # ── Clip extraction ────────────────────────────────────────────────────────
    clips: list[dict[str, Any]] = []
    if not args.no_clips and active_frames:
        windows = activity_windows(
            active_frames, fps,
            gap_sec=args.activity_gap,
            min_duration_sec=args.min_activity,
        )
        if windows:
            clips_dir = output_dir / "clips"
            clips_dir.mkdir(exist_ok=True)
            print(f"  Extracting {len(windows)} condensed clip(s) via ffmpeg…")
            clips = extract_clips_ffmpeg(video_path, windows, clips_dir)
            if clips:
                print(f"  → {len(clips)} clip(s) saved")
            else:
                print("  [WARN] ffmpeg extraction returned no output "
                      "(ffmpeg may not be installed or wrong PATH)")

    return clips


# ── Report generation ────────────────────────────────────────────────────────────
def write_report(
    output_dir: Path,
    all_detections: list[dict],
    clusterer: FaceClusterer,
    all_clips: list[dict[str, Any]],
) -> None:
    # CSV
    if all_detections:
        csv_path = output_dir / "detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_detections[0].keys())
            writer.writeheader()
            writer.writerows(all_detections)
        print(f"\n  detections.csv  ({len(all_detections):,} rows)")

    face_dets  = [d for d in all_detections if d["type"] == "face"]
    obj_dets   = [d for d in all_detections if d["type"] == "object"]
    plate_dets = [d for d in all_detections if d["type"] == "plate"]
    scene_dets = [d for d in all_detections if d["type"] == "scene"]

    obj_counts    = Counter(d["label"] for d in obj_dets)
    plate_counts  = Counter(d["label"] for d in plate_dets)
    scene_counts  = Counter(d["label"] for d in scene_dets)
    videos_done   = sorted(set(d["video"] for d in all_detections))

    summary = {
        "videos_processed":    videos_done,
        "total_detections":    len(all_detections),
        "faces_detected":      len(face_dets),
        "unique_people":       clusterer.next_id,
        "objects_detected":    len(obj_dets),
        "top_objects":         obj_counts.most_common(15),
        "plates_detected":     len(plate_dets),
        "unique_plates":       [p for p, _ in plate_counts.most_common()],
        "top_scenes":          scene_counts.most_common(5),
        "condensed_clips":     len(all_clips),
        "clip_paths":          [clip["path"] for clip in all_clips],
        "clip_details":        all_clips,
    }

    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    html_path = output_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_build_html(summary, output_dir))

    # Console summary
    print("\n" + "═" * 60)
    print("  ANALYSIS COMPLETE")
    print("═" * 60)
    print(f"  Videos processed : {len(videos_done)}")
    print(f"  Faces detected   : {len(face_dets)}  →  {clusterer.next_id} unique people")
    print(f"  Objects detected : {len(obj_dets)}")
    if obj_counts:
        print(f"    top: {', '.join(f'{l}×{n}' for l, n in obj_counts.most_common(5))}")
    print(f"  Plates detected  : {len(plate_dets)}"
          f"  ({len(plate_counts)} unique)")
    if plate_counts:
        print(f"    → {', '.join(plate_counts.keys())}")
    print(f"  Condensed clips  : {len(all_clips)}")
    print(f"\n  Output folder: {output_dir}")
    print(f"  Open report.html for a browsable summary with face grids.")
    print("═" * 60)


def _build_html(summary: dict, output_dir: Path) -> str:
    # Face cluster grid
    faces_dir = output_dir / "faces"
    cluster_html = ""
    if faces_dir.exists():
        for cdir in sorted(faces_dir.iterdir()):
            if not cdir.is_dir():
                continue
            crops = sorted(cdir.glob("*.jpg"))[:8]
            if not crops:
                continue
            thumbs = "".join(
                f'<img src="{c.relative_to(output_dir)}" '
                f'style="width:80px;height:80px;object-fit:cover;margin:2px;'
                f'border-radius:4px;border:1px solid #ccc">'
                for c in crops
            )
            count = len(list(cdir.glob("*.jpg")))
            cluster_html += (
                f'<div style="display:inline-block;margin:8px;padding:8px;'
                f'background:#f9f9f9;border:1px solid #ddd;border-radius:8px;'
                f'vertical-align:top">'
                f'<div style="font-weight:bold;margin-bottom:4px">{cdir.name}'
                f' <small style="color:#888">({count} crops)</small></div>'
                f'{thumbs}</div>'
            )

    # Objects table
    obj_rows = "".join(
        f"<tr><td>{lbl}</td><td>{cnt}</td></tr>"
        for lbl, cnt in summary.get("top_objects", [])
    )

    # Plates table
    plate_rows = "".join(
        f"<tr><td>{p}</td></tr>"
        for p in summary.get("unique_plates", [])
    ) or "<tr><td><em>none</em></td></tr>"

    # Scene table
    scene_rows = "".join(
        f"<tr><td>{lbl}</td><td>{cnt}</td></tr>"
        for lbl, cnt in summary.get("top_scenes", [])
    ) or "<tr><td colspan='2'><em>none</em></td></tr>"

    # Clips
    clips_html = "".join(
        f'<li><a href="clips/{Path(c).name}">{Path(c).name}</a></li>'
        for c in summary.get("clip_paths", [])
    ) or "<li><em>none generated</em></li>"

    videos_html = "".join(
        f"<li>{v}</li>" for v in summary.get("videos_processed", [])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Video Analysis Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ border-bottom: 2px solid #444; padding-bottom: 4px; margin-top: 32px; }}
  table {{ border-collapse: collapse; margin: 8px 0; }}
  td, th {{ border: 1px solid #bbb; padding: 6px 14px; }}
  th {{ background: #eee; }}
  .stat {{ display:inline-block; background:#1a1a2e; color:#fff;
           border-radius:8px; padding:12px 22px; margin:6px; min-width:120px;
           text-align:center; }}
  .stat .n {{ font-size:2em; font-weight:bold; }}
  .stat .l {{ font-size:0.85em; margin-top:2px; opacity:0.8; }}
</style>
</head>
<body>
<h1>Surveillance Video Analysis Report</h1>

<div>
  <div class="stat"><div class="n">{len(summary['videos_processed'])}</div><div class="l">Videos</div></div>
  <div class="stat"><div class="n">{summary['unique_people']}</div><div class="l">Unique People</div></div>
  <div class="stat"><div class="n">{summary['faces_detected']}</div><div class="l">Face Detections</div></div>
  <div class="stat"><div class="n">{summary['objects_detected']}</div><div class="l">Object Detections</div></div>
  <div class="stat"><div class="n">{summary['plates_detected']}</div><div class="l">Plates Read</div></div>
  <div class="stat"><div class="n">{summary['condensed_clips']}</div><div class="l">Condensed Clips</div></div>
</div>

<h2>Face Clusters — Unique People</h2>
<div>{cluster_html or '<p><em>No faces detected in these videos.</em></p>'}</div>

<h2>Top Detected Objects</h2>
<table><tr><th>Object</th><th>Count</th></tr>{obj_rows or '<tr><td colspan=2><em>none</em></td></tr>'}</table>

<h2>License Plates</h2>
<table><tr><th>Plate Text</th></tr>{plate_rows}</table>

<h2>Scene Classification</h2>
<table><tr><th>Scene</th><th>Frames</th></tr>{scene_rows}</table>

<h2>Condensed Activity Clips</h2>
<ul>{clips_html}</ul>

<h2>Videos Processed</h2>
<ul>{videos_html}</ul>
</body>
</html>
"""


# ── Entry point ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify CodeProject.AI is reachable
    try:
        pr = requests.get(f"{CPAI_URL}/v1/server/status/ping", timeout=10)
        resp = pr.json()
        if not resp.get("success"):
            print("ERROR: CodeProject.AI server responded but reported not OK.")
            sys.exit(1)
        print(f"CodeProject.AI v{resp.get('message','?')} reachable at {CPAI_URL}")
    except Exception as exc:
        print(f"ERROR: Cannot reach CodeProject.AI at {CPAI_URL}: {exc}")
        print("Start CodeProject.AI first:")
        print('  & "C:\\Program Files\\CodeProject\\AI\\Server\\CodeProject.AI.Server.exe"')
        sys.exit(1)

    videos = collect_videos(args.input)
    if not videos:
        print(f"No video files found in: {args.input}")
        sys.exit(1)
    print(f"Found {len(videos)} video(s)\n")
    for v in videos:
        print(f"  {v}")

    clusterer = FaceClusterer(threshold=args.face_threshold)
    all_detections: list[dict] = []
    all_clips: list[str] = []

    t0 = time.time()
    for video_path in videos:
        clips = process_video(
            video_path, args, output_dir, clusterer, all_detections
        )
        all_clips.extend(clips)

    elapsed = time.time() - t0
    print(f"\nTotal processing time: {elapsed:.0f}s")
    write_report(output_dir, all_detections, clusterer, all_clips)


if __name__ == "__main__":
    main()
