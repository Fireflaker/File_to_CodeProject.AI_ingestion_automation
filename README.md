# Fireflaker

Fireflaker is a lightweight Windows-first surveillance ingestion toolchain built for dropped video files, not live NVR camera management.

It keeps CodeProject.AI as the inference backend and avoids rebuilding detection, ALPR, OCR, or face matching from scratch.

## What it is

- A folder watcher for incoming video files
- A clip-first processing pipeline
- A persistent people library with cross-video face grouping
- A small-summary output format instead of a heavy dashboard report

## What it is not

- Not an NVR
- Not a replacement for CodeProject.AI
- Not a custom deep-learning stack
- Not a full media management product like Immich

## Why this exists

Agent DVR was the wrong abstraction for drop-folder batch processing. The useful core was already the manual Python pipeline plus CodeProject.AI. Fireflaker packages that workflow cleanly and removes the NVR overhead.

## Core decisions

1. Keep CodeProject.AI for inference
   Reason: it already provides usable object detection, face detection, face matching, ALPR, OCR, and scene classification on Windows with low setup overhead.

2. Use watchfiles for folder watching
   Reason: do not reinvent file watching, debouncing, or polling fallbacks.

3. Keep reports minimal
   Reason: the main useful output is tagged clips plus persistent people groups, not a large HTML report.

4. Keep people state locally
   Reason: the Immich-like value is persistent person grouping and thumbnails across videos.

5. Process one file at a time
   Reason: simpler failure handling, predictable archives, and less contention against CodeProject.AI on CPU-first systems.

## Repo contents

- `auto_ingest.py`
  Main automation entrypoint.

- `analyze_videos.py`
  CodeProject.AI-backed analyzer reused by the automation layer.

- `run_auto_ingest.ps1`
  Portable launcher.

- `requirements.txt`
  Minimal Python dependencies.

- `AGENTS.md`
  AI-oriented deployment and usage notes.

## Quick start

1. Create a venv in the repo root.
2. Install `requirements.txt`.
3. Ensure CodeProject.AI is running and reachable at `http://localhost:32168`.
4. Run:

```powershell
& ".\run_auto_ingest.ps1"
```

Default paths are repo-local:

- `drop_incoming/`
- `workspace/`

## Outputs

Per processed video:

- `workspace/reports/<timestamp>__<video-name>/summary.txt`
- `workspace/reports/<timestamp>__<video-name>/summary.json`
- `workspace/reports/<timestamp>__<video-name>/detections.csv`
- `workspace/reports/<timestamp>__<video-name>/clips/...`
- `workspace/reports/<timestamp>__<video-name>/faces/...`

Global persistent state:

- `workspace/people/index.json`
- `workspace/people/person_0001/...`
- `workspace/processed_originals/...`
- `workspace/failed/...`

## Intended use

Drop MP4 or other supported video files into `drop_incoming/`. Fireflaker waits for the copy to finish, analyzes the file, writes tagged clips and a compact summary, updates the people library, and archives the original.