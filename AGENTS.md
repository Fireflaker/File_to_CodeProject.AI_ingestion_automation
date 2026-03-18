# Fireflaker AI Notes

This file is written for another AI agent that needs to deploy or extend this repo quickly.

## Mission

Deploy a drop-folder surveillance video workflow with minimal reasoning overhead and minimal custom ML work.

## Use these pieces as-is

1. `watchfiles`
   Use it for filesystem watching and debounce behavior. Do not replace it with a custom polling loop unless there is a platform-specific failure.

2. `CodeProject.AI`
   Use it as the inference backend. Do not replace it with raw YOLO, DeepFace, or other OSS stacks unless there is a demonstrated accuracy gap on the target footage.

3. `auto_ingest.py`
   This is the orchestration layer. Extend this first before adding new services.

4. `analyze_videos.py`
   This is the reusable scan engine. Keep it as the main place for frame sampling, inference calls, and clip extraction.

## Architectural intent

The correct abstraction is:

- incoming folder
- stable-file detection
- single-file processing
- tagged clips
- persistent people library
- processed archive
- failed archive

The wrong abstraction is:

- NVR cameras
- live stream management
- accounts and dashboards
- rebuilding an entire media product

## Key deployment assumptions

1. Windows is the primary target.
2. CodeProject.AI is expected at `http://localhost:32168`.
3. ffmpeg and ffprobe should be on PATH for clip extraction and metadata time parsing.
4. Python should be 3.9+.

## Minimal deployment steps

1. Create `.venv` in repo root.
2. Install `requirements.txt`.
3. Start CodeProject.AI.
4. Run `run_auto_ingest.ps1`.
5. Drop videos into `drop_incoming`.

## When to change the system

Only extend the pipeline if there is a clear failure mode:

1. tiny far-away objects missed consistently
2. plate reads consistently poor on target footage
3. face grouping unstable across similar clips
4. clip tags too weak for browsing

## Preferred extension order

1. Improve clip tagging logic
2. Improve persistent people metadata
3. Improve file queue and retry handling
4. Add optional review UI
5. Only then consider swapping any inference component

## Non-goals

1. Do not add Agent DVR back into the ingestion loop.
2. Do not replace CodeProject.AI without evidence.
3. Do not build a large HTML reporting system unless explicitly requested.