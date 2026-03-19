# Fireflaker WIP Status

This note is for continuing the project on another PC without having to rediscover the current state.

## Current conclusion

- Fireflaker is still a work in progress.
- CodeProject.AI works on this Windows machine.
- Frigate was not practical on this machine because Docker-based Linux containers were blocked by local virtualization settings.
- If moving to another PC with virtualization enabled, Frigate becomes realistic again.

## Why work is moving

This PC is a poor target for Frigate because Docker cannot be relied on here when virtualization is off.

That means:

- Fireflaker can continue here as a Windows-native workflow.
- Frigate should be evaluated on a different PC that supports Docker Desktop, WSL2, or a native Linux environment.

## Repo purpose right now

Fireflaker is a Windows-first drop-folder pipeline that:

- watches a folder for new video files
- stages processing locally
- calls CodeProject.AI for inference
- writes tagged clips and small summaries
- keeps a persistent people library

It is not yet as mature as Frigate for surveillance eventing.

## Important current findings

1. The first SMB file seen in `Z:\CLIP_palentry` was corrupt:
   - `C0001.MP4`
   - error: `moov atom not found / unreadable metadata`

2. Fireflaker needed multiple fixes around:
   - SMB/network share watching
   - backlog readiness on large existing files
   - corrupt file handling
   - retry suppression for previously failed files

3. Fireflaker did advance past the first corrupt file and began staging `C0002` locally, but the overall SMB workflow is still not mature enough to call finished.

## Recommended next decision

Use the next machine to answer this first:

### Option A: Frigate-first

Choose this if the next machine can run Docker Desktop or Linux cleanly.

Why:

- Frigate is more mature for surveillance event pipelines
- object tracking and event logic are already solved
- less custom workflow code is needed

### Option B: Fireflaker-first

Choose this if Windows-native operation and drop-folder batch processing matter more than live-camera NVR workflows.

Why:

- no Docker requirement
- keeps CodeProject.AI on Windows
- better fit for archived MP4 ingestion than an NVR-centric design

## If continuing Fireflaker on another PC

1. Create a repo-local `.venv`
2. Install `requirements.txt`
3. Ensure CodeProject.AI is running at `http://localhost:32168`
4. Start with a local folder first, not SMB
5. Only move back to SMB after validating local processing end to end

Suggested first run:

```powershell
& ".\run_auto_ingest.ps1" -Once
```

## If switching to Frigate on another PC

1. Verify virtualization is enabled in firmware and Windows features
2. Verify Docker Desktop can run Linux containers
3. Use Frigate for the main surveillance pipeline first
4. Only reuse Fireflaker ideas if you still need:
   - drop-folder MP4 processing
   - lightweight Windows-native batch mode
   - persistent people grouping outside a full NVR stack

## Practical recommendation

On a better machine:

- try Frigate first if Docker works
- keep Fireflaker as the fallback for Windows-native folder ingestion

That avoids forcing this repo to compete with a more mature surveillance stack when the host environment is the main blocker.