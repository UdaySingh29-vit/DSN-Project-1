# SYSTEM PROMPT — Person 1: Detection, Depth Estimation & Integration Lead

## Who I Am
I am Person 1 on a 3-person team (reduced from an original 6-person team) building the **Real-Time Assistive Navigation System** — a Python-based AI pipeline that helps visually impaired users navigate by detecting objects, estimating their distance, tracking motion, and giving spoken alerts, using only a standard camera (no LiDAR/stereo hardware). This is a college exhibition project (Team 195, DSN-1, VIT Bhopal University) with roughly a month of active build time remaining.

I have a strong existing background in Python and AI/ML. Assume I understand ML pipelines, tensors, model inference, and system architecture — don't explain basic ML concepts unless I ask. Focus on precise, implementation-level help.

## Mission & Scope
I own three consolidated modules:

1. **Object Detection** — YOLOv8-based real-time detection of navigation-relevant objects (people, furniture, doors, and hazard classes like stairs/curbs via fine-tuning).
2. **Distance/Depth Estimation** — hybrid distance estimation combining a known-object-size geometric heuristic (pinhole camera model) with MiDaS monocular relative depth for objects without a known reference size.
3. **Integration & Demo Lead** — I own the final pipeline architecture: merging my own modules with Person 2's (Tracking & Egomotion) and Person 3's (Decision Logic & Voice) output into a single real-time loop, plus owning the live demo.

### Progress So Far (do not repeat this work — build on it)
- Local dev environment working: Python venv, VS Code, on Windows.
- `test_detect.py` — YOLOv8n running live on webcam via `ultralytics` + `opencv-python`, confirmed working.
- `distance_estimate.py` — known-object-size distance estimation implemented: calibrated `FOCAL_LENGTH` via a person standing at a measured 2m distance, using the pinhole formula `distance_m = (real_height_m * focal_length) / pixel_height`. Red/Green/Blue proximity banding implemented (≤5m / ≤10m / >10m — thresholds still tunable).
- Identified and fixed a real limitation: when an object is closer than ~1m, its bounding box gets clipped by the frame edge, breaking the pixel-height assumption. Fixed with edge-clipping detection (`FRAME_EDGE_MARGIN`).
- `distance_with_memory.py` — upgraded further: added a per-class rolling history (`deque`, `HISTORY_LEN=8`) of recent `(timestamp, distance)` readings, with a simple moving-average velocity calculation (`compute_velocity`). When a box is edge-clipped, the system now **extrapolates** distance from last-known distance + closing velocity × elapsed time, rather than showing an unreliable or frozen number.
- **Known open limitation**: history is currently keyed per class name, not per individual object instance — two people in frame will have their distance histories mixed together. This is expected to be resolved once Person 2's ByteTrack integration provides persistent per-object track IDs — at that point, history should be re-keyed by `track_id` instead of `class_name`.
- MiDaS has NOT been integrated yet — currently only known-size classes (person, chair, couch, tv, backpack) get distance estimates; everything else is undated. This is the next task.

## Model Accuracy & Performance Thresholds
- **YOLOv8 detection confidence**: minimum 0.45 confidence to trigger a downstream distance/decision pipeline event; log (but don't act on) detections between 0.25–0.45 for later threshold tuning.
- **Detection inference speed**: target ≥15 FPS on CPU with `yolov8n`; if fine-tuning custom hazard classes on `yolov8s`, target ≥10 FPS minimum — reassess model size if this isn't met on target hardware.
- **Fine-tuned hazard classes (stairs/curbs) mAP50 target**: ≥0.6 as a minimum viable bar for the exhibition demo; document actual achieved mAP honestly rather than overclaiming.
- **Known-size distance estimation error**: target ≤15% relative error in the 2–8m range (validate against manually tape-measured ground truth); accuracy is expected to degrade outside this range and must be documented, not hidden.
- **MiDaS depth (once integrated)**: since MiDaS gives *relative* depth only, the threshold is *consistency*, not absolute accuracy — frame-to-frame depth jitter for a static object should stay within ~10% of its own rolling average; large unexplained jumps indicate a bug, not real distance change.
- **Extrapolation validity window**: `MAX_EXTRAPOLATION_AGE` should not exceed ~1.5–2 seconds without a fresh real reading — beyond that, fall back to a generic "very close" warning rather than trusting stale velocity data.
- **End-to-end pipeline target (post-integration)**: ≥10 FPS sustained on the demo laptop, since this is the frame rate ceiling every other module's real-time performance depends on.

## To-Do List
1. Integrate MiDaS depth model for classes outside `KNOWN_HEIGHTS` — get it running standalone on a sample frame first, then combine with existing known-size logic (known-size takes priority when available; MiDaS is the fallback).
2. Calibrate MiDaS relative depth against known-size readings in the same frame, to normalize MiDaS's arbitrary depth units into an approximate meter scale.
3. Fine-tune YOLOv8 on a hazard dataset (stairs, curbs) sourced from Roboflow Universe; document dataset size, source, and resulting mAP.
4. Refactor `distance_with_memory.py`'s history keying from `class_name` to `track_id`, once Person 2 delivers tracked object IDs — coordinate this handoff explicitly.
5. Define and finalize the full interface contract (exact dict/field names) that Person 2 and Person 3 will build their modules against — do this BEFORE they write significant code, not after.
6. Build the pipeline skeleton (placeholder functions per stage) so Person 2 and Person 3 can develop independently against a stable interface.
7. Once Person 2/3 modules exist, integrate incrementally: Detection+Distance (done) → add Tracking+Egomotion → add Decision+Voice → full loop.
8. Profile end-to-end latency once integrated; identify and optimize the slowest stage if FPS target isn't met.
9. Prepare and rehearse the live demo, including a recorded backup video in case live hardware fails during exhibition.

## Integration Mandate — Architecture for Stitching Person 2 & Person 3's Code Into the Final Pipeline

As Integration Lead, I define the following interface contract. Person 2 and Person 3 must build their modules to consume/produce **exactly** these data shapes — do not let field names drift without updating this contract first.

### Stage 1 — My output (Detection + Distance): `object_cube`
```python
object_cube = {
    "class_name": "person",
    "confidence": 0.87,
    "bbox": (x1, y1, x2, y2),
    "distance_m": 4.2,
    "distance_band": "red",        # "red" | "green" | "blue"
    "distance_method": "known_size",  # "known_size" | "midas_relative" | "extrapolated"
    "center_x_frac": 0.52,         # 0=left edge, 1=right edge — for path-in-frame logic
    "frame_id": 1042,
}
```
Output per frame: `List[object_cube]`.

### Stage 2 — Person 2's output (Tracking + Egomotion): `tracked_object` + `egomotion_status`
```python
tracked_object = {
    # ...all fields from object_cube, plus:
    "track_id": 17,
    "motion_state": "moving",       # "moving" | "static" | "unknown"
    "velocity_mps": 1.1,
    "direction": "toward_user",     # "toward_user" | "away_from_user" | "lateral" | "unknown"
}

egomotion_status = {
    "frame_id": 1042,
    "displacement_score": 0.34,
    "moved_significantly": False,
}
```

### Stage 3 — Person 3's output (Decision + Voice): `speech_action`
```python
speech_action = {
    "should_speak": True,
    "message": "person, 3 meters, ahead",
    "priority": 1,
    "trigger_reason": "in_path_close",  # "in_path_close" | "moving_toward_user" | "rescan_prompt" | "user_query_response"
}
```

### My integration responsibilities specifically:
1. **Own the main loop** — I write the top-level `for frame in camera_stream:` loop that calls each person's module in sequence: my detection/distance → Person 2's tracking/egomotion → Person 3's decision/voice.
2. **Own the object memory re-keying transition** — when Person 2's `track_id` becomes available, I update my `distance_with_memory.py` history structure to key by `track_id` instead of `class_name`. This is my responsibility, not Person 2's, since it's my module being upgraded.
3. **Own graceful degradation** — if Person 2 or Person 3's module isn't ready yet or throws an error on a given frame, the pipeline must not crash; my job is to wrap each stage call in error handling that logs the failure and either skips that stage's contribution for that frame or falls back to my last-known-good state (this is exactly the extrapolation pattern I already built for distance — reuse that philosophy).
4. **Own latency profiling** — instrument each stage with timing, identify the bottleneck stage, and either optimize it or negotiate scope-cutting with the team if the target FPS isn't achievable.
5. **Enforce the interface contract** — if either teammate needs to change a field name or structure, that change must be agreed with me first, since I'm the one wiring everything together and a silent field rename breaks integration invisibly.
6. **Own the demo script and fallback plan** — including a pre-recorded backup video in case live camera/hardware fails during the actual exhibition.

When I paste code from Person 2 or Person 3 into this project, help me: (a) verify their output matches the contract above, (b) write the glue code to feed their output into the next stage, and (c) flag any mismatch between what they built and what this contract specifies.
