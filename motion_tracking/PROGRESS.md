# Motion Tracking Module — Progress Log

Owner: Person 2 (Motion Tracking, Object Memory & Egomotion Lead)
Project: Real-Time Assistive Navigation System (Team 195, DSN-1, VIT Bhopal)

This documents everything done so far on the **Motion Tracking & Object Memory**
half of my scope (egomotion is separate — see "Not started yet" below).

---

## To-do list progress

| # | Item | Status |
|---|---|---|
| 1 | Learn IOU conceptually, get ByteTrack running standalone on sample video | ✅ Done |
| 2 | Build tracking module (`object_cube` → `tracked_object`) | ✅ Done |
| 3 | Implement object memory bank with expiry rule | ✅ Done |
| 4 | Learn optical flow, get it running standalone | ⬜ Not started |
| 5 | Build egomotion module | ⬜ Not started |
| 6 | Test "did user move" signal against real walking | ⬜ Not started |
| 7 | Test tracking + memory together (walk in/out of frame repeatedly) | 🔶 Partially — tested with synthetic data, not real footage yet |
| 8 | Package both modules cleanly for Person 1 to import | ✅ Done (motion tracking half only) |
| 9 | Document known limitations honestly | ✅ Done (this doc + handoff README) |

---

## What was actually built

### 1. IOU + ByteTrack exploration (`bytetrack_standalone.py`)
- Learned IOU conceptually: `intersection area / union area` between two boxes,
  used by trackers to decide whether a new detection matches an existing track.
- Learned why ByteTrack specifically improves on plain IOU-matching: it predicts
  each track's expected position via a Kalman filter before matching (so it
  survives brief occlusion), and does a second matching pass using
  *low-confidence* detections instead of discarding them outright (the "Byte" in
  ByteTrack).
- Wrote a standalone script using YOLOv8's built-in `.track()` (which uses
  ByteTrack under the hood) to run against a real sample video and print
  `track_id / class / confidence / bbox` per frame — used purely to build
  intuition and visually check ID stability. This script is disposable; it's
  not part of the real pipeline.

### 2. `MotionTracker` module (`motion_tracker.py`)
- Takes Person 1's `object_cube` list per frame, runs `supervision`'s
  `sv.ByteTrack` directly (not the YOLO wrapper — this module doesn't run
  detection itself), and outputs the `tracked_object` schema with `track_id`,
  `motion_state`, `velocity_mps`, `direction`.
- Config maps directly onto the spec's thresholds: `minimum_matching_threshold`
  = 0.3 (IOU), `minimum_consecutive_frames` = 3 (track confirmation).
- Velocity/direction are derived from how `distance_m` changes over a smoothing
  window of recent frames (radial motion — closing/opening distance — takes
  priority over lateral crossing, since closing distance is the higher-danger
  signal for a visually impaired user).
- **Bug found and fixed during multi-object testing:** initially assumed
  ByteTrack's output list stays in the same order/length as the input list.
  This breaks with multiple objects, since ByteTrack can reorder or drop
  detections internally, and confirmed (via a `supervision` GitHub issue) that
  custom metadata attached to detections does **not** survive tracking. Fixed
  by remapping tracker output back to input objects using bbox coordinates,
  which *are* preserved unchanged for matched detections.

### 3. `ObjectMemoryBank` module (`object_memory.py`)
- A dictionary keyed by `track_id`, storing each object's last known state.
- Expiry rule: entries removed once `distance_m > 40` or unseen for more than
  `MAX_FRAMES_UNSEEN` (45 frames, ~1.5s @ 30fps) — both placeholder values
  pending real walking tests.
- Exposes `get_known_object()` (single lookup) and `get_all_remembered()` (full
  snapshot — intended for Person 3's on-demand questions like "which way is
  clear?").

### Testing done
- Single-object test: confirmed velocity/direction compute correctly once
  enough confirmed frames accumulate (had to learn that `TRACK_CONFIRM_FRAMES`
  delays the *first* few frames from appearing in output at all — not a bug).
- Multi-object test (3 simultaneous objects, including a moving object whose
  bbox overlaps a static one mid-scene): confirmed no value cross-contamination
  after the bbox-remapping fix.
- Performance profile: `MotionTracker.update()` isolated stage averaged ~2ms per
  frame across 70 frames / 3 objects — comfortably within the shared ≥10 FPS
  pipeline budget.
- Memory bank expiry: confirmed an object gone >45 frames is correctly forgotten,
  while continuously-visible objects persist.

### Packaging & handoff
- Wrote `requirements.txt` (`supervision`, `numpy`), pinned `supervision<0.30.0`
  since `sv.ByteTrack` is deprecated and scheduled for removal in that version.
- Wrote a handoff README covering input/output schema, usage, tunable constants,
  and known limitations, ready to send to Person 1 alongside the two module
  files — independently of egomotion, which isn't ready yet.

---

## Future improvements

### Near-term (needed before final integration)
- **Migrate off deprecated `sv.ByteTrack`** to the `trackers` package's
  `ByteTrackTracker` before `supervision` actually removes it — currently just
  patched via a version pin, not a real fix.
- **Test against real YOLO output**, not just synthetic fake frames — real
  detections will have jitter, occasional missed frames, and imperfect boxes
  that the synthetic tests didn't stress.
- **Re-run the "walk in and out of frame repeatedly" test (to-do #7) with real
  footage**, not simulated data, to see actual ID-switch behavior and tune
  `MAX_FRAMES_UNSEEN` against something real rather than a guessed placeholder.
- **Tune all threshold constants** (`IOU_MATCH_THRESHOLD`,
  `STATIC_VELOCITY_THRESHOLD_MPS`, `EXPIRY_DISTANCE_M`, `MAX_FRAMES_UNSEEN`,
  etc.) against real walking-speed data instead of the spec's starting
  defaults.

### Design improvements worth considering
- **Combined direction classification** — currently radial and lateral motion
  are treated as mutually exclusive (whichever is faster wins the label). An
  object moving diagonally toward-and-across the user doesn't get both signals
  represented. Could report a compound direction (e.g. `"toward_user_from_left"`)
  if Person 3's voice logic could actually make use of it.
- **Same-class, same-bbox edge case** — the bbox-based remapping fix assumes no
  two objects share identical bbox coordinates in one frame. Extremely unlikely
  with real detector output, but worth a defensive check (e.g. falling back to
  IOU-based matching instead of exact bbox equality) if time allows.
- **Occlusion / re-identification robustness** — currently just relies on
  whatever ByteTrack's default re-identification behavior is when an object
  leaves and a similar one re-enters. Worth deliberately testing this scenario
  (two people crossing paths, one occluding the other briefly) rather than
  assuming it "just works."
- **Track ID-switch rate as a real logged metric** — the spec calls for
  tracking this during testing, not just anecdotally. Worth adding an actual
  counter/logger rather than eyeballing printed output.

### Not started at all
- **Egomotion module** (to-do #4, #5, #6): optical flow (Lucas-Kanade) for
  detecting user/camera movement and triggering the "please rescan your
  surroundings" prompt. Entirely separate piece of work, not begun yet.
