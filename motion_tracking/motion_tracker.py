"""
Motion Tracking module — Person 2, item #2 on the to-do list.

Takes Person 1's per-frame `object_cube` list as input, runs ByteTrack to assign
persistent track_ids, and computes motion_state / velocity_mps / direction from
the distance_m history of each track. Outputs the exact `tracked_object` schema
from person2.md — confirm any field changes with Person 1 before shipping.

Setup:
    pip install supervision numpy

Assumed input schema per object, from Person 1 (confirm before deviating):
    object_cube = {
        "class_name": "person",
        "confidence": 0.87,
        "bbox": (x1, y1, x2, y2),
        "distance_m": 4.2,
        "distance_band": "red",
        "distance_method": "known_size",
        "center_x_frac": 0.52,
        "frame_id": 1042,
    }
"""

from collections import deque, defaultdict
import numpy as np
import supervision as sv

# ---- Tunable constants (per person2.md spec) — expose these, don't hardcode inline ----
IOU_MATCH_THRESHOLD = 0.3          # ByteTrack association threshold
TRACK_CONFIRM_FRAMES = 3           # frames before a track is "confirmed"
STATIC_VELOCITY_THRESHOLD_MPS = 0.15   # below this = "static"
VELOCITY_SMOOTHING_WINDOW = 5      # frames of distance history used to compute velocity
LATERAL_MOVEMENT_THRESHOLD = 0.03  # center_x_frac change/frame that counts as lateral motion


class MotionTracker:
    """
    Wraps ByteTrack + per-track distance history to turn raw detections into
    tracked_object dicts with motion state.
    """

    def __init__(self, fps: float):
        self.fps = fps
        self.tracker = sv.ByteTrack(
            minimum_matching_threshold=IOU_MATCH_THRESHOLD,
            minimum_consecutive_frames=TRACK_CONFIRM_FRAMES,
        )
        # track_id -> deque of (frame_id, distance_m, center_x_frac)
        self.history = defaultdict(lambda: deque(maxlen=VELOCITY_SMOOTHING_WINDOW))

    def update(self, object_cubes: list[dict]) -> list[dict]:
        """
        object_cubes: this frame's list of object_cube dicts from Person 1.
        Returns: list of tracked_object dicts (see schema in person2.md).
        """
        if not object_cubes:
            # still call the tracker with an empty Detections so it can age out lost tracks
            detections = sv.Detections.empty()
            self.tracker.update_with_detections(detections)
            return []

        # --- Build a supervision Detections object from Person 1's cubes ---
        boxes = np.array([obj["bbox"] for obj in object_cubes], dtype=float)
        confidences = np.array([obj["confidence"] for obj in object_cubes], dtype=float)
        # supervision wants integer class ids, not names — map names to a stable int per session
        class_names = [obj["class_name"] for obj in object_cubes]
        class_ids = np.array([self._class_id(name) for name in class_names])

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )

        # --- Run ByteTrack: assigns/updates tracker_id per detection ---
        # IMPORTANT: supervision's ByteTrack can drop or reorder detections
        # internally (two-pass high/low-confidence matching), so `tracked[i]`
        # does NOT reliably correspond to `object_cubes[i]`. Its `data` field
        # is also confirmed lost in transit (supervision issue #754), so we
        # can't smuggle an index through that way either. Instead, map back
        # by bbox coordinates, which ARE preserved unchanged for every
        # detection the tracker actually matches this frame.
        bbox_to_cube = {
            tuple(round(v, 4) for v in obj["bbox"]): obj for obj in object_cubes
        }
        tracked = self.tracker.update_with_detections(detections)

        results = []
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            output_bbox = tuple(round(v, 4) for v in tracked.xyxy[i].tolist())
            source_obj = bbox_to_cube.get(output_bbox)
            if source_obj is None:
                # Shouldn't happen if bboxes are unique per frame — if it does,
                # two objects had identical coordinates this frame, or the
                # preserved-bbox assumption broke. Skip rather than silently
                # attach the wrong object's distance/class to this track_id.
                continue
            frame_id = source_obj["frame_id"]
            distance_m = source_obj["distance_m"]
            center_x_frac = source_obj["center_x_frac"]

            # update this track's history, then derive motion from it
            self.history[track_id].append((frame_id, distance_m, center_x_frac))
            velocity_mps, direction = self._compute_motion(track_id)
            motion_state = (
                "static" if abs(velocity_mps) < STATIC_VELOCITY_THRESHOLD_MPS else "moving"
            )

            results.append({
                "class_name": source_obj["class_name"],
                "confidence": source_obj["confidence"],
                "bbox": tuple(source_obj["bbox"]),
                "distance_m": distance_m,
                "distance_band": source_obj["distance_band"],
                "distance_method": source_obj["distance_method"],
                "center_x_frac": center_x_frac,
                "frame_id": frame_id,
                "track_id": track_id,
                "motion_state": motion_state,
                "velocity_mps": velocity_mps,
                "direction": direction,
            })

        self._expire_stale_tracks(tracked.tracker_id)
        return results

    def _compute_motion(self, track_id: int) -> tuple[float, str]:
        """
        Derives (velocity_mps, direction) from a track's recent distance/position history.
        Radial velocity (toward/away) takes priority over lateral, since closing distance
        is the higher-danger signal for a visually impaired user.
        """
        hist = self.history[track_id]
        if len(hist) < 2:
            return 0.0, "stationary"  # not enough history yet to say anything

        first_frame, first_dist, first_x = hist[0]
        last_frame, last_dist, last_x = hist[-1]
        frame_delta = last_frame - first_frame
        if frame_delta <= 0:
            return 0.0, "stationary"

        time_delta_s = frame_delta / self.fps
        radial_velocity = (first_dist - last_dist) / time_delta_s  # positive = approaching
        lateral_velocity = (last_x - first_x) / time_delta_s        # frac/sec, not m/s

        speed = abs(radial_velocity)

        if speed >= STATIC_VELOCITY_THRESHOLD_MPS:
            direction = "toward_user" if radial_velocity > 0 else "away_from_user"
            return radial_velocity, direction

        # radial motion is negligible — check lateral crossing instead
        if abs(lateral_velocity) >= LATERAL_MOVEMENT_THRESHOLD:
            direction = "left_to_right" if lateral_velocity > 0 else "right_to_left"
            # report lateral speed as velocity_mps too, so motion_state still reflects it
            return lateral_velocity, direction

        return radial_velocity, "stationary"

    def _expire_stale_tracks(self, active_ids) -> None:
        """Drop history for any track_id ByteTrack is no longer reporting."""
        active_ids = set(int(i) for i in active_ids)
        for stale_id in list(self.history.keys()):
            if stale_id not in active_ids:
                del self.history[stale_id]

    _class_name_to_id: dict = {}

    def _class_id(self, name: str) -> int:
        """supervision needs int class ids; assign stable ints to class names as seen."""
        if name not in self._class_name_to_id:
            self._class_name_to_id[name] = len(self._class_name_to_id)
        return self._class_name_to_id[name]


# --- Minimal usage example against a fake per-frame feed, for standalone testing ---
if __name__ == "__main__":
    tracker = MotionTracker(fps=30.0)

    # Simulated Person 1 output over 5 frames: one person walking toward the
    # camera at a realistic ~1.2 m/s (average human walking speed), so distance
    # drops by ~0.04m per frame at 30fps. bbox grows slightly frame-to-frame to
    # mimic getting closer (not used in the velocity math, just for realism).
    #
    # NOTE: TRACK_CONFIRM_FRAMES = 3 means frames 1-2 get consumed internally by
    # ByteTrack to build confidence and won't appear in the output at all - only
    # frames 3, 4, 5 will actually print. That's expected, not a bug.
    fake_frames = [
        [{"class_name": "person", "confidence": 0.9, "bbox": (100, 100, 200, 300),
          "distance_m": 6.00, "distance_band": "green", "distance_method": "known_size",
          "center_x_frac": 0.50, "frame_id": 1}],
        [{"class_name": "person", "confidence": 0.9, "bbox": (101, 100, 202, 301),
          "distance_m": 5.96, "distance_band": "green", "distance_method": "known_size",
          "center_x_frac": 0.50, "frame_id": 2}],
        [{"class_name": "person", "confidence": 0.9, "bbox": (102, 100, 204, 302),
          "distance_m": 5.92, "distance_band": "green", "distance_method": "known_size",
          "center_x_frac": 0.50, "frame_id": 3}],
        [{"class_name": "person", "confidence": 0.9, "bbox": (103, 100, 206, 303),
          "distance_m": 5.88, "distance_band": "green", "distance_method": "known_size",
          "center_x_frac": 0.50, "frame_id": 4}],
        [{"class_name": "person", "confidence": 0.9, "bbox": (104, 100, 208, 304),
          "distance_m": 5.84, "distance_band": "green", "distance_method": "known_size",
          "center_x_frac": 0.50, "frame_id": 5}],
    ]

    for frame in fake_frames:
        output = tracker.update(frame)
        for obj in output:
            print(obj)
