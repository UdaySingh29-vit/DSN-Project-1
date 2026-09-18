"""
Motion Tracking module — Person 2, item #2 on the to-do list.

Takes Person 1's per-frame `object_cube` list as input, runs ByteTrack to assign
persistent track_ids, and computes motion_state / velocity_mps / direction from
the distance_m history of each track.
"""

from collections import deque, defaultdict
import numpy as np
import supervision as sv

# ---- Tunable constants (per person2.md spec) ----
IOU_MATCH_THRESHOLD = 0.3
TRACK_CONFIRM_FRAMES = 3
STATIC_VELOCITY_THRESHOLD_MPS = 0.15
VELOCITY_SMOOTHING_WINDOW = 5
LATERAL_MOVEMENT_THRESHOLD = 0.03


class MotionTracker:
    def __init__(self, fps: float):
        self.fps = fps
        self.tracker = sv.ByteTrack(
            minimum_matching_threshold=IOU_MATCH_THRESHOLD,
            minimum_consecutive_frames=TRACK_CONFIRM_FRAMES,
        )
        self.history = defaultdict(lambda: deque(maxlen=VELOCITY_SMOOTHING_WINDOW))

    def update(self, object_cubes: list[dict]) -> list[dict]:
        if not object_cubes:
            detections = sv.Detections.empty()
            self.tracker.update_with_detections(detections)
            return []

        boxes = np.array([obj["bbox"] for obj in object_cubes], dtype=float)
        confidences = np.array([obj["confidence"] for obj in object_cubes], dtype=float)
        class_names = [obj["class_name"] for obj in object_cubes]
        class_ids = np.array([self._class_id(name) for name in class_names])

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )

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
                continue
                
            frame_id = source_obj["frame_id"]
            distance_m = source_obj["distance_m"]
            center_x_frac = source_obj["center_x_frac"]

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
        hist = self.history[track_id]
        if len(hist) < 2:
            return 0.0, "stationary"

        first_frame, first_dist, first_x = hist[0]
        last_frame, last_dist, last_x = hist[-1]
        frame_delta = last_frame - first_frame
        if frame_delta <= 0:
            return 0.0, "stationary"

        time_delta_s = frame_delta / self.fps
        radial_velocity = (first_dist - last_dist) / time_delta_s
        lateral_velocity = (last_x - first_x) / time_delta_s

        speed = abs(radial_velocity)

        if speed >= STATIC_VELOCITY_THRESHOLD_MPS:
            direction = "toward_user" if radial_velocity > 0 else "away_from_user"
            return radial_velocity, direction

        if abs(lateral_velocity) >= LATERAL_MOVEMENT_THRESHOLD:
            direction = "left_to_right" if lateral_velocity > 0 else "right_to_left"
            return lateral_velocity, direction

        return radial_velocity, "stationary"

    def _expire_stale_tracks(self, active_ids) -> None:
        active_ids = set(int(i) for i in active_ids)
        for stale_id in list(self.history.keys()):
            if stale_id not in active_ids:
                del self.history[stale_id]

    _class_name_to_id: dict = {}

    def _class_id(self, name: str) -> int:
        if name not in self._class_name_to_id:
            self._class_name_to_id[name] = len(self._class_name_to_id)
        return self._class_name_to_id[name]