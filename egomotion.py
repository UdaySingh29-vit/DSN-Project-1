"""
Egomotion Detection module — Person 2.
Uses OpenCV Lucas-Kanade optical flow to estimate camera movement.
"""

import cv2
import numpy as np

# ---- Tunable constants ----
MIN_FEATURES = 15
MOVEMENT_THRESHOLD = 50.0
ACCUMULATOR_DECAY = 0.9
FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class EgomotionDetector:
    def __init__(self):
        self.old_gray = None
        self.p0 = None
        self.displacement_accumulator = 0.0

    def update(self, frame: np.ndarray, frame_id: int) -> dict:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        moved_significantly = False
        current_displacement = 0.0

        if self.old_gray is None or self.p0 is None or len(self.p0) < MIN_FEATURES:
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **FEATURE_PARAMS)
            self.old_gray = frame_gray.copy()
            
            if self.p0 is None or len(self.p0) < MIN_FEATURES:
                return {
                    "frame_id": frame_id,
                    "displacement_score": 0.0,
                    "moved_significantly": False,
                    "status": "unreliable_low_features"
                }

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **LK_PARAMS)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            if len(good_new) >= MIN_FEATURES:
                distances = np.linalg.norm(good_new - good_old, axis=1)
                current_displacement = float(np.median(distances))
                
                self.displacement_accumulator = (self.displacement_accumulator * ACCUMULATOR_DECAY) + current_displacement

                if self.displacement_accumulator > MOVEMENT_THRESHOLD:
                    moved_significantly = True
                    self.displacement_accumulator = 0.0
            
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **FEATURE_PARAMS)
        else:
            self.p0 = None 

        return {
            "frame_id": frame_id,
            "displacement_score": round(self.displacement_accumulator, 2),
            "moved_significantly": moved_significantly,
            "status": "reliable"
        }