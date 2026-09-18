"""
Module: Detection & Distance/Depth Estimation (Person 1)
Role: Runs YOLOv8 detection and hybrid distance estimation (Pinhole + MiDaS relative depth),
      producing a standardized List[object_cube] conforming to the Stage 1 contract.
"""

import time
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class DetectionDepthEstimator:
    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        focal_length: float = 800.0,
        enable_midas: bool = True,
        conf_threshold: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.focal_length = focal_length
        self.frame_edge_margin = 8
        self.max_extrapolation_age = 1.5
        self.history_len = 8

        # Reference physical heights in meters
        self.known_heights = {
            "person": 1.7,
            "chair": 0.9,
            "couch": 0.85,
            "tv": 0.5,
            "backpack": 0.45,
        }

        # History keyed by identifier (track_id or class_name fallback)
        self.history: Dict[Any, deque] = {}

        # Load YOLOv8
        self.detector = YOLO(yolo_model_path)

        # MiDaS setup
        self.enable_midas = enable_midas
        self.midas = None
        self.midas_transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rolling_midas_scale: Optional[float] = None
        self.last_scale_update_time: float = 0.0

        if self.enable_midas:
            self._init_midas()

    def _init_midas(self):
        """Initializes MiDaS Small for CPU/GPU relative depth estimation."""
        try:
            # Bypass SSL/HTTP verification hurdles if needed
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.midas.to(self.device)
            self.midas.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.midas_transform = midas_transforms.small_transform
        except Exception as e:
            print(f"[Warning] MiDaS failed to load ({e}). Falling back to geometric distance only.")
            self.enable_midas = False

    def _compute_velocity(self, hist: deque) -> float:
        """Calculates closing velocity (m/s). Negative = approaching."""
        if len(hist) < 2:
            return 0.0
        diffs = []
        items = list(hist)
        for i in range(1, len(items)):
            t0, d0 = items[i - 1]
            t1, d1 = items[i]
            dt = t1 - t0
            if dt > 0:
                diffs.append((d1 - d0) / dt)
        return float(np.mean(diffs)) if diffs else 0.0

    def _get_distance_band(self, distance_m: float) -> str:
        """Returns distance band according to system schema: red | green | blue."""
        if distance_m <= 5.0:
            return "red"
        elif distance_m <= 10.0:
            return "green"
        return "blue"

    def _run_midas(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Infers dense relative disparity map from RGB frame."""
        if not self.enable_midas or self.midas is None:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.midas_transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Executes Detection and Distance Estimation on a single camera frame.
        """
        now = time.time()
        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()

        # 1. Run YOLOv8 detection
        results = self.detector(frame, verbose=False, conf=self.conf_threshold)[0]
        boxes = results.boxes

        # 2. Run MiDaS disparity if enabled
        disparity_map = self._run_midas(frame) if self.enable_midas else None

        candidates = []
        calibration_anchors = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
            pixel_height = y2 - y1
            center_x_frac = round(((x1 + x2) / 2.0) / frame_w, 3)

            candidates.append({
                "class_name": class_name,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2),
                "center_x_frac": center_x_frac,
                "pixel_height": pixel_height,
                "is_clipped": (y1 <= self.frame_edge_margin) or (y2 >= frame_h - self.frame_edge_margin),
            })

        # 3. First Pass: Compute geometric distances for known classes
        for cand in candidates:
            c_name = cand["class_name"]
            c_key = c_name  # Until track_id is provided by Person 2

            if c_key not in self.history:
                self.history[c_key] = deque(maxlen=self.history_len)

            cand["distance_m"] = None
            cand["distance_method"] = "unknown"

            if c_name in self.known_heights:
                real_height = self.known_heights[c_name]
                if not cand["is_clipped"] and cand["pixel_height"] > 0:
                    distance_m = (real_height * self.focal_length) / cand["pixel_height"]
                    cand["distance_m"] = round(distance_m, 2)
                    cand["distance_method"] = "known_size"
                    self.history[c_key].append((now, distance_m))

                    # Collect anchor for MiDaS metric scaling
                    if disparity_map is not None:
                        x1, y1, x2, y2 = cand["bbox"]
                        roi_disp = disparity_map[y1:y2, x1:x2]
                        if roi_disp.size > 0:
                            med_disp = float(np.median(roi_disp))
                            if med_disp > 0:
                                calibration_anchors.append(distance_m * med_disp)
                else:
                    # Extrapolation when edge-clipped
                    hist = self.history[c_key]
                    if len(hist) >= 2:
                        last_time, last_distance = hist[-1]
                        age = now - last_time
                        if age <= self.max_extrapolation_age:
                            vel = self._compute_velocity(hist)
                            pred_dist = max(0.3, last_distance + vel * age)
                            cand["distance_m"] = round(pred_dist, 2)
                            cand["distance_method"] = "extrapolated"
                        else:
                            cand["distance_m"] = 0.8
                            cand["distance_method"] = "extrapolated"
                    else:
                        cand["distance_m"] = 0.9
                        cand["distance_method"] = "extrapolated"

        # Update dynamic calibration scale factor
        if calibration_anchors:
            current_scale = float(np.median(calibration_anchors))
            if self.rolling_midas_scale is None:
                self.rolling_midas_scale = current_scale
            else:
                self.rolling_midas_scale = 0.7 * self.rolling_midas_scale + 0.3 * current_scale
            self.last_scale_update_time = now

        # 4. Second Pass: Estimate distance via MiDaS for objects without geometric distance
        is_scale_stale = (now - self.last_scale_update_time) > 2.0
        for cand in candidates:
            if cand["distance_m"] is None:
                if disparity_map is not None and self.rolling_midas_scale is not None and not is_scale_stale:
                    x1, y1, x2, y2 = cand["bbox"]
                    roi_disp = disparity_map[y1:y2, x1:x2]
                    if roi_disp.size > 0:
                        disp_val = float(np.median(roi_disp))
                        if disp_val > 1e-3:
                            approx_dist = self.rolling_midas_scale / disp_val
                            cand["distance_m"] = round(float(np.clip(approx_dist, 0.3, 20.0)), 2)
                            cand["distance_method"] = "midas_relative"
                
                # Default safety fallback
                if cand["distance_m"] is None:
                    cand["distance_m"] = 6.0
                    cand["distance_method"] = "unknown"

        # 5. Build standard object_cube list and annotate frame
        object_cubes: List[Dict[str, Any]] = []
        for cand in candidates:
            dist_m = cand["distance_m"]
            band = self._get_distance_band(dist_m)

            cube = {
                "class_name": cand["class_name"],
                "confidence": cand["confidence"],
                "bbox": cand["bbox"],
                "distance_m": dist_m,
                "distance_band": band,
                "distance_method": cand["distance_method"],
                "center_x_frac": cand["center_x_frac"],
                "frame_id": frame_id,
            }
            object_cubes.append(cube)

            # Debug Visualization
            x1, y1, x2, y2 = cand["bbox"]
            color = (0, 0, 255) if band == "red" else ((0, 200, 0) if band == "green" else (255, 100, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cand['class_name']} {dist_m}m [{cand['distance_method'][:4]}]"
            cv2.putText(annotated, lbl, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return object_cubes, annotated