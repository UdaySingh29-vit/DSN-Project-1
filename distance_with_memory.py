from ultralytics import YOLO
import cv2
import time
from collections import deque

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

FOCAL_LENGTH = 322  # <-- your calibrated value
FRAME_EDGE_MARGIN = 8

KNOWN_HEIGHTS = {
    "person": 1.7,
    "chair": 0.9,
    "couch": 0.85,
    "tv": 0.5,
    "backpack": 0.45,
}

# How many recent readings to keep per class, for velocity smoothing
HISTORY_LEN = 8
# If we haven't had a real (non-clipped) reading in this many seconds, stop trusting extrapolation
MAX_EXTRAPOLATION_AGE = 1.5

# history[class_name] = deque of (timestamp, distance_m)
history = {}

def get_distance_band(distance_m):
    if distance_m <= 5:
        return "RED", (0, 0, 255)
    elif distance_m <= 10:
        return "GREEN", (0, 200, 0)
    else:
        return "BLUE", (255, 100, 0)

def compute_velocity(hist):
    """Average closing speed (m/s) from consecutive readings. Negative = approaching."""
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
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)  # simple moving average - smooths out noisy single-frame jitter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names
    annotated = frame.copy()
    frame_height = frame.shape[0]

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name not in KNOWN_HEIGHTS:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (150, 150, 150), 2)
            cv2.putText(annotated, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            continue

        if class_name not in history:
            history[class_name] = deque(maxlen=HISTORY_LEN)

        is_clipped = (y1 <= FRAME_EDGE_MARGIN) or (y2 >= frame_height - FRAME_EDGE_MARGIN)

        if not is_clipped:
            pixel_height = y2 - y1
            real_height = KNOWN_HEIGHTS[class_name]
            if pixel_height > 0:
                distance_m = (real_height * FOCAL_LENGTH) / pixel_height
                history[class_name].append((now, distance_m))
                band, color = get_distance_band(distance_m)
                label = f"{class_name} {distance_m:.1f}m [{band}]"
            else:
                color = (200, 200, 200)
                label = class_name
        else:
            hist = history[class_name]
            if len(hist) >= 2:
                last_time, last_distance = hist[-1]
                age = now - last_time
                if age <= MAX_EXTRAPOLATION_AGE:
                    velocity = compute_velocity(hist)
                    predicted_distance = max(0.3, last_distance + velocity * age)
                    label = f"{class_name} ~{predicted_distance:.1f}m [EXTRAPOLATED]"
                    color = (0, 165, 255)
                else:
                    label = f"{class_name} <1m [RED - very close, stale]"
                    color = (0, 0, 255)
            else:
                label = f"{class_name} <1m [RED - very close]"
                color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Distance + Memory - press q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()