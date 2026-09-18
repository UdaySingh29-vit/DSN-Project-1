from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

FOCAL_LENGTH = 322# REPLACE with the number calibrate.py printed for you

# Average real-world heights (meters) for common classes - approximate, good enough for banding
KNOWN_HEIGHTS = {
    "person": 1.7,
    "chair": 0.9,
    "couch": 0.85,
    "tv": 0.5,
    "backpack": 0.45,
}

def get_distance_band(distance_m):
    if distance_m <= 5:
        return "RED", (0, 0, 255)
    elif distance_m <= 10:
        return "GREEN", (0, 200, 0)
    else:
        return "BLUE", (255, 100, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names
    annotated = frame.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name in KNOWN_HEIGHTS:
            pixel_height = y2 - y1
            real_height = KNOWN_HEIGHTS[class_name]
            if pixel_height > 0:
                distance_m = (real_height * FOCAL_LENGTH) / pixel_height
                band, color = get_distance_band(distance_m)
                label = f"{class_name} {distance_m:.1f}m [{band}]"
            else:
                color = (200, 200, 200)
                label = class_name
        else:
            color = (150, 150, 150)
            label = class_name

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Distance Estimation - press q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()