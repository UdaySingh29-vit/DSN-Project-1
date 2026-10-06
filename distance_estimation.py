from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

FOCAL_LENGTH = 388.3# REPLACE with the number calibrate.py printed for you

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
                frame_h, frame_w = frame.shape[:2]
                # Check if bounding box is cut off by the top or bottom of the frame
                is_clipped = (y1 <= 10) or (y2 >= frame_h - 10)
                
                distance_m = (real_height * FOCAL_LENGTH) / pixel_height
                
                if is_clipped:
                    # If you are so close your whole body doesn't fit, the true pixel height 
                    # would be larger. This means true distance is less than calculated distance.
                    # We estimate it's closer by a factor (e.g., 0.6)
                    distance_m = distance_m * 0.6
                    label = f"{class_name} {distance_m:.1f}m [CLIPPED]"
                else:
                    label = f"{class_name} {distance_m:.1f}m"

                band, color = get_distance_band(distance_m)
                label += f" [{band}]"
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
