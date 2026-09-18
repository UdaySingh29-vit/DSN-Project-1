from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

KNOWN_DISTANCE_M = 2.0     # you will stand exactly this far from the camera
KNOWN_HEIGHT_M = 1.7       # your approximate height in meters (adjust to your real height)

print("Stand exactly 2 meters from the camera, face it, then press 'c' to capture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Calibration - press c to capture, q to quit", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        boxes = results[0].boxes
        names = results[0].names
        person_box = None
        for box in boxes:
            cls_id = int(box.cls[0])
            if names[cls_id] == "person":
                person_box = box
                break

        if person_box is not None:
            x1, y1, x2, y2 = person_box.xyxy[0]
            pixel_height = float(y2 - y1)
            focal_length = (pixel_height * KNOWN_DISTANCE_M) / KNOWN_HEIGHT_M
            print(f"\nDetected pixel height: {pixel_height:.1f}px")
            print(f"YOUR FOCAL LENGTH: {focal_length:.1f}")
            print("Copy this number into distance_estimate.py\n")
        else:
            print("No person detected - make sure you're fully visible and try again.")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()