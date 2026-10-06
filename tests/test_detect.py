from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 model (downloads automatically the first time, ~6MB)
model = YOLO("yolov8n.pt")  # "n" = nano, the fastest variant, good for a laptop webcam

# Open the default webcam (0 = built-in camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on this frame
    results = model(frame, verbose=False)

    # Draw the boxes/labels directly onto the frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Live Detection - press q to quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()