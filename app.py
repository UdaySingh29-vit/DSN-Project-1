import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
import traceback

from detector_depth import DetectionDepthEstimator
from motion_tracker import MotionTracker
from object_memory import ObjectMemoryBank
from egomotion import EgomotionDetector
from decision_engine import DecisionEngine
from speech_manager import SpeechManager

app = Flask(__name__)

# Global variables for sharing data between the thread and flask
latest_frame = None
system_status = {
    "fps": 0,
    "objects": [],
    "logs": []
    
}
lock = threading.Lock()
command_queue = []
is_muted = True  # Default to muted so it doesn't immediately speak on startup

def system_loop():
    global latest_frame, system_status, command_queue
    print("[SYSTEM] Initializing models and modules...")
    
    # 1. Initialize Modules
    try:
        detector = DetectionDepthEstimator(yolo_model_path="yolov8n.pt", enable_midas=False, focal_length=388.3)
        tracker = MotionTracker(fps=15.0) 
        memory = ObjectMemoryBank()
        egomotion = EgomotionDetector()
        decision = DecisionEngine()
        speech = SpeechManager()
    except Exception as e:
        print(f"[ERROR] Failed to initialize core modules: {e}")
        return

    speech.start()
    speech.speak("Web dashboard initialized.", priority=0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    frame_id = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        fps_counter += 1
        now = time.time()

        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now

        # STAGE 1
        try:
            object_cubes, annotated_frame = detector.process_frame(frame, frame_id)
        except Exception as e:
            object_cubes = []
            annotated_frame = frame.copy()
            print(f"[Stage 1 Error]: {e}")

        # STAGE 2
        tracked_objects = []
        ego_status = {"moved_significantly": False}
        try:
            tracked_objects = tracker.update(object_cubes)
            memory.update(tracked_objects, frame_id)
            for obj in tracked_objects:
                if "bbox" in obj and "track_id" in obj:
                    x1, y1, x2, y2 = map(int, obj["bbox"])
                    text = f"ID:{obj['track_id']} {obj['motion_state']}"
                    cv2.putText(annotated_frame, text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception as e:
            pass
            
        try:
            ego_status = egomotion.update(frame, frame_id)
        except Exception:
            pass

        # STAGE 3
        try:
            remembered_objects = memory.get_all_remembered()
            action = decision.update(remembered_objects, ego_status, current_time=now)
            if action.get("should_speak"):
                if not is_muted:
                    speech.speak(action["message"], priority=action["priority"])
                with lock:
                    prefix = "[MUTED] " if is_muted else ""
                    system_status["logs"].insert(0, f"[{time.strftime('%H:%M:%S')}] SYS: {prefix}{action['message']}")
                    if len(system_status["logs"]) > 20:
                        system_status["logs"].pop()
        except Exception:
            pass

        # Handle UI Commands
        if command_queue:
            cmd = command_queue.pop(0)
            resp = decision.handle_user_query(cmd, memory.get_all_remembered())
            if resp.get("should_speak"):
                speech.speak(resp["message"], priority=resp["priority"])
                with lock:
                    system_status["logs"].insert(0, f"[{time.strftime('%H:%M:%S')}] YOU: {cmd}")
                    system_status["logs"].insert(0, f"[{time.strftime('%H:%M:%S')}] SYS: {resp['message']}")
                    if len(system_status["logs"]) > 20:
                        system_status["logs"] = system_status["logs"][:20]

        # Update globals for UI
        with lock:
            latest_frame = annotated_frame.copy()
            
            system_status["fps"] = round(current_fps, 1)
            system_status["is_muted"] = is_muted
            system_status["objects"] = []
            for obj in tracked_objects:
                system_status["objects"].append({
                    "id": obj.get("track_id", "?"),
                    "class": obj.get("class_name", "?"),
                    "distance": obj.get("distance_m", 0),
                    "motion": obj.get("motion_state", "unknown")
                })


# Start the background thread
thread = threading.Thread(target=system_loop, daemon=True)
thread.start()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        frame_copy = None
        with lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()
                
        if frame_copy is not None:
            # Encode outside the lock to prevent blocking the AI thread
            ret, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04) # Cap stream at ~25fps to save CPU overhead
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    with lock:
        return jsonify(system_status)

@app.route('/command', methods=['POST'])
def handle_command():
    data = request.json
    cmd = data.get("command", "")
    if cmd:
        command_queue.append(cmd)
    return jsonify({"status": "success"})

@app.route('/toggle_mute', methods=['POST'])
def toggle_mute():
    global is_muted
    is_muted = not is_muted
    return jsonify({"status": "success", "is_muted": is_muted})

if __name__ == '__main__':
    print("[FLASK] Starting Web Dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
