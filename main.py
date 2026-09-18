"""
Real-Time Assistive Navigation System - Main Integration Pipeline
Integrates Stage 1, Stage 2, and Stage 3.
"""

import cv2
import time
import traceback

# --- Import all modules ---
from detector_depth import DetectionDepthEstimator
from motion_tracker import MotionTracker
from object_memory import ObjectMemoryBank
from egomotion import EgomotionDetector
from decision_engine import DecisionEngine
from speech_manager import SpeechManager

def main():
    print("[SYSTEM] Initializing models and modules...")
    
    # 1. Initialize Modules
    try:
        detector = DetectionDepthEstimator(enable_midas=True)
        tracker = MotionTracker(fps=15.0) 
        memory = ObjectMemoryBank()
        egomotion = EgomotionDetector()
        decision = DecisionEngine()
        speech = SpeechManager()
    except Exception as e:
        print(f"[ERROR] Failed to initialize modules: {e}")
        traceback.print_exc()
        return

    # Start Speech Thread
    speech.start()
    speech.speak("Navigation system initialized. Starting camera.", priority=0)

    # 2. Open Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        speech.speak("Camera failed to open.", priority=1)
        speech.stop()
        return

    print("[SYSTEM] Pipeline running. Press 'q' to quit.")
    print("[SYSTEM] Press 'c' to ask 'which way is clear?'. Press 'a' to ask 'what is ahead?'.")

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

        # STAGE 1: Detection & Distance (Person 1)
        object_cubes = []
        annotated_frame = frame.copy()
        try:
            object_cubes, annotated_frame = detector.process_frame(frame, frame_id)
        except Exception as e:
            print(f"[Stage 1 Error] Detection failed on frame {frame_id}: {e}")

        # STAGE 2: Tracking, Memory, and Egomotion (Person 2)
        tracked_objects = []
        ego_status = {"moved_significantly": False}
        
        try:
            tracked_objects = tracker.update(object_cubes)
            memory.update(tracked_objects, frame_id)
        except Exception as e:
            print(f"[Stage 2 Error] Tracking failed on frame {frame_id}: {e}")
            
        try:
            ego_status = egomotion.update(frame, frame_id)
        except Exception as e:
            print(f"[Stage 2 Error] Egomotion failed on frame {frame_id}: {e}")

        # STAGE 3: Decision Engine & Voice (Person 3)
        try:
            action = decision.update(tracked_objects, ego_status, current_time=now)
            if action.get("should_speak"):
                speech.speak(action["message"], priority=action["priority"])
                
                cv2.putText(annotated_frame, f"SPOKE: {action['message']}", (20, frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        except Exception as e:
            print(f"[Stage 3 Error] Decision logic failed on frame {frame_id}: {e}")

        # DISPLAY & INPUT HANDLING
        color = (0, 255, 0) if current_fps >= 10 else (0, 0, 255)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Assistive Navigation System - Team 195", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            resp = decision.handle_user_query("which way is clear", tracked_objects)
            if resp.get("should_speak"):
                speech.speak(resp["message"], priority=resp["priority"])
        elif key == ord('a'):
            resp = decision.handle_user_query("what is ahead", tracked_objects)
            if resp.get("should_speak"):
                speech.speak(resp["message"], priority=resp["priority"])

    # Clean Shutdown
    print("[SYSTEM] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    speech.speak("System powering down.", priority=0)
    speech.stop()

if __name__ == "__main__":
    main()