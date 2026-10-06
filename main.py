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
from voice_listener import VoiceListener

def main():
    print("[SYSTEM] Initializing models and modules...")
    
    # 1. Initialize Modules
    try:
        # Pass the calibrated focal length, disable MiDaS, and use the Nano model to maximize FPS
        detector = DetectionDepthEstimator(yolo_model_path="yolov8n.pt", enable_midas=False, focal_length=388.3)
        tracker = MotionTracker(fps=15.0) 
        memory = ObjectMemoryBank()
        egomotion = EgomotionDetector()
        decision = DecisionEngine()
        speech = SpeechManager()
    except Exception as e:
        print(f"[ERROR] Failed to initialize core modules: {e}")
        traceback.print_exc()
        return

    try:
        voice = VoiceListener()
    except Exception as e:
        print(f"[WARNING] VoiceListener failed to initialize. Voice commands disabled: {e}")
        voice = None

    # Start Speech and Voice Threads
    speech.start()
    if voice:
        voice.start()
    speech.speak("Navigation system initialized. Starting camera.", priority=0)

    # 2. Open Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        speech.speak("Camera failed to open.", priority=1)
        speech.stop()
        return

    print("[SYSTEM] Pipeline running. Press 'q' to quit.")
    print("[SYSTEM] You can speak commands like 'what is ahead' or 'which way is clear'.")
    print("[SYSTEM] Or press 'c' to ask 'which way is clear?'. Press 'a' to ask 'what is ahead?'.")

    frame_id = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0
    
    last_spoken_text = ""
    last_spoken_time = 0.0

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
            
            # Draw tracking IDs and motion states so they appear in screenshots
            for obj in tracked_objects:
                if "bbox" in obj and "track_id" in obj:
                    x1, y1, x2, y2 = map(int, obj["bbox"])
                    text = f"ID:{obj['track_id']} {obj['motion_state']}"
                    cv2.putText(annotated_frame, text, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception as e:
            print(f"[Stage 2 Error] Tracking failed on frame {frame_id}: {e}")
            
        try:
            ego_status = egomotion.update(frame, frame_id)
        except Exception as e:
            print(f"[Stage 2 Error] Egomotion failed on frame {frame_id}: {e}")

        # STAGE 3: Decision Engine & Voice (Person 3)
        try:
            remembered_objects = memory.get_all_remembered()
            action = decision.update(remembered_objects, ego_status, current_time=now)
            if action.get("should_speak"):
                speech.speak(action["message"], priority=action["priority"])
                last_spoken_text = f"SPOKE: {action['message']}"
                last_spoken_time = now
        except Exception as e:
            print(f"[Stage 3 Error] Decision logic failed on frame {frame_id}: {e}")

        # Handle Keyboard and Voice Inputs (do this before drawing final text)
        key = cv2.waitKey(1) & 0xFF
        
        voice_query = voice.get_latest_command() if voice else None
        
        if key == ord('q'):
            break
        elif key == ord('s') or key == ord('S'):
            import os
            filename = f"screenshot_{frame_id}.png"
            cv2.imwrite(filename, annotated_frame)
            print(f"[SYSTEM] Screenshot saved as {filename} in {os.getcwd()}")
            # Brief visual feedback on screen
            cv2.putText(annotated_frame, "SCREENSHOT SAVED!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.imshow("Assistive Navigation System - Team 195", annotated_frame)
            cv2.waitKey(500) # pause for half a second to show the text
        elif voice_query:
            resp = decision.handle_user_query(voice_query, memory.get_all_remembered())
            if resp.get("should_speak"):
                speech.speak(resp["message"], priority=resp["priority"])
                last_spoken_text = f"HEARD: {voice_query} | SPOKE: {resp['message']}"
                last_spoken_time = now
        elif key == ord('c') or key == ord('C'):
            resp = decision.handle_user_query("which way is clear", memory.get_all_remembered())
            if resp.get("should_speak"):
                speech.speak(resp["message"], priority=resp["priority"])
                last_spoken_text = f"SPOKE: {resp['message']}"
                last_spoken_time = now
        elif key == ord('a') or key == ord('A'):
            resp = decision.handle_user_query("what is ahead", memory.get_all_remembered())
            if resp.get("should_speak"):
                speech.speak(resp["message"], priority=resp["priority"])
                last_spoken_text = f"SPOKE: {resp['message']}"
                last_spoken_time = now

        # DISPLAY
        color = (0, 255, 0) if current_fps >= 10 else (0, 0, 255)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
        if now - last_spoken_time < 3.0 and last_spoken_text:
            cv2.putText(annotated_frame, last_spoken_text, (20, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Assistive Navigation System - Team 195", annotated_frame)

    # Clean Shutdown
    print("[SYSTEM] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    speech.speak("System powering down.", priority=0)
    if voice:
        voice.stop()
    speech.stop()

if __name__ == "__main__":
    main()