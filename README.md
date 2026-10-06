# Real-Time Assistive Navigation System

An AI-powered navigation aid that helps visually impaired users understand their surroundings in real time — detecting obstacles, estimating distance, tracking motion, and speaking only when it matters. Built to run on a regular camera, with no specialized hardware required.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Visually impaired individuals commonly rely on canes and guide dogs for navigation. While these aids provide effective real-time assistance, they have limitations in detecting and communicating information about distant or dynamically changing objects and unfamiliar surroundings. This project combines object detection, monocular depth estimation, motion tracking, and lightweight object memory into a single real-time pipeline— — using only a standard webcam or phone camera — to give users spoken, situational awareness of their environment as they move.

This system is designed as a **supplement** to existing mobility aids, not a replacement.

---

## Core Features

- **Real-time object detection** — identifies people, furniture, doors, stairs, curbs, and other everyday obstacles
- **Distance estimation** — combines known-object-size geometry with AI-based monocular depth estimation to sort objects into proximity zones
- **Motion tracking** — distinguishes moving objects (e.g. a walking person) from static ones (e.g. a parked chair)
- **Object memory** — remembers previously seen objects so turning your head doesn't reset the system's understanding
- **Egomotion detection** — recognizes when the user has physically moved enough to warrant refreshing their surroundings
- **Smart voice alerts** — speaks only when an object is close, in the user's path, or approaching — not constant narration
- **On-demand voice queries** — users can ask the system questions about their surroundings

---

## Tech Stack

| Component | Tool/Library |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Depth estimation | MiDaS / Distance Heuristics |
| Motion tracking | Custom Object Memory & Tracker |
| Egomotion estimation | OpenCV optical flow (Lucas-Kanade) |
| Text-to-speech | pyttsx3 |
| Speech recognition | speech_recognition |
| Web Dashboard | Flask, HTML, Vanilla CSS & JS |
| Core language | Python |

---

## System Architecture

```
Camera Frame
   → Object Detection (YOLOv8)
   → Distance Estimation (known-size heuristic + MiDaS)
   → Object Cube Construction (class, size, position, distance)
   → Motion Tracking + Object Memory (ByteTrack)
   → Egomotion Check (optical flow) → triggers re-scan if user moved significantly
   → Decision Logic (in-path / close / approaching?)
   → Voice Output (speak alert or stay silent)
```

See `/docs` for the full workflow diagram and interface contract detailing exact data formats passed between modules.

---

## Project Structure

```text
├── app.py                     # Main Flask web server & application entry point
├── detector_depth.py          # YOLOv8 object detection & smart distance logic
├── decision_engine.py         # NLP keyword parsing & alert logic
├── motion_tracker.py          # Custom motion tracking logic
├── object_memory.py           # Object memory bank for persistence
├── voice_listener.py          # Speech recognition and audio input
├── speech_manager.py          # Text-to-speech manager for system voice
├── static/                    # CSS and JavaScript for the web dashboard
├── templates/                 # HTML templates for the web dashboard
├── tests/                     # Test scripts and graph generators
├── yolov8n.pt                 # YOLOv8 nano model weights
└── README.md
```

---

## Setup

```bash
git clone https://github.com/UdaySingh29-vit/DSN-Project-1.git
cd DSN-Project-1
pip install flask ultralytics opencv-python pyttsx3 SpeechRecognition setuptools 

# To run the local server and start the camera:
python app.py
```

Once running, the AI pipeline will start in the background. Open your web browser and navigate to `http://127.0.0.1:5000` to view the Live System Dashboard!

---

## Team

| Role | Contribution |
|---|---|
| Object Detection Lead | Real-time object detection using YOLOv8, fine-tuned for hazard-specific classes |
| Distance/Depth Estimation Lead | Distance estimation combining known-object-size geometry and MiDaS |
| Motion Tracking & Object Memory Lead | Object tracking, moving/static classification, memory bank & re-identification |
| Egomotion Lead | User movement detection via optical flow, triggers memory refresh |
| Decision Logic & Voice Interface Lead | Alert prioritization, silent-unless-relevant logic, TTS/STT voice interface |
| Integration & Demo Lead | Full pipeline integration, interface contract design, live demo |

---

## Roadmap

- [x] Ideation and system design
- [x] Team learning phase (Python/CV/ML foundations)
- [ ] Standalone module development
- [ ] Pairwise module integration
- [ ] Full pipeline integration
- [ ] Real-world testing and threshold tuning
- [ ] Exhibition demo and presentation

---

## Impact & Benefits

| Impact/Benefit | Description |
|---|---|
| **Independence** | Enables visually impaired users to safely navigate unknown environments without needing constant human assistance. |
| **Silent Intelligence** | The system avoids overwhelming the user with constant narration. It only speaks up when something is approaching, in the direct path, or dangerously close. |
| **Real-Time Responsiveness** | Built using the fast YOLOv8n model and optimized for high FPS so users can react instantly to dynamic environments. |

---

## References & Research Links

| Topic | Link |
|---|---|
| YOLOv8 (Ultralytics) | https://docs.ultralytics.com/ |
| MiDaS (Monocular Depth Estimation) | https://github.com/isl-org/MiDaS |
| ByteTrack | https://github.com/ifzhang/ByteTrack |


---

## License

*(Add your chosen license here — MIT is a common permissive choice for academic/exhibition projects.)*
