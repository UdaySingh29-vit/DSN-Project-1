# 4-Week Learning Plan — Assistive Navigation Project

**Starting point**: Everyone knows Python. Nobody has done an ML/CV project before.
**Goal**: By end of Week 4, every person is comfortable enough with their role's tools to start actual building in Month 2, and Person 6 understands enough of everyone's part to explain the whole system confidently.

---

## Shared Foundation — Everyone does this in Week 1 (before splitting into roles)

Spend the first week together, same material, so nobody falls behind on basics later.

- **NumPy basics**: arrays, indexing, broadcasting (2–3 hrs) — freeCodeCamp NumPy crash course or NumPy's own official quickstart
- **OpenCV basics**: reading images/video, drawing boxes/text on frames, webcam capture loop (`cv2.VideoCapture`) — this is the backbone every role touches. Use OpenCV's official Python tutorials (just the "Core Operations" + "Video Analysis" sections)
- **What is a neural network, really** (conceptual, not math-heavy): what training vs inference means, what "pretrained weights" and "fine-tuning" mean, what a CNN does differently from a normal neural net. StatQuest's YouTube videos on neural networks are excellent for this — short, visual, no heavy math
- **Git/GitHub basics**: clone, branch, commit, push, pull, resolving a merge conflict — you'll be merging 6 people's code, this will save you real pain later
- **Set up Google Colab**: everyone should be able to run a notebook, connect free GPU, upload/download files

By end of Week 1, everyone should be able to: open a webcam feed in OpenCV, draw a box on screen, and explain in one sentence what "fine-tuning a pretrained model" means.

---

## Person 1 — Object Detection Lead

**Week 2 — Learn the theory + library**
- What object detection actually does (bounding boxes + class labels) vs plain classification
- Learn the **Ultralytics YOLOv8** library specifically (not YOLO theory from scratch) — their own docs/quickstart are enough
- Run pretrained YOLOv8 on sample images/webcam — get comfortable with `model.predict()`, understanding output (boxes, confidence, class)

**Week 3 — Hands-on fine-tuning**
- Learn what a labeled dataset looks like (images + bounding box annotation files)
- Learn **Roboflow** — browsing existing public datasets, exporting in YOLO format
- Practice fine-tuning YOLOv8 on a small public dataset (not your real one yet — just practice, e.g. a toy dataset with 2-3 classes) end-to-end: train → evaluate → run inference
- Learn what mAP (mean average precision) means, how to read a training log

**Week 4 — Apply to your actual project**
- Search Roboflow Universe specifically for stairs/curbs/obstacle datasets
- Do a real trial fine-tune on whatever real data you've found so far
- Document: what classes you'll detect, what dataset you're using, current accuracy baseline

---

## Person 2 — Distance/Depth Estimation Lead

**Week 2 — Camera geometry basics**
- Learn the pinhole camera model conceptually: how real-world size + focal length + pixel size relate to distance (this is just similar-triangles geometry, not advanced math)
- Learn what focal length means practically, how to find/estimate it for a given camera
- Implement a basic "distance from known object size" calculator in Python using a couple of test photos (e.g., estimate distance to a door or a person in a photo you take yourself)

**Week 3 — Depth estimation models**
- Learn what monocular depth estimation is and why it only gives *relative* depth, not metric distance
- Run **MiDaS** (Intel's pretrained depth model) on sample images/webcam via their official GitHub repo — just get it running and visualize the depth map
- Practice combining: run YOLO + MiDaS on the same frame, extract the depth value at each detected object's location

**Week 4 — Calibration and integration**
- Practice combining known-size distance estimates (for person/door/chair) with MiDaS relative depth (for everything else) into one unified distance number
- Test in a real hallway/room: does red/green/blue banding feel right at 3-5m, 5-10m, 10m+? Tune thresholds

---

## Person 3 — Motion Tracking & Object Memory Lead

**Week 2 — Tracking fundamentals**
- Learn what "object tracking" means vs detection (tracking = keeping the same ID across frames)
- Learn IOU (Intersection over Union) — the basic concept trackers use to match boxes frame-to-frame
- Learn **ByteTrack** (or DeepSORT as backup) — just get a demo running on a sample video using their GitHub repo instructions

**Week 3 — Velocity and motion classification**
- Practice computing simple velocity from tracked positions (change in position / change in time) to classify moving vs static
- Learn basic Python data structures for a "memory bank" (dictionaries or simple classes) — practice storing/updating/deleting entries by ID

**Week 4 — Object re-identification**
- Learn similarity matching basics: comparing class + size + position between a "lost" object and a "new" detection to decide if it's the same one reappearing
- Build a small practice version: simulate an object leaving and re-entering frame, write logic to reconnect it to its old memory entry instead of treating it as new

---

## Person 4 — Egomotion / Head-Movement Detection Lead

**Week 2 — Feature tracking basics**
- Learn what optical flow is (tracking pixel movement between two frames) — conceptually, then in OpenCV (`cv2.calcOpticalFlowPyrLK` — Lucas-Kanade method)
- Practice: run optical flow on a webcam feed and visualize motion vectors

**Week 3 — Estimating movement from feature shift**
- Learn about feature detectors (ORB is easiest to start with) — detecting stable points in a frame, matching them between frames
- Practice: measure how much matched features shift between two frames taken a few steps apart — this becomes your rough "has the user moved" signal

**Week 4 — Threshold and trigger logic**
- Build a simple rule: accumulate feature displacement over frames, and when it crosses a threshold, trigger "memory stale, please rescan" event
- Test practically: walk a few steps with a webcam/phone, see if your displacement estimate reasonably reflects real movement

---

## Person 5 — Decision Logic & Voice Interface Lead

**Week 2 — Voice output**
- Learn **pyttsx3** (offline text-to-speech) — get basic "hello world" spoken output working
- Learn **speech_recognition** library for capturing voice commands (for the onboarding "look around" step and on-demand questions)

**Week 3 — Decision logic**
- Learn basic state machine design in Python (simple if you keep it to plain classes/enums — no need for a fancy library)
- Practice writing rules like: "if object distance < threshold AND object is roughly centered in frame → speak; else stay silent"
- Practice combining voice output with conditional logic: TTS only fires under specific conditions, not every frame

**Week 4 — Integrate with teammates' outputs**
- Once Persons 1–4 have working pieces, practice wiring their outputs (detections, distance, motion, displacement) into your decision logic with dummy/sample data
- Refine what gets said and when — avoid "audio spam," prioritize most urgent object first

---

## Person 6 — Integration & Demo Lead (needs breadth across everyone's role)

Since this person needs to explain the whole system, their month looks different — less deep specialization, more breadth plus the actual integration engineering skill.

**Week 2 — Systems engineering basics + breadth pass 1**
- Learn Python threading/multiprocessing basics (why real-time pipelines need this — one slow module shouldn't freeze the whole loop)
- Sit in on Person 1 and Person 2's practice sessions this week; understand *what* YOLO and MiDaS do and *why*, even without running the code yourself in depth

**Week 3 — Breadth pass 2 + integration practice**
- Sit in on Person 3 and Person 4's practice sessions; understand tracking/memory and egomotion at a conceptual level
- Start building the skeleton pipeline: a Python script structure where each module (detection, distance, tracking, egomotion, voice) is a placeholder function — practice plugging in fake/dummy outputs and passing them between stages

**Week 4 — Full integration + presentation prep**
- Sit in on Person 5's practice session; understand the decision/voice logic
- Begin actually wiring together whatever real pieces teammates have working by now (even partial ones) into a single pipeline
- Draft the demo script and presentation narrative: what problem you're solving, why each design choice (known-size distance trick, object memory, silent-unless-relevant) was made, and be ready to answer "how does the depth estimation work" or "how does tracking work" in your own words, not memorized jargon

---

## Weekly Team Sync (recommended for everyone)
Once a week, all 6 people spend 30–45 minutes explaining what they learned to each other in plain language (no jargon dumps). This does two things: keeps Person 6 genuinely up to speed, and forces everyone to actually understand their material well enough to teach it — the best test of real understanding.
