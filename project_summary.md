# Project Summary: Real-Time Assistive Navigation System for the Visually Impaired

## The Problem
Visually impaired individuals rely on canes, guide dogs, or memorized routes to navigate — none of which give real-time awareness of moving obstacles, sudden hazards, or unfamiliar environments. Existing "smart cane" or app solutions are either too simple (basic obstacle beeping) or require expensive hardware (LiDAR-equipped devices). We're building a system that works with just a regular camera and gives rich, intelligent, spoken awareness of surroundings — the kind of situational awareness a sighted person takes for granted.

## What We're Building
A live camera-based system (webcam/phone camera) that:

1. **Detects every object** in the camera's view in real time (people, furniture, doors, obstacles, etc.)
2. **Estimates how far away each object is** — grouped into color-coded danger zones internally (Red = 3–5m, Green = 5–10m, Blue = beyond) — using a known-object-size distance trick plus AI-based depth estimation for anything else
3. **Tracks whether each object is moving or still**, so a parked chair and a walking person are treated very differently
4. **Remembers what it's already seen** — if the user turns their head and looks back, the system recognizes objects it already scanned instead of recalculating everything from zero, updating just their distance/speed
5. **Knows when the user has physically moved** far enough that its "memory" of the room is outdated, and prompts a quick re-scan (turn your head around) to refresh understanding of the space
6. **Speaks only when it matters** — not a constant stream of noise, but targeted spoken alerts only when something is close, in the user's direct path, or moving toward them. Users can also ask it questions on demand ("which way is clear?")

## The Technical Core (in plain terms)
- **Object detection**: a pretrained AI vision model (YOLOv8), fine-tuned to also recognize hazards like stairs and curbs
- **Distance estimation**: combines a physics-based trick (using known average sizes of people/doors/furniture and camera geometry) with an AI depth-estimation model (MiDaS) for anything without a known size
- **Motion tracking**: an object-tracking algorithm that follows detected objects across frames to tell moving from static, and estimate direction/speed
- **Object memory**: a lightweight internal memory bank that "remembers" previously seen objects (so turning your head doesn't cause the system to lose track and start over), expiring memories once objects are far away (30–40m) or the user has moved on
- **Voice interface**: text-to-speech for alerts and guided onboarding, speech recognition for on-demand questions

## Why This Is a Strong Exhibition Project

- **Real, visible social impact** — this isn't an abstract accuracy benchmark, it's solving a problem people actually live with every day. Judges and visitors connect with this immediately, technical or not.
- **Live, dramatic demo potential** — a blindfolded/simulated walkthrough where the system actively guides a person around real obstacles is far more memorable than a slide of numbers.
- **Genuine technical depth, not a template project** — this combines object detection, depth estimation, motion tracking, and lightweight object memory/re-identification — techniques used in real research (including autonomous vehicles), scoped down sensibly to work with just a normal camera. It's not a copy-paste tutorial project.
- **Clear problem-solving story** — every design decision (why color-zone distance bands, why object memory, why speak only when relevant) has a reasoned justification the team can explain confidently, which is exactly what judges probe for.
- **Achievable and demoable in a month** — built entirely on pretrained models fine-tuned for our needs (no training from scratch), so the team can focus energy on the harder, more original pieces: distance estimation, motion logic, and memory — the parts that actually make this stand out.

## Team Structure (6 roles)
1. Object Detection Lead
2. Distance/Depth Estimation Lead
3. Motion Tracking & Object Memory Lead
4. Egomotion/Head-movement Detection Lead
5. Decision Logic & Voice Interface Lead
6. Integration & Demo Lead

## Timeline
- **Week 1**: Core object detection running live; distance estimation math in progress
- **Week 2**: Distance bands working, basic voice alerts, onboarding flow
- **Week 3**: Motion tracking (moving vs static), speak-only-when-relevant logic
- **Week 4**: Object memory + re-scan triggering, real-world testing, demo polish, presentation rehearsal
