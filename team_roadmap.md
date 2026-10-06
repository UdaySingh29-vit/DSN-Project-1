# Project Roadmap — Assistive Navigation System

This is the full journey from where you are now to exhibition day, broken into phases with the crucial checkpoints that decide whether you're on track.

---

## Phase 0 — Foundation (Done)
- Idea locked: real-time assistive navigation using object detection, distance estimation, motion tracking, object memory, and voice guidance
- Roles assigned across 6 people
- Team summary written so everyone's aligned on scope

---

## Phase 1 — Learning Month (Weeks 1–4)
Covered in your learning plan. The crucial checkpoint here isn't "did everyone finish the material" — it's:

**🔑 Checkpoint 1 (end of Week 1)**: Can everyone open a webcam feed in OpenCV and explain what "fine-tuning a pretrained model" means in their own words? If not, don't move forward — this foundation is load-bearing for every role.

**🔑 Checkpoint 2 (end of Week 4)**: Can each person run their core tool on sample data (YOLO detecting objects, MiDaS producing a depth map, ByteTrack tracking a video, optical flow measuring motion, TTS speaking a sentence)? This is the "do I actually have working hands-on skill" test, not just "did I watch tutorials."

If checkpoint 2 isn't solid, it's better to extend learning by a few days than to start building on shaky footing — module bugs are 10x harder to debug when the person also doesn't understand the underlying tool.

---

## Phase 2 — Build Month (Weeks 5–8)

### Week 5 — Standalone modules working in isolation
Each person builds their piece **separately**, not yet connected to anyone else's code. This is crucial: debugging is far easier when only one thing can be wrong.
- Person 1: YOLO detecting real objects live on webcam
- Person 2: distance estimation producing a number for a detected box
- Person 3: tracking assigning consistent IDs across frames
- Person 4: displacement/motion signal from optical flow
- Person 5: TTS speaking test phrases + basic decision rule stub
- Person 6: pipeline skeleton with placeholder functions for each module, so everyone knows exactly what input/output shape their function needs to match

**🔑 Checkpoint 3 (end of Week 5)**: Every person has a working standalone demo of their piece — even if crude. This is your "go/no-go" for integration. Don't start wiring things together if even one module isn't producing real output yet — fix that first.

### Week 6 — Pairwise integration
Instead of everyone merging into one system at once (chaos), integrate in pairs first:
- Detection + Distance (Person 1 + 2) → now every detected object has a distance
- Tracking + Egomotion (Person 3 + 4) → now objects have motion state AND the system knows if the user moved
- Decision Logic + Voice (Person 5) starts consuming fake/sample combined data from the above pairs

**🔑 Checkpoint 4 (mid Week 6)**: Detection+Distance pair and Tracking+Egomotion pair each work end-to-end on their own. This halves your integration risk before the full merge.

### Week 7 — Full system integration
Person 6 leads wiring all pieces into the single real-time loop (as sketched in the workflow diagram). This week is usually the hardest — expect friction (mismatched data formats, timing/latency issues, one module slowing down the whole loop).
- Build the full loop end-to-end, even if rough
- Get it running on a real hallway/room walk-through, not just a static test image

**🔑 Checkpoint 5 (end of Week 7)**: The full pipeline runs live, start to finish, even if it's not perfect — detects, estimates distance, tracks motion, remembers objects, and speaks something reasonable. This is your "we have a project" milestone. If this isn't working by end of week 7, you cut stretch features immediately (spatial audio, curve guidance, OCR) rather than risk having nothing working for exhibition.

### Week 8 — Testing, polish, and presentation
- Real-world testing across different environments (indoor hallway, doorway, stairs if possible, varying lighting)
- Log and fix failure cases (false alarms, missed objects, laggy audio)
- Tune thresholds (distance bands, speaking triggers) based on real test feedback, not guesswork
- Build final demo script — decide exactly what you'll show live vs pre-recorded backup (record a backup video in case live demo hardware fails on exhibition day — this is a real risk, don't skip it)
- Presentation rehearsal — everyone should be able to answer questions about their own module, and Person 6 should be able to answer questions about the whole system

**🔑 Checkpoint 6 (few days before exhibition)**: Full dry run — treat it exactly like exhibition day, live demo plus Q&A practice with people outside your team throwing questions at you.

---

## Team Process — How you work together throughout

- **Weekly sync** (carry this over from the learning month): 30–45 min, everyone explains progress and blockers in plain language
- **Git workflow**: one shared repo, each person works on their own branch, merge into a `dev` branch regularly (don't let branches diverge for weeks — merge conflicts get exponentially worse the longer you wait)
- **Shared interface contracts**: before Week 5 starts, agree as a team on exactly what data format passes between modules (e.g., what fields does a "detected object" dictionary contain — class, box coordinates, distance, track ID). Person 6 should own and document this, since they're the one integrating everything. Getting this agreed early avoids painful rework in Week 7
- **Daily/every-other-day quick check-ins during Weeks 5–7** (even just a 5-minute message thread): "what did you finish, what are you stuck on" — catches blockers before they cost days
- **Fallback mindset**: treat Tier 3/4 stretch features (spatial audio, curve guidance, OCR) as bonus, not requirements. Protect your Tier 1/2 core so you always have something solid to demo, even if the ambitious extras don't fully land in time

---

## The Single Most Important Rule
**A working simple system beats a broken ambitious one.** If Week 7's checkpoint shows you're behind, cut scope immediately rather than pushing everything to the last days. Judges reward a system that reliably does what it claims, demoed confidently, over one that's ambitious on paper but glitches live.
