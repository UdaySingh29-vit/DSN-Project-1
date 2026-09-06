"""
Integration test for MotionTracker + ObjectMemoryBank.

Checks the three things flagged before handoff:
  1. Multi-object correctness — 3 simultaneous objects, including overlapping
     boxes, to stress-test the bbox-based remapping fix in motion_tracker.py.
  3. Per-frame performance profile of the tracker stage in isolation.
  5. Memory bank wired to tracker output, including an object that disappears
     for a while (simulating occlusion / user looking away) then checking
     whether it's still remembered vs. correctly expired later.

Run: python integration_test.py
"""

import time
import statistics

from motion_tracker import MotionTracker
from object_memory import ObjectMemoryBank

FPS = 30.0
NUM_FRAMES = 70
PRINT_AT_FRAMES = {3, 4, 5, 20, 21, 40, 65, 70}  # frames worth inspecting closely

# Rough sanity budget: spec target is >=10 FPS for the WHOLE pipeline (100ms/frame
# total, shared across detection + distance + tracking + voice). This isn't your
# real allotted share of that budget — just a loose upper bound to catch anything
# wildly slow early. Tighten this once Person 1 tells you your actual budget.
FULL_PIPELINE_BUDGET_MS = 100.0


def generate_frame(frame_id: int) -> list[dict]:
    """Builds one frame of fake object_cube data with 2-3 simultaneous objects."""
    objects = []

    # Object A: person walking toward the camera the whole time (~0.6 m/s closing)
    person_distance = round(6.0 - (frame_id - 1) * 0.02, 3)
    objects.append({
        "class_name": "person", "confidence": 0.90,
        "bbox": (100.0 + frame_id, 100.0, 200.0 + frame_id, 300.0),
        "distance_m": person_distance,
        "distance_band": "red" if person_distance < 5.0 else "green",
        "distance_method": "known_size",
        "center_x_frac": 0.50,
        "frame_id": frame_id,
    })

    # Object B: static chair. Its bbox deliberately overlaps the person's box
    # around frame ~50-70 as the person walks across — this is what actually
    # stresses whether the bbox-remapping fix holds up under overlap.
    objects.append({
        "class_name": "chair", "confidence": 0.85,
        "bbox": (150.0, 150.0, 250.0, 350.0),
        "distance_m": 7.0,
        "distance_band": "green",
        "distance_method": "depth_model",
        "center_x_frac": 0.60,
        "frame_id": frame_id,
    })

    # Object C: a door, present only frames 1-20, then gone — simulates the
    # user turning away / the object leaving frame, to test memory bank recall
    # vs. eventual expiry.
    if frame_id <= 20:
        objects.append({
            "class_name": "door", "confidence": 0.80,
            "bbox": (400.0, 50.0, 500.0, 400.0),
            "distance_m": 10.0,
            "distance_band": "blue",
            "distance_method": "depth_model",
            "center_x_frac": 0.85,
            "frame_id": frame_id,
        })

    return objects


def main():
    tracker = MotionTracker(fps=FPS)
    memory = ObjectMemoryBank()
    frame_times_ms = []

    for frame_id in range(1, NUM_FRAMES + 1):
        cubes = generate_frame(frame_id)

        # --- Point 3: profile this stage in isolation ---
        start = time.perf_counter()
        tracked = tracker.update(cubes)
        elapsed_ms = (time.perf_counter() - start) * 1000
        frame_times_ms.append(elapsed_ms)

        # --- Point 5: wire tracker output into the memory bank ---
        memory.update(tracked, current_frame_id=frame_id)

        if frame_id in PRINT_AT_FRAMES:
            print(f"\n--- frame {frame_id} ---")
            for obj in tracked:
                print(" ", obj)
            print("  memory stats:", memory.stats())

    print("\n=== Performance profile: MotionTracker.update() only ===")
    print(f"frames processed : {len(frame_times_ms)}")
    print(f"avg              : {statistics.mean(frame_times_ms):.3f} ms")
    print(f"median           : {statistics.median(frame_times_ms):.3f} ms")
    print(f"max              : {max(frame_times_ms):.3f} ms")
    print(f"min              : {min(frame_times_ms):.3f} ms")

    over_budget = [t for t in frame_times_ms if t > FULL_PIPELINE_BUDGET_MS]
    print(f"frames over the full {FULL_PIPELINE_BUDGET_MS:.0f}ms pipeline "
          f"budget (rough sanity check only): {len(over_budget)}")

    print("\n=== Final memory bank contents (frame 70) ===")
    # Track IDs are assigned by ByteTrack internally, not fixed here — don't
    # assume which int belongs to the door. Instead check by class_name: if the
    # door's class doesn't appear at all, it correctly expired after being gone
    # since frame 20 (50 frames unseen > MAX_FRAMES_UNSEEN=45).
    remaining = memory.get_all_remembered()
    remaining_classes = [obj["class_name"] for obj in remaining]
    print("Objects still remembered:", remaining_classes)
    print("Door correctly expired?", "yes" if "door" not in remaining_classes else "no (still remembered)")


if __name__ == "__main__":
    main()
