"""
Object Memory Bank — Person 2, to-do item #3.

Sits downstream of MotionTracker. Its job: remember each tracked object's last
known state even after ByteTrack briefly stops reporting it (user looked away,
object occluded for a moment), so the system doesn't lose context. Entries expire
once the object is too far away to matter, or hasn't been seen in too long.

This is deliberately a separate concern from MotionTracker's internal velocity-
smoothing history (which only needs a few recent frames and gets wiped the
moment a track drops). The memory bank is meant to persist much longer.
"""

from dataclasses import dataclass, field

# ---- Tunable constants (expose, don't hardcode inline) ----
EXPIRY_DISTANCE_M = 40.0        # per spec: forget objects once this far away
MAX_FRAMES_UNSEEN = 45          # ~1.5s at 30fps — tune against real re-entry tests


@dataclass
class MemoryEntry:
    last_state: dict            # the last full tracked_object dict we received
    last_seen_frame: int        # frame_id it was last actually detected on
    frames_unseen: int = 0      # how many update() calls it's been missing for


class ObjectMemoryBank:
    def __init__(self):
        self.bank: dict[int, MemoryEntry] = {}

    def update(self, tracked_objects: list[dict], current_frame_id: int) -> None:
        """
        Call once per frame with MotionTracker's output.
        Refreshes entries for objects seen this frame, ages out ones that weren't.
        """
        seen_ids = set()

        for obj in tracked_objects:
            track_id = obj["track_id"]
            seen_ids.add(track_id)
            self.bank[track_id] = MemoryEntry(
                last_state=obj,
                last_seen_frame=current_frame_id,
                frames_unseen=0,
            )

        # anything in the bank that wasn't in this frame's output just got older
        for track_id, entry in self.bank.items():
            if track_id not in seen_ids:
                entry.frames_unseen += 1

        self._expire_stale_entries()

    def _expire_stale_entries(self) -> None:
        """Drop entries that are too far away or have been missing too long."""
        expired_ids = []
        for track_id, entry in self.bank.items():
            too_far = entry.last_state["distance_m"] > EXPIRY_DISTANCE_M
            too_stale = entry.frames_unseen > MAX_FRAMES_UNSEEN
            if too_far or too_stale:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.bank[track_id]

    def get_known_object(self, track_id: int) -> dict | None:
        """Used by re-identification: 'have I seen this track_id recently?'"""
        entry = self.bank.get(track_id)
        return entry.last_state if entry else None

    def get_all_remembered(self) -> list[dict]:
        """
        Full current memory snapshot — useful for Person 3's on-demand questions
        like 'which way is clear?', since it includes objects not detected THIS
        exact frame but still considered relevant.
        """
        return [entry.last_state for entry in self.bank.values()]

    def stats(self) -> dict:
        """Quick debug/demo helper — handy for judges' Q&A ('how many objects is it tracking?')."""
        return {
            "active_memory_count": len(self.bank),
            "track_ids": list(self.bank.keys()),
        }


# --- Standalone demo: object leaves frame for a while, then reappears ---
if __name__ == "__main__":
    memory = ObjectMemoryBank()

    # Frame 10: person visible, confirmed track
    memory.update([{
        "class_name": "person", "confidence": 0.9, "bbox": (100, 100, 200, 300),
        "distance_m": 4.5, "distance_band": "red", "distance_method": "known_size",
        "center_x_frac": 0.5, "frame_id": 10, "track_id": 1,
        "motion_state": "static", "velocity_mps": 0.0, "direction": "stationary",
    }], current_frame_id=10)
    print("After frame 10:", memory.stats())

    # Frames 11-30: person walks out of view, ByteTrack reports nothing for track_id 1
    for f in range(11, 31):
        memory.update([], current_frame_id=f)
    print("After 20 frames unseen:", memory.stats())
    print("Still remembers track 1?", memory.get_known_object(1) is not None)

    # Frame 60: still nothing — now past MAX_FRAMES_UNSEEN, should expire
    for f in range(31, 61):
        memory.update([], current_frame_id=f)
    print("After 50 frames unseen:", memory.stats())
    print("Still remembers track 1?", memory.get_known_object(1) is not None)
