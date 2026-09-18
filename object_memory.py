"""
Object Memory Bank — Person 2.
"""

from dataclasses import dataclass, field

# ---- Tunable constants ----
EXPIRY_DISTANCE_M = 40.0        # per spec: forget objects once this far away
MAX_FRAMES_UNSEEN = 45          # ~1.5s at 30fps

@dataclass
class MemoryEntry:
    last_state: dict            # the last full tracked_object dict we received
    last_seen_frame: int        # frame_id it was last actually detected on
    frames_unseen: int = 0      # how many update() calls it's been missing for


class ObjectMemoryBank:
    def __init__(self):
        self.bank: dict[int, MemoryEntry] = {}

    def update(self, tracked_objects: list[dict], current_frame_id: int) -> None:
        seen_ids = set()

        for obj in tracked_objects:
            track_id = obj["track_id"]
            seen_ids.add(track_id)
            self.bank[track_id] = MemoryEntry(
                last_state=obj,
                last_seen_frame=current_frame_id,
                frames_unseen=0,
            )

        for track_id, entry in self.bank.items():
            if track_id not in seen_ids:
                entry.frames_unseen += 1

        self._expire_stale_entries()

    def _expire_stale_entries(self) -> None:
        expired_ids = []
        for track_id, entry in self.bank.items():
            too_far = entry.last_state["distance_m"] > EXPIRY_DISTANCE_M
            too_stale = entry.frames_unseen > MAX_FRAMES_UNSEEN
            if too_far or too_stale:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.bank[track_id]

    def get_known_object(self, track_id: int) -> dict | None:
        entry = self.bank.get(track_id)
        return entry.last_state if entry else None

    def get_all_remembered(self) -> list[dict]:
        return [entry.last_state for entry in self.bank.values()]

    def stats(self) -> dict:
        return {
            "active_memory_count": len(self.bank),
            "track_ids": list(self.bank.keys()),
        }