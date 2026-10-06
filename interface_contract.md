# Interface Contract — Data Formats Between Modules

**Purpose**: every person builds their module to output exactly this shape, and consume exactly this shape from the previous module. If everyone follows this, Week 7 integration becomes "plug pieces together" instead of "renegotiate formats under deadline pressure."

Rule: **use these exact field names.** If a field doesn't apply yet (e.g. distance not computed), use `None`, don't omit the field or rename it.

---

## 1. Detection Module Output (Person 1 → Person 2)

One dictionary per detected object, per frame. A frame produces a **list** of these.

```python
detected_object = {
    "class_name": "person",       # string, YOLO class label
    "confidence": 0.87,           # float, 0-1
    "bbox": (x1, y1, x2, y2),     # pixel coordinates, top-left and bottom-right corners
    "frame_id": 1042,             # int, which frame this came from (use a running counter)
}
```

Output per frame: `List[detected_object]`

---

## 2. Distance Module Output (Person 2 → Cube Builder)

Takes each `detected_object` and adds distance info. Extends the same dictionary rather than creating a separate one — every downstream module gets everything in one place.

```python
detected_object_with_distance = {
    # ...all fields from detected_object above, plus:
    "distance_m": 4.2,                # float, estimated meters (best guess, can be approximate)
    "distance_band": "red",           # string, one of: "red" (3-5m), "green" (5-10m), "blue" (10m+)
    "distance_method": "known_size",  # string, one of: "known_size", "midas_relative" — useful for debugging
}
```

---

## 3. Object Cube (Cube Builder → Tracking Module)

This is the unified internal representation every later module reasons about.

```python
object_cube = {
    "class_name": "person",
    "confidence": 0.87,
    "bbox": (x1, y1, x2, y2),
    "distance_m": 4.2,
    "distance_band": "red",
    "center_x_frac": 0.52,   # float 0-1, horizontal position in frame (0=left edge, 1=right edge) — used later to check "is this in the user's path"
    "frame_id": 1042,
}
```

Output per frame: `List[object_cube]`

---

## 4. Tracking Module Output (Person 3 → Memory Bank)

Adds a persistent ID and motion info to each cube.

```python
tracked_object = {
    # ...all fields from object_cube above, plus:
    "track_id": 17,             # int, stable ID across frames for the same physical object
    "motion_state": "moving",   # string, one of: "moving", "static", "unknown" (unknown = not enough frames yet to tell)
    "velocity_mps": 1.1,        # float, estimated speed in meters/second, 0.0 if static
    "direction": "toward_user", # string, one of: "toward_user", "away_from_user", "lateral", "unknown"
}
```

---

## 5. Object Memory Bank (Person 3, internal state — but Person 6 needs to know its shape)

Not per-frame output — this is a persistent dictionary keyed by `track_id`, updated every frame.

```python
memory_bank = {
    17: {  # key = track_id
        "class_name": "person",
        "last_seen_frame": 1042,
        "last_distance_m": 4.2,
        "last_bbox": (x1, y1, x2, y2),
        "velocity_mps": 1.1,
        "direction": "toward_user",
        "frames_since_seen": 0,   # increments when object isn't detected this frame; used for expiry
    },
    # more track_ids...
}
```

**Expiry rule**: remove an entry when `last_distance_m > 40` OR `frames_since_seen` exceeds an agreed threshold (decide this number together once you're testing — start with something like 90 frames at your camera's fps, adjust from real testing).

---

## 6. Egomotion Module Output (Person 4 → Decision Logic + Memory Bank)

One value computed per frame, not per object — this describes the *user's* movement, not any single object.

```python
egomotion_status = {
    "frame_id": 1042,
    "displacement_score": 0.34,      # float, accumulated feature-shift magnitude since last reset — units are arbitrary/relative, tune threshold empirically
    "moved_significantly": False,    # bool, True once displacement_score crosses your agreed threshold
}
```

**When `moved_significantly` becomes `True`**: this is the signal that triggers (a) wiping the memory bank and (b) the voice prompt asking the user to look around again. After the rescan completes, reset `displacement_score` to 0.

---

## 7. Decision Logic Output (Person 5 → Voice Interface)

Takes the full list of `tracked_object`s (or memory bank entries) plus `egomotion_status`, and decides what to say, if anything.

```python
speech_action = {
    "should_speak": True,
    "message": "person, 3 meters, ahead",   # string, the exact sentence to speak
    "priority": 1,                          # int, lower = more urgent, used if multiple objects qualify in the same frame (only speak the top priority one, don't stack messages)
    "trigger_reason": "in_path_close",      # string, one of: "in_path_close", "moving_toward_user", "rescan_prompt", "user_query_response"
}
```

If nothing qualifies to be spoken this frame: `{"should_speak": False, "message": None, "priority": None, "trigger_reason": None}`

---

## 8. Voice Interface (Person 5 — consumes `speech_action`, no further downstream module)

Just consumes the dict above: if `should_speak` is `True`, call TTS with `message`. No new contract needed beyond what's above.

---

## Frame Loop Summary — who calls what, in order

```python
# Pseudocode Person 6 will build the real version of
for frame in camera_stream:
    detections = person1_detect(frame)                          # -> List[detected_object]
    with_distance = person2_estimate_distance(detections, frame) # -> List[detected_object_with_distance]
    cubes = build_cubes(with_distance)                           # -> List[object_cube]
    tracked = person3_track(cubes)                               # -> List[tracked_object], updates memory_bank
    ego_status = person4_egomotion(frame)                        # -> egomotion_status

    if ego_status["moved_significantly"]:
        memory_bank.clear()
        trigger_rescan_prompt()

    action = person5_decide(tracked, ego_status)                 # -> speech_action
    if action["should_speak"]:
        speak(action["message"])
```

---

## What to do if a field genuinely doesn't fit
If your role turns up a real need to change a field name or add something new, don't just change it silently in your own code — flag it in the weekly sync so everyone updates together. A silent field-name change is exactly the kind of bug that costs a full day to track down in Week 7.
