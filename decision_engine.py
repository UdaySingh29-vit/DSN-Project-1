"""Decision logic for the Real-Time Assistive Navigation System.
Person 3 module.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, Iterable, List, Optional

SILENT_ACTION: Dict[str, Any] = {
    "should_speak": False,
    "message": "",
    "priority": 0,
    "trigger_reason": "none",
}

ALLOWED_TRIGGER_REASONS = {
    "in_path_close",
    "moving_toward_user",
    "rescan_prompt",
    "user_query_response",
    "onboarding",
    "none",
}

def no_speech_action() -> Dict[str, Any]:
    return dict(SILENT_ACTION)

def speech_action(message: str, priority: int, trigger_reason: str) -> Dict[str, Any]:
    if trigger_reason not in ALLOWED_TRIGGER_REASONS:
        raise ValueError(f"Unsupported trigger reason: {trigger_reason}")
    return {
        "should_speak": True,
        "message": str(message).strip(),
        "priority": int(priority),
        "trigger_reason": trigger_reason,
    }

def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def valid_distance(value: Any) -> Optional[float]:
    distance = safe_float(value)
    return distance if distance is not None and distance > 0 else None

def normalized_x(value: Any) -> Optional[float]:
    x = safe_float(value)
    if x is None or not 0.0 <= x <= 1.0:
        return None
    return x

def class_label(obj: Dict[str, Any]) -> str:
    name = obj.get("class_name")
    if not isinstance(name, str) or not name.strip():
        return "Obstacle"
    return name.strip().replace("_", " ").capitalize()

def location_from_x(center_x_frac: Any, in_path_min: float, in_path_max: float) -> str:
    x = normalized_x(center_x_frac)
    if x is None or in_path_min <= x <= in_path_max:
        return "ahead"
    return "to your left" if x < in_path_min else "to your right"

def format_distance(distance_m: Any) -> Optional[str]:
    distance = valid_distance(distance_m)
    if distance is None:
        return None
    if distance < 1.0:
        return "very close"
    rounded = max(1, round(distance))
    unit = "meter" if rounded == 1 else "meters"
    return f"{rounded} {unit}"

def format_alert_message(
    obj: Dict[str, Any],
    approaching: bool,
    in_path_min: float,
    in_path_max: float,
) -> str:
    label = class_label(obj)
    location = location_from_x(obj.get("center_x_frac"), in_path_min, in_path_max)
    distance_text = format_distance(obj.get("distance_m"))

    if approaching:
        if location == "ahead":
            return f"{label} approaching ahead" + (f", {distance_text}." if distance_text else ".")
        side = "from the left" if location == "to your left" else "from the right"
        return f"{label} approaching {side}" + (f", {distance_text}." if distance_text else ".")

    if distance_text == "very close":
        return f"{label} very close {location}."
    if distance_text:
        return f"{label}, {distance_text} {location}."
    return "Obstacle ahead." if location == "ahead" else f"{label} {location}."

def evaluate_object(
    obj: Dict[str, Any],
    in_path_min: float,
    in_path_max: float,
    close_distance_m: float,
) -> Dict[str, Any]:
    x = normalized_x(obj.get("center_x_frac"))
    distance = valid_distance(obj.get("distance_m"))
    distance_band = str(obj.get("distance_band", "")).lower()
    direction = str(obj.get("direction", "unknown")).lower()
    motion_state = str(obj.get("motion_state", "unknown")).lower()
    velocity = safe_float(obj.get("velocity_mps"))

    in_path = x is not None and in_path_min <= x <= in_path_max
    close = (distance is not None and distance <= close_distance_m) or (
        distance is None and distance_band == "red"
    )

    approaching = direction == "toward_user" or (
        direction == "unknown" and motion_state == "moving" and velocity is not None and velocity > 0.15
    )

    qualifies = (in_path and close) or approaching

    if approaching and in_path and close:
        priority = 1
    elif approaching and close:
        priority = 2
    elif in_path and close:
        priority = 3
    elif approaching:
        priority = 4
    else:
        priority = 99

    reason = "moving_toward_user" if approaching else "in_path_close" if in_path and close else "none"

    proximity_score = 0.0 if distance is None else max(0.0, close_distance_m + 2.0 - distance)
    score = (100 - priority * 10) + proximity_score

    return {
        "obj": obj,
        "track_id": obj.get("track_id"),
        "distance": distance,
        "in_path": in_path,
        "is_close": close,
        "approaching": approaching,
        "qualifies": qualifies,
        "priority": priority,
        "score": score,
        "trigger_reason": reason,
    }

def choose_best(evaluations: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [item for item in evaluations if item["qualifies"]]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["priority"],
            item["distance"] if item["distance"] is not None else float("inf"),
            str(item["track_id"]),
        ),
    )

class DecisionEngine:
    def __init__(
        self,
        in_path_min: float = 0.30,
        in_path_max: float = 0.70,
        close_distance_m: float = 3.0,
        alert_cooldown_seconds: float = 1.5,
        same_object_repeat_seconds: float = 3.0,
        rescan_cooldown_seconds: float = 5.0,
        escalation_distance_m: float = 0.75,
        approaching_velocity_threshold_mps: float = 0.15,
    ) -> None:
        if not (0.0 <= in_path_min < in_path_max <= 1.0):
            raise ValueError("in-path bounds must satisfy 0 <= min < max <= 1")
        if close_distance_m <= 0:
            raise ValueError("close_distance_m must be positive")

        self.in_path_min = in_path_min
        self.in_path_max = in_path_max
        self.close_distance_m = close_distance_m
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.same_object_repeat_seconds = same_object_repeat_seconds
        self.rescan_cooldown_seconds = rescan_cooldown_seconds
        self.escalation_distance_m = escalation_distance_m
        self.approaching_velocity_threshold_mps = approaching_velocity_threshold_mps

        self._last_alert_time = float("-inf")
        self._last_track_id: Any = None
        self._last_distance: Optional[float] = None
        self._last_priority = 99
        self._last_message: Optional[str] = None
        self._last_rescan_time = float("-inf")
        self._previous_moved_significantly = False

    def update(
        self,
        tracked_objects: Optional[List[Dict[str, Any]]],
        egomotion_status: Optional[Dict[str, Any]],
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = time.monotonic() if current_time is None else current_time
        objects = tracked_objects if isinstance(tracked_objects, list) else []
        status = egomotion_status if isinstance(egomotion_status, dict) else {}

        evaluations = [
            evaluate_object(obj, self.in_path_min, self.in_path_max, self.close_distance_m)
            for obj in objects
            if isinstance(obj, dict)
        ]
        best = choose_best(evaluations)

        if best is not None and self._should_emit_object_alert(best, now):
            message = format_alert_message(
                best["obj"],
                best["approaching"],
                self.in_path_min,
                self.in_path_max,
            )
            action = speech_action(message, best["priority"], best["trigger_reason"])
            self._record_object_alert(best, message, now)
            return action

        moved = bool(status.get("moved_significantly", False))
        rescan_event = moved and not self._previous_moved_significantly
        self._previous_moved_significantly = moved

        if (
            best is None
            and rescan_event
            and now - self._last_rescan_time >= self.rescan_cooldown_seconds
            and now - self._last_alert_time >= self.alert_cooldown_seconds
        ):
            self._last_rescan_time = now
            self._last_alert_time = now
            self._last_track_id = None
            self._last_distance = None
            self._last_priority = 99
            self._last_message = "Please look around again before moving."
            return speech_action(self._last_message, 5, "rescan_prompt")

        return no_speech_action()

    def _should_emit_object_alert(self, best: Dict[str, Any], now: float) -> bool:
        if now - self._last_alert_time < self.alert_cooldown_seconds:
            return False

        same_track = best["track_id"] is not None and best["track_id"] == self._last_track_id
        if not same_track:
            return True

        if now - self._last_alert_time >= self.same_object_repeat_seconds:
            return True

        new_distance = best["distance"]
        distance_dropped = (
            self._last_distance is not None
            and new_distance is not None
            and self._last_distance - new_distance >= self.escalation_distance_m
        )
        priority_increased = best["priority"] < self._last_priority
        return distance_dropped or priority_increased

    def _record_object_alert(self, best: Dict[str, Any], message: str, now: float) -> None:
        self._last_alert_time = now
        self._last_track_id = best["track_id"]
        self._last_distance = best["distance"]
        self._last_priority = best["priority"]
        self._last_message = message

    def handle_user_query(
        self,
        query_text: Optional[str],
        tracked_objects: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        text = (query_text or "").strip().lower()
        objects = tracked_objects if isinstance(tracked_objects, list) else []

        if "which way" in text or "which side" in text or "clear" in text:
            return self.which_way_is_clear(objects)
        if "what is ahead" in text or text == "ahead":
            return self.what_is_ahead(objects)
        if "repeat" in text:
            message = self._last_message or "There is nothing to repeat yet."
            return speech_action(message, 6, "user_query_response")
        return no_speech_action()

    def which_way_is_clear(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        regions: Dict[str, List[Dict[str, Any]]] = {"left": [], "ahead": [], "right": []}
        for obj in tracked_objects:
            if not isinstance(obj, dict):
                continue
            x = normalized_x(obj.get("center_x_frac"))
            if x is None:
                continue
            if x < self.in_path_min:
                regions["left"].append(obj)
            elif x > self.in_path_max:
                regions["right"].append(obj)
            else:
                regions["ahead"].append(obj)

        risks = {name: self._region_risk_score(items) for name, items in regions.items()}
        lowest_risk = min(risks.values())
        safest = [name for name, risk in risks.items() if risk == lowest_risk]

        if all(not items for items in regions.values()):
            message = "No significant obstacles are detected. Please proceed carefully."
        elif len(safest) == 1:
            phrases = {
                "left": "The left side appears clearer.",
                "ahead": "The area ahead appears clearer.",
                "right": "The right side appears clearer.",
            }
            message = phrases[safest[0]]
        else:
            message = "No single direction appears clearly better. Please proceed carefully."
        return speech_action(message, 6, "user_query_response")

    def _region_risk_score(self, objects: List[Dict[str, Any]]) -> float:
        risk = 0.0
        for obj in objects:
            evaluation = evaluate_object(
                obj,
                self.in_path_min,
                self.in_path_max,
                self.close_distance_m,
            )
            distance = evaluation["distance"]
            if distance is None:
                risk += 8.0
            else:
                risk += max(0.0, self.close_distance_m + 1.0 - distance) * 20.0
            if evaluation["approaching"]:
                risk += 80.0
            elif str(obj.get("motion_state", "")).lower() == "moving":
                risk += 10.0
        return risk

    def what_is_ahead(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        ahead = []
        for obj in tracked_objects:
            if not isinstance(obj, dict):
                continue
            x = normalized_x(obj.get("center_x_frac"))
            if x is not None and self.in_path_min <= x <= self.in_path_max:
                ahead.append(
                    evaluate_object(obj, self.in_path_min, self.in_path_max, self.close_distance_m)
                )

        if not ahead:
            return speech_action("Nothing significant is detected ahead.", 6, "user_query_response")

        best = choose_best(ahead)
        if best is None:
            best = min(
                ahead,
                key=lambda item: item["distance"] if item["distance"] is not None else float("inf"),
            )
        message = format_alert_message(
            best["obj"], best["approaching"], self.in_path_min, self.in_path_max
        )
        return speech_action(message, 6, "user_query_response")