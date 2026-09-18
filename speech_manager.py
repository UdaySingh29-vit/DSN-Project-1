"""Thread-safe, non-blocking offline Text-to-Speech manager.
Person 3 module.
"""

from __future__ import annotations
import itertools
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

@dataclass(order=True)
class _SpeechRequest:
    priority: int
    sequence: int
    message: str = field(compare=False)

class SpeechManager:
    def __init__(self, rate: int = 170, volume: float = 1.0, max_queue_size: int = 1):
        if pyttsx3 is None:
            raise ImportError("pyttsx3 is not installed. Run: pip install pyttsx3")
        if max_queue_size != 1:
            raise ValueError("max_queue_size must be 1 to prevent stale alert backlogs")
        if not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")

        self.rate = rate
        self.volume = volume
        self._queue: queue.Queue[Optional[_SpeechRequest]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._is_speaking = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._sequence = itertools.count()

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @property
    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="SpeechManagerWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def speak(self, message: str, priority: int = 99) -> bool:
        if not isinstance(message, str) or not message.strip() or self._stop_event.is_set():
            return False
        if not self.is_running:
            self.start()

        try:
            priority = int(priority)
        except (TypeError, ValueError):
            priority = 99

        request = _SpeechRequest(priority, next(self._sequence), message.strip())

        try:
            pending = self._queue.get_nowait()
        except queue.Empty:
            pending = None

        if pending is not None and pending.priority < request.priority:
            chosen = pending
        else:
            chosen = request

        try:
            self._queue.put_nowait(chosen)
            return chosen is request
        except queue.Full:
            return False

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if request is None:
                break

            with self._lock:
                self._is_speaking = True
            try:
                self._speak_blocking(request.message)
            except Exception as exc:
                print(f"[SpeechManager] TTS error: {exc}")
            finally:
                with self._lock:
                    self._is_speaking = False

    def _speak_blocking(self, message: str) -> None:
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(message)
            engine.runAndWait()
        finally:
            if engine is not None:
                engine.stop()