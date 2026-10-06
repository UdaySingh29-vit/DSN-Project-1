import threading
import queue
import time
import numpy as np
import sounddevice as sd

try:
    import speech_recognition as sr
except ImportError:
    sr = None

class VoiceListener:
    def __init__(self):
        if sr is None:
            raise ImportError("SpeechRecognition is not installed.")
            
        self.recognizer = sr.Recognizer()
        self.command_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # Listen in 3-second blocks

    def start(self):
        if self.is_running:
            return
            
        try:
            # Query devices to ensure audio is working
            sd.query_devices()
            print("[VoiceListener] SoundDevice initialized successfully. Listening for commands...")
        except Exception as e:
            print(f"[VoiceListener] Microphone initialization failed: {e}")
            return
            
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._listen_loop,
            name="VoiceListenerThread",
            daemon=True
            
        )
        self._worker_thread.start()
        
    def stop(self, timeout=2.0):
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            
    @property
    def is_running(self):
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def get_latest_command(self):
        commands = []
        while not self.command_queue.empty():
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        if commands:
            return commands[-1]
        return None

    def _listen_loop(self):
        while not self._stop_event.is_set():
            try:
                # Record a chunk of audio
                audio_data = sd.rec(int(self.chunk_duration * self.sample_rate), 
                                    samplerate=self.sample_rate, 
                                    channels=1, 
                                    dtype='int16')
                
                # Wait until recording is finished, checking stop event occasionally
                for _ in range(int(self.chunk_duration * 10)):
                    if self._stop_event.is_set():
                        sd.stop()
                        return
                    time.sleep(0.1)
                
                # Check for silence to save API calls
                if np.max(np.abs(audio_data)) < 500:
                    continue
                
                # Convert raw PCM int16 to AudioData for SpeechRecognition
                audio_bytes = audio_data.tobytes()
                sr_audio = sr.AudioData(audio_bytes, self.sample_rate, 2)
                
                # Perform recognition
                text = self.recognizer.recognize_google(sr_audio).lower().strip()
                print(f"\n[VoiceListener] Heard: '{text}'")
                self.command_queue.put(text)
                
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"\n[VoiceListener] API Error: {e}")
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"\n[VoiceListener] Unexpected Error: {e}")
