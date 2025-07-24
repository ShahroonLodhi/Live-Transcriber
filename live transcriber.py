import whisper # type: ignore
import pyaudio # type: ignore
import wave
import threading
import queue
import numpy as np
from collections import deque
import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio configuration settings"""
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    channels: int = 1
    sample_rate: int = 16000
    chunk_duration: float = 5.0
    overlap_duration: float = 1.0
    language: str = "en"


@dataclass
class VADConfig:
    energy_threshold: float = 0.01
    min_speech_duration: float = 0.5
    max_silence_duration: float = 0.5
    zcr_min: float = 0.01
    zcr_max: float = 0.3


class VoiceActivityDetector:
    
    def __init__(self, config: VADConfig):
        self.config = config
        
    def detect_speech(self, audio_data: np.ndarray) -> bool:
        audio_float = audio_data.astype(np.float32)
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_float ** 2))
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float))))
        zcr = zero_crossings / len(audio_float) if len(audio_float) > 0 else 0
        
        # Voice typically has higher energy and moderate ZCR
        return (energy > self.config.energy_threshold and 
                self.config.zcr_min < zcr < self.config.zcr_max)


class LiveTranscriber:
    
    def __init__(self, 
                 model_size: str = "small",
                 audio_config: Optional[AudioConfig] = None,
                 vad_config: Optional[VADConfig] = None):
       
        self.model_size = model_size
        self.audio_config = audio_config or AudioConfig()
        self.vad_config = vad_config or VADConfig()
        
        # Initialize components
        self._setup_logging()
        self._load_whisper_model()
        self._setup_audio()
        self._setup_vad()
        self._setup_buffers()
        self._setup_control_flags()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def _load_whisper_model(self):
        self.logger.info(f"Loading Whisper model: {self.model_size}")
        try:
            self.model = whisper.load_model(self.model_size)
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
            
    def _setup_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def _setup_vad(self):
        self.vad = VoiceActivityDetector(self.vad_config)
        self.min_speech_samples = int(
            self.audio_config.sample_rate * self.vad_config.min_speech_duration
        )
        self.max_silence_chunks = int(
            self.audio_config.sample_rate * self.vad_config.max_silence_duration / 
            self.audio_config.chunk_size
        )
        
    def _setup_buffers(self):
        self.speech_buffer = deque()
        self.audio_queue = queue.Queue()
        
    def _setup_control_flags(self):
        self.is_recording = False
        self.is_transcribing = False
        self.is_speech_active = False
        self.silence_counter = 0
        self.recording_thread = None
        self.transcription_thread = None
        
    def start_recording(self):
        try:
            self.stream = self.audio.open(
                format=self.audio_config.format,
                channels=self.audio_config.channels,
                rate=self.audio_config.sample_rate,
                input=True,
                frames_per_buffer=self.audio_config.chunk_size
            )
            
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self.logger.info("Started recording with Voice Activity Detection")
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            raise
            
    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(
                    self.audio_config.chunk_size, 
                    exception_on_overflow=False
                )
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                self._process_audio_chunk(audio_data)
                
            except Exception as e:
                self.logger.error(f"Recording error: {e}")
                break
                
    def _process_audio_chunk(self, audio_data: np.ndarray):
        has_speech = self.vad.detect_speech(audio_data)
        
        if has_speech:
            self.speech_buffer.extend(audio_data)
            self.is_speech_active = True
            self.silence_counter = 0
            
        elif self.is_speech_active:
            self.speech_buffer.extend(audio_data)
            self.silence_counter += 1
            
            # End speech segment after enough silence
            if self.silence_counter >= self.max_silence_chunks:
                self._process_speech_segment()
                self._reset_speech_state()
                
        # Process if buffer gets too long
        max_samples = self.audio_config.sample_rate * self.audio_config.chunk_duration
        if len(self.speech_buffer) >= max_samples:
            self._process_speech_segment()
            self._reset_speech_state()
            
    def _process_speech_segment(self):
        if len(self.speech_buffer) >= self.min_speech_samples:
            speech_data = np.array(list(self.speech_buffer))
            self.audio_queue.put(speech_data)
            
        self.speech_buffer.clear()
        
    def _reset_speech_state(self):
        self.is_speech_active = False
        self.silence_counter = 0
        
    def start_transcription(self, callback: Optional[Callable] = None):
        self.is_transcribing = True
        self.transcription_thread = threading.Thread(
            target=self._transcribe_audio,
            args=(callback,)
        )
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        self.logger.info("Started transcription")
        
    def _transcribe_audio(self, callback: Optional[Callable] = None):
        while self.is_transcribing:
            try:
                # Wait for audio chunk
                audio_chunk = self.audio_queue.get(timeout=1)
                
                start_time = time.time()
                
                # Convert and normalize audio
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                
                # Transcribe
                result = self.model.transcribe(
                    audio_float,
                    language=self.audio_config.language,
                    task="transcribe",
                    fp16=False
                )
                
                text = result["text"].strip()
                
                if text:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {text}")
                    
                    if callback:
                        callback(text, timestamp, result)
                        
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Transcription error: {e}")
                
    def start_live_transcription(self, callback: Optional[Callable] = None):
        self.start_recording()
        time.sleep(0.5)  # Allow recording to stabilize
        self.start_transcription(callback)
        
    def stop(self):
        self.logger.info("Stopping transcription...")
        
        # Process any remaining speech
        if (self.speech_buffer and 
            len(self.speech_buffer) >= self.min_speech_samples):
            self._process_speech_segment()
            
        # Stop threads
        self.is_recording = False
        self.is_transcribing = False
        
        # Clean up audio resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.audio.terminate()
        
        self.logger.info("Transcription stopped")
        


def transcription_callback(text: str, timestamp: str, full_result: Dict[str, Any]):
    try:
        if text.strip():
            with open("live_transcription.txt", "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text.strip()}\n")
    except Exception as e:
        logging.error(f"Error in transcription callback: {e}")


def main():
    # Configure transcriber
    audio_config = AudioConfig(
        chunk_duration=5.0,
        overlap_duration=1.5,
        language="en"
    )
    
    vad_config = VADConfig(
        energy_threshold=0.01,
        min_speech_duration=0.4
    )
    
    transcriber = LiveTranscriber(
        model_size="small",
        audio_config=audio_config,
        vad_config=vad_config
    )
    
    try:
        # Start live transcription
        transcriber.start_live_transcription(transcription_callback)
        
        print("\n" + "="*50)
        print("   - Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        # Interactive loop for adjusting settings
        while True:
            try:
                user_input = input().strip().lower()
                if user_input == 's':
                    current_threshold = transcriber.vad_config.energy_threshold
                    print(f"Current VAD threshold: {current_threshold}")
                    try:
                        new_threshold = float(input(
                            "Enter new VAD threshold (0.001-0.1, lower=more sensitive): "
                        ))
                        transcriber.adjust_sensitivity(energy_threshold=new_threshold)
                    except ValueError:
                        print("Invalid input. Keeping current settings.")
                        
            except EOFError:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        transcriber.stop()
        print("\n✅ Transcription stopped.")


if __name__ == "__main__":
    main()