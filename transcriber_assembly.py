import queue
import pyaudio  # type: ignore
import webrtcvad  # type: ignore
from multiprocessing import Process, Queue, Event
import signal
import time
import threading
import websocket
import json
from urllib.parse import urlencode
from datetime import datetime

# AssemblyAI config
API_KEY = "********************"
ASSEMBLY_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,
    "speaker_labels": True
}

WS_URL = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(ASSEMBLY_PARAMS)}"

# Audio configuration
RATE = 16000
CHUNK_MS = 20
CHUNK_SIZE = int(RATE * CHUNK_MS / 1000)
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Buffering logic
MIN_BUFFER_MS = 200
MAX_BUFFER_MS = 500
SILENCE_BUFFER_MS = 100

MIN_BUFFER_SIZE = int(RATE * MIN_BUFFER_MS / 1000) * 2
MAX_BUFFER_SIZE = int(RATE * MAX_BUFFER_MS / 1000) * 2
SILENCE_BUFFER_SIZE = int(RATE * SILENCE_BUFFER_MS / 1000) * 2

vad = webrtcvad.Vad(1)
audio_interface = pyaudio.PyAudio()

def audio_capture_loop(audio_q: Queue, stop_event: Event):
    """Captures and buffers audio, sending only voiced segments."""
    stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    audio_buffer = b""
    speech_frames = 0
    silence_frames = 0

    print("🎙️ Audio capture started...")
    try:
        while not stop_event.is_set():
            frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_buffer += frame
            is_speech = vad.is_speech(frame, RATE)

            if is_speech:
                speech_frames += 1
                silence_frames = 0
                if len(audio_buffer) >= MIN_BUFFER_SIZE:
                    if not audio_q.full():
                        audio_q.put(audio_buffer)
                        audio_buffer = b""
            else:
                silence_frames += 1
                if silence_frames >= 20 and len(audio_buffer) >= SILENCE_BUFFER_SIZE:
                    if not audio_q.full():
                        audio_q.put(audio_buffer)
                        audio_buffer = b""
                        speech_frames = silence_frames = 0
                elif len(audio_buffer) >= MAX_BUFFER_SIZE:
                    if not audio_q.full():
                        audio_q.put(audio_buffer)
                        audio_buffer = b""
                        speech_frames = silence_frames = 0
    except Exception as e:
        print(f"Audio error: {e}")
    finally:
        if audio_buffer and not audio_q.full():
            audio_q.put(audio_buffer)
        stream.stop_stream()
        stream.close()


def assembly_ws_client(audio_q: Queue, stop_event: Event):
    """Handles the AssemblyAI WebSocket client."""
    def on_message(ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            if msg_type == "Begin":
                session_id = data.get('id')
                print(f"\n🔗 Session started: {session_id}")
            elif msg_type == "Turn":
                utterances = data.get("utterances", [])
                if utterances:
                    for u in utterances:
                        speaker = u.get("speaker", "unknown")
                        text = u.get("text", "").strip()
                        if text:
                            print(f"\r🗣️ Speaker {speaker}: {text}")
                else:
                    # fallback if no utterances, show unformatted text
                    text = data.get("transcript", "").strip()
                    if text:
                        print(f"\r📝 {text}")

                else:
                    print(f"\r🔄 {text}", end='', flush=True)
            elif msg_type == "Termination":
                print(f"\n🔚 Session ended.")
        except Exception as e:
            print(f"⚠️ Error parsing message: {e}")

    def on_error(ws, error):
        print(f"\n🚨 WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"\n🔌 WebSocket closed: {close_status_code} - {close_msg}")

    def on_open(ws):
        print("✅ Connected to AssemblyAI WebSocket.")

        def send_loop():
            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.1)
                    ws.send(chunk, websocket.ABNF.OPCODE_BINARY)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Send error: {e}")
                    break

        send_thread = threading.Thread(target=send_loop, daemon=True)
        send_thread.start()

    ws = websocket.WebSocketApp(
        WS_URL,
        header={"Authorization": API_KEY},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    try:
        ws.run_forever(ping_interval=30, ping_timeout=10)
    except Exception as e:
        if not stop_event.is_set():
            print(f"🛑 WS connection failed: {e}")


def signal_handler(signum, frame):
    print("\n🛑 Ctrl+C received. Stopping...")
    global stop_flag
    stop_flag.set()


def main():
    global stop_flag
    stop_flag = Event()
    signal.signal(signal.SIGINT, signal_handler)

    try:
        test = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        test.close()
        print("🎤 Mic check passed.")
    except Exception as e:
        print(f"❌ Mic error: {e}")
        return

    audio_queue = Queue(maxsize=30)

    print("\n🗣️ AssemblyAI Real-time Transcription")
    print("🕒 Speak naturally, Ctrl+C to stop\n")

    audio_proc = Process(target=audio_capture_loop, args=(audio_queue, stop_flag))
    ws_proc = Process(target=assembly_ws_client, args=(audio_queue, stop_flag))

    try:
        audio_proc.start()
        ws_proc.start()

        while not stop_flag.is_set():
            if not audio_proc.is_alive():
                print("\n⚠️ Audio process crashed")
                break
            if not ws_proc.is_alive():
                print("\n⚠️ WebSocket process crashed")
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_flag.set()
    finally:
        print("\n🔄 Cleaning up...")

        for name, proc in [("Audio", audio_proc), ("WebSocket", ws_proc)]:
            if proc.is_alive():
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
                    if proc.is_alive():
                        proc.kill()

        try:
            audio_interface.terminate()
        except:
            pass

        print("✅ Done!")


if __name__ == "__main__":
    main()
