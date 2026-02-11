import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from llm import ask_llm
import threading
import time
import asyncio
import edge_tts
import pygame
import uuid
import os

# ===============================
# LOAD MODEL (LIGHTWEIGHT)
# ===============================

print("🔄 Loading Tiny Whisper model...")
whisper_model = WhisperModel("tiny", compute_type="int8")

# ===============================
# GLOBAL STATE
# ===============================

interrupt_text = None
speaking = False
current_response = ""

pygame.mixer.init()

# ===============================
# AUDIO RECORD
# ===============================

def record_audio(duration=1.5, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs,
                   channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def transcribe(audio):
    segments, _ = whisper_model.transcribe(audio)
    text = ""
    for seg in segments:
        text += seg.text
    return text.strip().lower()

# ===============================
# EDGE TTS SPEAK
# ===============================

async def generate_audio(text, filename):
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-AriaNeural"
    )
    await communicate.save(filename)

def speak(text):
    global speaking

    speaking = True
    filename = f"temp_{uuid.uuid4()}.mp3"

    asyncio.run(generate_audio(text, filename))

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.unload()

    try:
        os.remove(filename)
    except:
        pass

    speaking = False

# ===============================
# INTERRUPT LISTENER
# ===============================

def interrupt_listener():
    global interrupt_text, speaking, current_response

    while True:
        if speaking:
            audio = record_audio(duration=1.0)
            text = transcribe(audio)

            # Ignore noise
            if not text or len(text) < 4:
                continue

            # Ignore echo of agent speech
            if text in current_response.lower():
                continue

            interrupt_text = text
            pygame.mixer.music.stop()
            speaking = False

        time.sleep(0.2)

# Start interrupt thread
threading.Thread(target=interrupt_listener, daemon=True).start()

# ===============================
# MAIN LOOP
# ===============================

print("\n📞 Real-Time Customer Voice Agent Started\n")

while True:

    audio = record_audio(duration=4)
    user_text = transcribe(audio)

    if not user_text:
        continue

    print("🧑:", user_text)

    response = ask_llm(user_text)
    current_response = response

    print("🤖:", response)

    interrupt_text = None

    speech_thread = threading.Thread(target=speak, args=(response,))
    speech_thread.start()

    while speech_thread.is_alive():

        if interrupt_text:
            new_query = interrupt_text
            interrupt_text = None

            print("🧑 (interrupt):", new_query)

            pygame.mixer.music.stop()

            new_response = ask_llm(new_query)
            current_response = new_response

            print("🤖:", new_response)

            speech_thread = threading.Thread(target=speak, args=(new_response,))
            speech_thread.start()

        time.sleep(0.2)
