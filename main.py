import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from llm import ask_llm
import threading
import time
import pygame
import uuid
import psycopg2
from dotenv import load_dotenv
import os
from elevenlabs.client import ElevenLabs
from datetime import datetime
from runbook_engine import search_runbook
import runbook_engine
print("Runbook file path:",runbook_engine.__file__)
load_dotenv()

# ===============================
# DATABASE CONNECTION                                                                                                                                                                   
# ===============================

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()
    print("Connected to internship DB successfully")

except Exception as e:
    print("Database connection failed:", e)
    exit()

# ===============================
# ELEVENLABS CLIENT
# ===============================

eleven_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

# ===============================
# STATIC IDS
# ===============================

agent_id = "aee54366-bf1b-4b00-af97-8c7c80d91fb0"
customer_id = "3fec2907-6bd3-45d6-b35b-7408c043d234"

# ===============================
# TRANSCRIPT FILE
# ===============================

TRANSCRIPT_FILE = "conversation_transcript.txt"

def save_to_transcript(speaker, text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {speaker}: {text}\n")

# ===============================
# LOAD WHISPER MODEL
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
# ELEVENLABS AUDIO GENERATION
# ===============================

def generate_audio(text, filename):
    audio_stream = eleven_client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        model_id="eleven_flash_v2",
        text=text
    )

    with open(filename, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)

# ===============================
# SPEAK FUNCTION
# ===============================

def speak(text):
    global speaking

    speaking = True
    filename = f"temp_{uuid.uuid4()}.mp3"

    generate_audio(text, filename)

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

            if not text or len(text) < 4:
                continue

            if text in current_response.lower():
                continue

            interrupt_text = text
            pygame.mixer.music.stop()
            speaking = False

        time.sleep(0.2)

threading.Thread(target=interrupt_listener, daemon=True).start()

# ===============================
# MAIN LOOP
# ===============================

print("\n📞 Real-Time Customer Voice Agent Started\n")

conversation_id = str(uuid.uuid4())
message_seq = 1

try:
    cursor.execute("""
    INSERT INTO conversations (id, customer_id, agent_id)
    VALUES (%s, %s, %s)
    """, (conversation_id, customer_id, agent_id))
    conn.commit()
    print("Conversation inserted successfully")

except Exception as e:
    conn.rollback()
    print("Conversation insert failed:", e)
    exit()

while True:

    audio = record_audio(duration=4)
    user_text = transcribe(audio)

    if not user_text:
        continue

    print("🧑:", user_text)

    try:
        message_id = str(uuid.uuid4())
        cursor.execute("""
        INSERT INTO conversation_messages
        (id, conversation_id, role, seq, content)
        VALUES (%s, %s, %s, %s, %s)
        """, (message_id, conversation_id, "user", message_seq, user_text))
        conn.commit()
        message_seq += 1

    except Exception as e:
        conn.rollback()
        print("User message insert failed:", e)

    save_to_transcript("User", user_text)
    print("Calling search_runbook.....")
    result = search_runbook(user_text)

    if result:
        response = result["solution"]
    else:
        response = ask_llm(user_text)
    current_response = response

    print("🤖:", response)

    try:
        message_id = str(uuid.uuid4())
        cursor.execute("""
        INSERT INTO conversation_messages
        (id, conversation_id, role, seq, content)
        VALUES (%s, %s, %s, %s, %s)
        """, (message_id, conversation_id, "agent", message_seq, response))
        conn.commit()
        message_seq += 1

    except Exception as e:
        conn.rollback()
        print("Agent message insert failed:", e)

    save_to_transcript("Agent", response)

    interrupt_text = None

    speech_thread = threading.Thread(target=speak, args=(response,))
    speech_thread.start()

    while speech_thread.is_alive():

        if interrupt_text:
            new_query = interrupt_text
            interrupt_text = None

            print("🧑 (interrupt):", new_query)

            try:
                message_id = str(uuid.uuid4())
                cursor.execute("""
                INSERT INTO conversation_messages
                (id, conversation_id, role, seq, content)
                VALUES (%s, %s, %s, %s, %s)
                """, (message_id, conversation_id, "user", message_seq, new_query))
                conn.commit()
                message_seq += 1

            except Exception as e:
                conn.rollback()
                print("Interrupt user message insert failed:", e)

            save_to_transcript("User (interrupt)", new_query)

            pygame.mixer.music.stop()

            result = search_runbook(new_query)

            if result:
                new_response = result["solution"]
            else:
                new_response = ask_llm(new_query)
            current_response = new_response

            print("🤖:", new_response)

            try:
                message_id = str(uuid.uuid4())
                cursor.execute("""
                INSERT INTO conversation_messages
                (id, conversation_id, role, seq, content)
                VALUES (%s, %s, %s, %s, %s)
                """, (message_id, conversation_id, "agent", message_seq, new_response))
                conn.commit()
                message_seq += 1

            except Exception as e:
                conn.rollback()
                print("Interrupt agent message insert failed:", e)

            save_to_transcript("Agent", new_response)

            speech_thread = threading.Thread(target=speak, args=(new_response,))
            speech_thread.start()

        time.sleep(0.2)