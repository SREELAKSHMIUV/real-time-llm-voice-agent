import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(user_text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        messages=[
            {"role": "system", "content": "You are a friendly, calm, and natural customer support assistant. Speak in a conversational and human-like tone. Keep responses concise but helpful. Avoid robotic explanations."},
            {"role": "user", "content": user_text}
        ]
    )

    return response.choices[0].message.content
