import os
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv # Used to load .env file
import io
import soundfile as sf
from pydub import AudioSegment

# Google Cloud STT/TTS clients
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1beta1 as texttospeech

# Google Gemini LLM client
import google.generativeai as genai

app = Flask(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- API Key/Credential Configuration ---

# 1. Google Cloud Service Account for STT and TTS
# This line tells Google Cloud client libraries where to find your JSON key file.
# You MUST set this environment variable BEFORE running the Flask app,
# either in your terminal or in the .env file if using dotenv for it.
# e.g., in your terminal: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-chefbot-key.json"
# Or in .env: GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-chefbot-key.json"
# The google.cloud clients will automatically pick this up.
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# 2. Google Gemini LLM API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro' if 1.5-flash isn't available or preferred.

# --- ChefBot's Persona and Knowledge Base (same as before) ---
SYSTEM_PROMPT = """
You are "Ross," a friendly and efficient AI assistant for "The Hungry Dragon" restaurant.
Your primary goal is to help customers with orders, table bookings, and provide information about the restaurant and menu.

**Here's the menu information:**
* **Pizzas:** Pepperoni ($15), Margherita ($14), Veggie ($16)
* **Pastas:** Spaghetti Carbonara ($12), Fettuccine Alfredo ($13)
* **Salads:** Garden Salad ($10), Caesar Salad ($11)
* **Special of the Day:** Dragon's Breath Chili ($18)
* **Drinks:** Water ($2), Soda ($3), Juice ($4)

**Restaurant Information:**
* **Opening Hours:** 11:00 AM to 10:00 PM daily.
* **Address:** 123 Main Street, Flavorville.
* **Phone:** (555) 123-4567.
* **Cuisine Type:** American with a fiery twist.
* **Atmosphere:** Cozy, rustic, and family-friendly.

**Booking Rules:**
* You can book tables for groups of 1 to 10 people.
* Always confirm date, time, and number of guests.
* Be aware of busy times (7 PM - 9 PM are often fully booked). If busy, suggest alternative times.

**Your Behavior:**
* Be polite, helpful, and clear.
* If you need more information (e.g., specific order items, booking details), politely ask clarifying questions.
* When confirming an order or booking, summarize the details clearly.
* If a customer tries to order something not on the menu, politely say "I'm sorry, that item is not on our menu."
* Do not try to process payments. If a customer asks, say "You can pay at the counter when you pick up your order or after your meal."
* Simulate actions: If an order or booking is confirmed, simply state that it has been "placed" or "confirmed" and print a message to the console on the backend server.

**Examples of interactions:**
* Customer: "I want to order a pizza." -> ChefBot: "Which pizza would you like? Pepperoni, Margherita, or Veggie?"
* Customer: "Book a table for 4 tomorrow at 7 PM." -> ChefBot: "Okay, booking a table for 4 people for tomorrow at 7 PM. Is that correct?"
* Customer: "What time do you close?" -> ChefBot: "We close at 10:00 PM daily."
"""

# Initialize Gemini chat session outside the route to maintain conversation history
chat = llm_model.start_chat(history=[
    {"role": "user", "parts": SYSTEM_PROMPT},
    {"role": "model", "parts": "Hello! I'm ChefBot, your AI assistant for The Hungry Dragon restaurant. How can I help you today? Are you looking to order, book a table, or something else?"}
])


@app.route('/')
def index():
    return open('index.html').read()

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    try:
        # Convert audio to a format suitable for Google Speech-to-Text (e.g., mono, 16kHz, LINEAR16)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_file.mimetype.split('/')[-1])
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2) # Mono, 16kHz, 16-bit
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0) # Reset buffer position for reading

        # 1. Speech-to-Text (STT)
        audio = speech.RecognitionAudio(content=wav_buffer.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        response = speech_client.recognize(config=config, audio=audio)
        customer_text = ""
        for result in response.results:
            customer_text += result.alternatives[0].transcript

        if not customer_text:
            return jsonify({'text': "I didn't catch that. Could you please repeat?", 'audio': ''})

        print(f"Customer said: \"{customer_text}\"")

        # 2. Large Language Model (LLM) for understanding and generation
        llm_response = chat.send_message(customer_text)
        chefbot_text = llm_response.text

        print(f"ChefBot replied: \"{chefbot_text}\"")

        # 3. Simulate actions based on ChefBot's response (simple console logs for now)
        if "order placed" in chefbot_text.lower():
            print("--- Action: Food order placed! ---")
        if "table confirmed" in chefbot_text.lower() or "booking confirmed" in chefbot_text.lower():
            print("--- Action: Table booking confirmed! ---")


        # 4. Text-to-Speech (TTS)
        synthesis_input = texttospeech.SynthesisInput(text=chefbot_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        tts_response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Send the audio back
        return send_file(
            io.BytesIO(tts_response.audio_content),
            mimetype="audio/mpeg",
            as_attachment=False
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make sure to set the GOOGLE_APPLICATION_CREDENTIALS environment variable
    # to the path of your downloaded JSON service account key file
    # before running this script.
    # Example (in your terminal BEFORE running python app.py):
    # export GOOGLE_APPLICATION_CREDENTIALS="/Users/youruser/path/to/google-cloud-chefbot-key.json"
    # (For Windows, use: set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\google-cloud-chefbot-key.json")
    app.run(debug=True, port=5000)