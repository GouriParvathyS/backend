from flask import Flask, request, jsonify
import speech_recognition as sr
import google.generativeai as genai
import os
from flask_cors import CORS
import traceback  # Helps print detailed error logs

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure Google Gemini API
GENAI_API_KEY = "APIKEY"  # Replace with your valid API Key
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        if "file" not in request.files:
            print("❌ Error: No file uploaded")  
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = "temp_audio.wav"
        print(f"📁 Saving file to {file_path}")  
        file.save(file_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

        os.remove(file_path)  # Cleanup

        # Convert speech to text
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"✅ Transcription: {text}")  

        return jsonify({"transcription": text})

    except sr.UnknownValueError:
        print("⚠️ Speech recognition could not understand audio")  
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {str(e)}")  
        return jsonify({"error": "Speech recognition service unavailable"}), 500
    except Exception as e:
        print("🔥 Full Error Traceback:\n", traceback.format_exc())  
        return jsonify({"error": f"Audio processing error: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        print(f"💬 Received message: {user_message}")  

        if not user_message:
            return jsonify({"response": "I couldn't understand that."}), 400

        chat_session = model.start_chat(history=[])

        # Catch API quota errors
        try:
            ai_response = chat_session.send_message(user_message)
            print(f"🤖 AI Response: {ai_response.text}")  
            return jsonify({"response": ai_response.text})
        except Exception as api_error:
            print(f"❌ API Error: {str(api_error)}")  
            return jsonify({"error": "Google Gemini API quota exceeded. Try again later."}), 429

    except Exception as e:
        print("🔥 Chat Processing Error:\n", traceback.format_exc())  
        return jsonify({"error": f"Chat processing error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
