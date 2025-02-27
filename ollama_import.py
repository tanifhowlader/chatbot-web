import os
from dotenv import load_dotenv
import requests
from flask import Flask, render_template, request, jsonify
from colorama import init, Fore, Style
import logging

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Flask and Colorama
init(autoreset=True)
app = Flask(__name__)

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Hugging Face Model URL
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"

# ✅ Chat with DeepSeek-R1 Model
def chat_with_open_source_model(prompt):
    api_key = os.getenv('HF_API_KEY')
    if not api_key:
        print("❌ Error: HF_API_KEY is not set. Please configure it in your .env or Render environment variables.")
        return "Error: API key is missing."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    try:
        print(f"🚀 Sending request to: {HUGGING_FACE_API_URL}")
        print(f"📤 Payload: {payload}")

        response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload, timeout=60)

        print(f"📥 Response Status Code: {response.status_code}")
        print(f"✅ Raw API Response: {response.text}")

        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'error' in result:
            error_msg = result['error']
            print(f"⚠️ Model Error: {error_msg}")
            return f"⚠️ DeepSeek-R1 Model Error: {error_msg}"
        else:
            return "⚠️ Unexpected response from the model."

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error: {http_err}")
        return "Error: HTTP request failed."
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Request Exception: {req_err}")
        return "Error: Could not reach the model API."
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return "Error: An unexpected error occurred."

# ✅ Routes
@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'templates/index.html' exists

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print(f"💬 Received user input: {user_input}")  # Debugging log

    if not user_input:
        print("⚠️ No input provided.")
        return jsonify({'response': 'Please enter a valid message.'})

    ai_response = chat_with_open_source_model(user_input)
    print(f"🤖 AI Response: {ai_response}")  # Debugging response
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
