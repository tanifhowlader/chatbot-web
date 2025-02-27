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

# ✅ DeepSeek API Details
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ✅ Function to Chat with DeepSeek API
def chat_with_open_source_model(prompt):
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DeepSeek API key is missing. Set it in your environment variables.")
        return "Error: Missing API key."

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 512
    }

    try:
        print(f"🚀 Sending request to: {DEEPSEEK_API_URL}")
        print(f"📤 Payload: {payload}")

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)

        print(f"📥 Response Status Code: {response.status_code}")
        print(f"✅ Raw API Response: {response.text}")

        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"⚠️ API Error: {result['error']['message']}"
        else:
            return "⚠️ Unexpected response format from DeepSeek API."

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error: {http_err}")
        return "Error: HTTP request failed."
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Request Exception: {req_err}")
        return "Error: Could not reach the DeepSeek API."
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
