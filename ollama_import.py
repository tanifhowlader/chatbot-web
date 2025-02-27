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

# ✅ Groq API Details (Correct Endpoint)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ✅ Function to Chat with Groq API (With Debugging Logs)
def chat_with_open_source_model(prompt):
    if not GROQ_API_KEY:
        print("❌ Error: Groq API key is missing. Set it in your environment variables.")
        return "Error: Missing API key."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",  # Other options: "llama3-70b", "gemini-pro"
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 512
    }

    try:
        print(f"🚀 Sending request to: {GROQ_API_URL}")
        print(f"📤 Payload: {payload}")

        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)

        print(f"📥 Response Status Code: {response.status_code}")
        print(f"✅ Raw API Response: {response.text}")

        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            print(f"⚠️ API Error: {result['error']['message']}")
            return f"⚠️ API Error: {result['error']['message']}"
        else:
            return "⚠️ Unexpected response format from Groq API."

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error on Render: {http_err}")
        return f"Error: HTTP request failed. {http_err}"
    except requests.exceptions.ConnectionError as conn_err:
        print(f"❌ Connection Error on Render: {conn_err}")
        return f"Error: Connection failed. {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        print(f"❌ Timeout Error on Render: {timeout_err}")
        return f"Error: API request timed out. {timeout_err}"
    except requests.exceptions.RequestException as req_err:
        print(f"❌ API Request Failed on Render: {req_err}")
        return f"Error: Could not reach the Groq API. {req_err}"

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
