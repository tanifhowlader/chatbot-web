import os
import requests
import wikipediaapi
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Set API Keys & URLs
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()  # Strip any accidental newlines
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DUCKDUCKGO_SEARCH_URL = "https://api.duckduckgo.com/"
LAST_QUERY = ""  # Store last searched topic for better definition handling

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Wikipedia Configuration (Fixed)
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnvironmentalScienceChatbot/1.0 (https://github.com/your-repo)",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def clean_query(query):
    """Removes unnecessary characters and normalizes user input."""
    query = query.strip().lower().replace("?", "").replace(".", "")
    
    # Fix common queries
    if query in ["what environmental science", "define environmental science"]:
        query = "environmental science"
    
    return query

def wikipedia_search(topic):
    """Fetches summary from Wikipedia if available."""
    logging.info(f"📖 Searching Wikipedia for: {topic}")
    page = wiki.page(topic)
    if page.exists():
        return f"📖 **Wikipedia:** {page.summary[:500]}..."  # Limit response to 500 chars
    else:
        logging.warning(f"⚠️ Wikipedia page not found for: {topic}")
        return None

def duckduckgo_search(query):
    """Searches DuckDuckGo for relevant environmental data."""
    params = {"q": query, "format": "json"}
    logging.info(f"🚀 Searching DuckDuckGo for: {query}")
    try:
        response = requests.get(DUCKDUCKGO_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        if "AbstractText" in result and result["AbstractText"]:
            return f"🌍 **DuckDuckGo Search:** {result['AbstractText']}"
        elif "RelatedTopics" in result and result["RelatedTopics"]:
            return f"🌍 **DuckDuckGo Search:** {result['RelatedTopics'][0]['Text']}"
        else:
            return "⚠️ No relevant results found on DuckDuckGo."
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error fetching DuckDuckGo search results: {e}")
        return "⚠️ Unable to fetch search results."

def chat_with_open_source_model(prompt):
    """Handles intelligent responses by integrating Wikipedia, DuckDuckGo, and Groq AI."""
    global LAST_QUERY
    clean_prompt = clean_query(prompt)
    if not clean_prompt:
        return "⚠️ Please enter a valid question."
    if clean_prompt == "definition" and LAST_QUERY:
        clean_prompt = LAST_QUERY  # Use the last searched topic instead of 'definition' alone
    else:
        LAST_QUERY = clean_prompt  # Store the current topic for later reference

    # ✅ Wikipedia Search
    wiki_result = wikipedia_search(clean_prompt)
    if wiki_result:
        return wiki_result

    # ✅ DuckDuckGo Search (Fallback)
    search_result = duckduckgo_search(clean_prompt)
    if search_result and "⚠️" not in search_result:
        return search_result

    # ✅ AI Response if No Search Results
    if not GROQ_API_KEY:
        logging.error("❌ API key is missing. Please check your .env file.")
        return "⚠️ Error: API key is missing. Please check your setup."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an expert environmental science assistant."},
            {"role": "user", "content": clean_prompt}
        ],
        "temperature": 0.6,
        "max_tokens": 512
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"⚠️ API Error: {result['error']['message']}"
        else:
            return "⚠️ Unexpected response format from Groq API."
    except requests.exceptions.RequestException as e:
        return f"❌ API Request Failed: {e}"

# ✅ Flask Routes
@app.route("/")
def home():
    """Renders the chat UI."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Processes user input and returns an intelligent response."""
    user_input = request.json.get("message")
    logging.info(f"💬 Received user input: {user_input}")
    if not user_input:
        return jsonify({"response": "⚠️ Please enter a valid message."})
    ai_response = chat_with_open_source_model(user_input)
    logging.info(f"🤖 AI Response: {ai_response}")
    return jsonify({"response": ai_response})

# ✅ Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"🚀 Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
