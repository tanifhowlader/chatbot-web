import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import wikipediaapi
import requests
from groq import Groq

# ✅ Load .env variables
load_dotenv()

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Groq Client
client = Groq()

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Wikipedia API setup
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnvironmentalScienceChatbot/1.0 (https://github.com/your-repo)",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# ✅ DuckDuckGo endpoint
DUCKDUCKGO_SEARCH_URL = "https://api.duckduckgo.com/"
LAST_QUERY = ""

def clean_query(query):
    query = query.strip().lower().replace("?", "").replace(".", "")
    if query in ["what environmental science", "define environmental science"]:
        query = "environmental science"
    return query

def wikipedia_search(topic):
    logging.info(f"🔍 Searching Wikipedia: {topic}")
    page = wiki.page(topic)
    return f"📖 **Wikipedia:** {page.summary[:500]}..." if page.exists() else None

def duckduckgo_search(query):
    logging.info(f"🌐 DuckDuckGo search: {query}")
    params = {"q": query, "format": "json"}
    try:
        response = requests.get(DUCKDUCKGO_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "AbstractText" in data and data["AbstractText"]:
            return f"🌍 **DuckDuckGo:** {data['AbstractText']}"
        elif "RelatedTopics" in data and data["RelatedTopics"]:
            return f"🌍 **DuckDuckGo:** {data['RelatedTopics'][0]['Text']}"
        return "⚠️ No relevant results found."
    except Exception as e:
        logging.error(f"❌ DuckDuckGo error: {e}")
        return "⚠️ Unable to search DuckDuckGo."

def chat_with_model(prompt):
    global LAST_QUERY
    clean_prompt = clean_query(prompt)
    if not clean_prompt:
        return "⚠️ Please enter a valid question."
    if clean_prompt == "definition" and LAST_QUERY:
        clean_prompt = LAST_QUERY
    else:
        LAST_QUERY = clean_prompt

    # ✅ Try Wikipedia
    wiki_result = wikipedia_search(clean_prompt)
    if wiki_result:
        return wiki_result

    # ✅ Try DuckDuckGo
    ddg_result = duckduckgo_search(clean_prompt)
    if ddg_result and "⚠️" not in ddg_result:
        return ddg_result

    # ✅ Groq fallback
    try:
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "mistral-saba-24b"),
            messages=[
                {"role": "system", "content": "You are an expert environmental science assistant."},
                {"role": "user", "content": clean_prompt}
            ],
            temperature=0.6,
            max_completion_tokens=2048,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"❌ Groq API error: {e}")
        return "⚠️ AI service is currently unavailable. Please try again later."

# ✅ Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    logging.info(f"💬 User input: {user_input}")
    if not user_input:
        return jsonify({"response": "⚠️ Please enter a message."})
    response = chat_with_model(user_input)
    return jsonify({"response": response})

# ✅ Run server locally or on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
