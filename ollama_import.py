import os
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import wikipediaapi
from duckduckgo_search import DDGS  # pip install duckduckgo-search
from groq import Groq

# ✅ Load .env variables
load_dotenv()

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Wikipedia API setup (HTML format avoids raw wiki markup leaking)
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnvironmentalScienceChatbot/1.0 (https://github.com/your-repo)",
    extract_format=wikipediaapi.ExtractFormat.HTML
)

# ✅ Definition-style trigger words — only use Wikipedia for these
DEFINITION_TRIGGERS = ["what is", "define", "explain", "meaning of", "definition"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_query(query: str) -> str:
    """Trim whitespace and trailing punctuation. Preserve case for proper nouns."""
    return query.strip().rstrip("?.!")


def is_definition_query(query: str) -> bool:
    lowered = query.lower()
    return any(trigger in lowered for trigger in DEFINITION_TRIGGERS)


def wikipedia_search(topic: str) -> str | None:
    logging.info(f"🔍 Wikipedia search: {topic}")
    page = wiki.page(topic)
    if page.exists():
        # Strip HTML tags from summary for clean plain-text output
        import re
        clean = re.sub(r"<[^>]+>", "", page.summary[:600])
        return f"📖 **Wikipedia:** {clean}..."
    return None


def duckduckgo_search(query: str) -> str | None:
    logging.info(f"🌐 DuckDuckGo search: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return f"🌍 **Web:** {results[0]['body']}"
    except Exception as e:
        logging.error(f"❌ DuckDuckGo error: {e}")
    return None


def build_messages(history: list, prompt: str) -> list:
    """Build Groq message list with system prompt + history + current user message."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert environmental science assistant. "
                "Answer clearly and concisely. If asked about topics outside "
                "environmental science, politely redirect."
            )
        },
        *history,  # conversation history passed from frontend
        {"role": "user", "content": prompt}
    ]


def groq_stream(messages: list):
    """Generator: streams Groq response chunks."""
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        stream=True,
        temperature=0.6,
        max_completion_tokens=2048
    )
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# ─────────────────────────────────────────────
# CORE PIPELINE (single source of truth)
# ─────────────────────────────────────────────

def generate_stream(user_input: str, history: list, last_query: str):
    """
    Three-tier fallback pipeline:
      1. Wikipedia  (only for definition-style queries)
      2. DuckDuckGo (real web search)
      3. Groq LLM   (with full conversation history)
    """
    # Handle bare "definition" follow-up using client-supplied last_query
    clean_prompt = clean_query(user_input)
    if clean_prompt.lower() == "definition" and last_query:
        clean_prompt = f"define {last_query}"

    # ── Tier 1: Wikipedia ───────────────────────
    if is_definition_query(clean_prompt):
        # Extract the topic from the query (strip trigger words)
        topic = clean_prompt.lower()
        for t in DEFINITION_TRIGGERS:
            topic = topic.replace(t, "").strip()
        wiki_result = wikipedia_search(topic)
        if wiki_result:
            yield wiki_result
            return

    # ── Tier 2: DuckDuckGo ──────────────────────
    ddg_result = duckduckgo_search(clean_prompt)
    if ddg_result:
        yield ddg_result
        return

    # ── Tier 3: Groq LLM ────────────────────────
    try:
        messages = build_messages(history, clean_prompt)
        yield from groq_stream(messages)
    except Exception as e:
        logging.error(f"❌ Groq streaming error: {e}")
        yield "⚠️ AI service is currently unavailable. Please try again later."

# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for gunicorn / deployment platforms."""
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    user_input = data.get("message", "").strip()
    history = data.get("history", [])       # [{"role": "user/assistant", "content": "..."}]
    last_query = data.get("last_query", "") # last non-trivial query from frontend

    # Input validation
    if not user_input:
        return jsonify({"response": "⚠️ Please enter a message."}), 400
    if len(user_input) > 500:
        return jsonify({"response": "⚠️ Message too long. Please keep it under 500 characters."}), 400

    logging.info(f"💬 User: {user_input}")

    return Response(
        stream_with_context(generate_stream(user_input, history, last_query)),
        mimetype="text/plain"
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
