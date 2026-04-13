import os
import re
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import wikipediaapi
from duckduckgo_search import DDGS
from groq import Groq

# ✅ Load .env variables
load_dotenv()

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Wikipedia API — WIKI format (plain text, no HTML tags to strip)
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnvironmentalScienceChatbot/1.0 (https://github.com/your-repo)",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

# ✅ FIX 3: Wikipedia only fires for these trigger phrases
DEFINITION_TRIGGERS = ["what is", "define", "explain", "meaning of", "definition of"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_query(query: str) -> str:
    """
    FIX 2: NO .lower() — preserves case for proper nouns (MSX, eDNA, etc.)
    Only strips surrounding whitespace and trailing punctuation.
    """
    return query.strip().rstrip("?.!")


def is_definition_query(query: str) -> bool:
    """FIX 3: Only returns True when query explicitly asks for a definition."""
    lowered = query.lower()
    return any(lowered.startswith(trigger) for trigger in DEFINITION_TRIGGERS)


def extract_topic(query: str) -> str:
    """
    Strip definition trigger words while PRESERVING original casing of the topic.
    e.g. 'what is MSX disease' → 'MSX disease'  (not 'msx disease')
    """
    topic = query
    for trigger in DEFINITION_TRIGGERS:
        topic = re.sub(re.escape(trigger), "", topic, flags=re.IGNORECASE).strip()
    return topic


def wikipedia_search(topic: str) -> str | None:
    logging.info(f"🔍 Wikipedia search: {topic}")  # ✅ will now log 'MSX' not 'msx'
    page = wiki.page(topic)
    if page.exists():
        summary = page.summary[:600]
        return f"📖 **Wikipedia:** {summary}..."
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
    return [
        {
            "role": "system",
            "content": (
                "You are an expert environmental science assistant. "
                "Answer clearly and concisely. If asked about topics outside "
                "environmental science, politely redirect."
            )
        },
        *history,
        {"role": "user", "content": prompt}
    ]


def groq_stream(messages: list):
    """FIX 1: Updated model — mistral-saba-24b is decommissioned."""
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),  # ✅ FIX 1
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
# CORE PIPELINE
# ─────────────────────────────────────────────

def generate_stream(user_input: str, history: list, last_query: str):
    """
    Three-tier fallback:
      1. Wikipedia  — ONLY for definition-style queries (FIX 3)
      2. DuckDuckGo — full web search for everything else
      3. Groq LLM   — with conversation history, updated model (FIX 1)
    """
    clean_prompt = clean_query(user_input)  # FIX 2: no lowercasing

    # Handle bare "definition" follow-up using client-supplied last_query
    if clean_prompt.lower() == "definition" and last_query:
        clean_prompt = f"define {last_query}"

    # ── Tier 1: Wikipedia ───────────────────────────────────────────────────
    # FIX 3: Guard — Wikipedia only fires if user explicitly asks for a definition
    if is_definition_query(clean_prompt):
        topic = extract_topic(clean_prompt)  # FIX 2: topic preserves original case
        wiki_result = wikipedia_search(topic)
        if wiki_result:
            yield wiki_result
            return
        # If Wikipedia misses, fall through to DuckDuckGo

    # ── Tier 2: DuckDuckGo ─────────────────────────────────────────────────
    # Now correctly reached for bare queries like "MSX", "biology", "oyster disease"
    ddg_result = duckduckgo_search(clean_prompt)
    if ddg_result:
        yield ddg_result
        return

    # ── Tier 3: Groq LLM ───────────────────────────────────────────────────
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
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    history    = data.get("history", [])
    last_query = data.get("last_query", "")

    if not user_input:
        return jsonify({"response": "⚠️ Please enter a message."}), 400
    if len(user_input) > 500:
        return jsonify({"response": "⚠️ Message too long. Keep it under 500 characters."}), 400

    logging.info(f"💬 User input: {user_input}")

    return Response(
        stream_with_context(generate_stream(user_input, history, last_query)),
        mimetype="text/plain"
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

