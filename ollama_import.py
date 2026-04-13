import os
import re
import logging
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import wikipediaapi
from ddgs import DDGS  # ✅ renamed from duckduckgo_search
from groq import Groq

load_dotenv()
logging.basicConfig(level=logging.INFO)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = Flask(__name__)

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnvironmentalScienceChatbot/1.0 (https://github.com/your-repo)",
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

DEFINITION_TRIGGERS = ["what is", "define", "explain", "meaning of", "definition of"]

SYSTEM_PROMPTS = {
    "general": (
        "You are an expert environmental science assistant. "
        "Answer clearly and concisely. If asked about topics outside "
        "environmental science, politely redirect. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    ),
    "research": (
        "You are a marine biologist specializing in eDNA surveillance, "
        "oyster diseases (MSX, Dermo), aquaculture monitoring, and "
        "bioenvironmental assessment. Respond with scientific precision, "
        "reference detection methodologies (qPCR, metabarcoding, eDNA filtration), "
        "and suggest relevant monitoring techniques where applicable. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    ),
    "policy": (
        "You are an environmental policy advisor specializing in marine protected areas, "
        "aquaculture regulation, and environmental monitoring frameworks. "
        "Frame all answers around regulatory implications, management strategies, "
        "and policy recommendations. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    )
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_query(query: str) -> str:
    return query.strip().rstrip("?.!")


def is_definition_query(query: str) -> bool:
    lowered = query.lower()
    return any(trigger in lowered for trigger in DEFINITION_TRIGGERS)


def extract_topic(query: str) -> str:
    topic = query
    for trigger in DEFINITION_TRIGGERS:
        topic = re.sub(re.escape(trigger), "", topic, flags=re.IGNORECASE).strip()
    return topic


def wikipedia_search(topic: str) -> str | None:
    logging.info(f"🔍 Wikipedia search: {topic}")
    page = wiki.page(topic)
    if page.exists():
        summary = re.sub(r"\[\[.*?\]\]", "", page.summary[:600]).strip()
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


def build_messages(history: list, prompt: str, mode: str = "general") -> list:
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    return [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": prompt}
    ]


def groq_stream(messages: list):
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
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

def generate_stream(user_input: str, history: list, last_query: str, mode: str, force_ai: bool = False):
    """
    Three-tier fallback pipeline.
    force_ai=True skips Wikipedia + DuckDuckGo entirely → always uses Groq.
    This allows Research/Policy mode prompts to fully activate.
    """
    clean_prompt = clean_query(user_input)

    if clean_prompt.lower() == "definition" and last_query:
        clean_prompt = f"define {last_query}"

    if not force_ai:
        # ── Tier 1: Wikipedia ───────────────────────────────────────────────
        if is_definition_query(clean_prompt):
            topic = extract_topic(clean_prompt)
            wiki_result = wikipedia_search(topic)
            if wiki_result:
                yield wiki_result
                return

        # ── Tier 2: DuckDuckGo ─────────────────────────────────────────────
        ddg_result = duckduckgo_search(clean_prompt)
        if ddg_result:
            yield ddg_result
            return

    # ── Tier 3: Groq LLM ───────────────────────────────────────────────────
    # Always reached when force_ai=True, regardless of mode
    try:
        logging.info(f"🤖 Groq responding | mode={mode} | force_ai={force_ai}")
        messages = build_messages(history, clean_prompt, mode)
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


@app.route("/debug")
def debug():
    return jsonify({
        "model":        os.getenv("GROQ_MODEL", "NOT SET"),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "port":         os.environ.get("PORT", "NOT SET")
    })


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.json
    user_input = data.get("message", "").strip()
    history    = data.get("history", [])
    last_query = data.get("last_query", "")
    mode       = data.get("mode", "general")
    force_ai   = data.get("force_ai", False)  # ✅ new

    if not isinstance(history, list):
        history = []
    history = [m for m in history if isinstance(m, dict) and "role" in m and "content" in m]

    if not user_input:
        return jsonify({"response": "⚠️ Please enter a message."}), 400
    if len(user_input) > 500:
        return jsonify({"response": "⚠️ Message too long. Keep it under 500 characters."}), 400

    logging.info(f"💬 User: {user_input} | Mode: {mode} | Force AI: {force_ai}")

    return Response(
        stream_with_context(generate_stream(user_input, history, last_query, mode, force_ai)),
        mimetype="text/plain"
    )


@app.route("/env-data")
def env_data():
    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude":  44.3601,
                "longitude": -78.3197,
                "current":   "pm2_5,carbon_monoxide,nitrogen_dioxide,european_aqi"
            },
            timeout=8
        )
        r.raise_for_status()
        data = r.json().get("current", {})

        aqi = data.get("european_aqi", 0)
        if   aqi <= 20: label, color = "Good",      "#2e7d32"
        elif aqi <= 40: label, color = "Fair",      "#f9a825"
        elif aqi <= 60: label, color = "Moderate",  "#e65100"
        elif aqi <= 80: label, color = "Poor",      "#b71c1c"
        else:           label, color = "Very Poor", "#6a1b9a"

        return jsonify({
            "aqi":   aqi,
            "label": label,
            "color": color,
            "pm2_5": round(data.get("pm2_5", 0), 1),
            "co":    round(data.get("carbon_monoxide", 0), 1),
            "no2":   round(data.get("nitrogen_dioxide", 0), 1),
        })
    except Exception as e:
        logging.error(f"❌ Env data error: {e}")
        return jsonify({"error": "Unable to fetch environmental data."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)



