import os
import re
import logging
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import wikipediaapi
from ddgs import DDGS
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
        "Answer clearly and concisely. If context from a web search is provided, "
        "use it to enrich your answer but always give a complete, detailed response. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    ),
    "research": (
        "You are a marine biologist specializing in eDNA surveillance, "
        "oyster diseases (MSX, Dermo), aquaculture monitoring, and bioenvironmental assessment. "
        "Respond with scientific precision, reference detection methodologies "
        "(qPCR, metabarcoding, eDNA filtration), and suggest relevant monitoring techniques. "
        "If web search context is provided, incorporate it and expand with technical depth. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    ),
    "policy": (
        "You are an environmental policy advisor specializing in marine protected areas, "
        "aquaculture regulation, and environmental monitoring frameworks. "
        "Frame all answers around regulatory implications, management strategies, "
        "and policy recommendations. Incorporate any web search context provided. "
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
        summary = re.sub(r"\[\[.*?\]\]", "", page.summary[:800]).strip()
        return summary
    return None


def duckduckgo_search(query: str) -> str | None:
    """
    Returns combined snippets from top 3 results as context for Groq.
    Never returned directly to user anymore.
    """
    logging.info(f"🌐 DuckDuckGo context fetch: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                combined = " | ".join(
                    r["body"] for r in results if r.get("body")
                )
                return combined[:1200]  # cap context at 1200 chars
    except Exception as e:
        logging.error(f"❌ DuckDuckGo error: {e}")
    return None


def build_messages(history: list, prompt: str, mode: str = "general", context: str = None) -> list:
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])

    # ✅ Inject web/wiki context into user message so Groq uses it
    if context:
        enriched_prompt = (
            f"Context from web search (use this to inform your answer):\n{context}\n\n"
            f"User question: {prompt}"
        )
    else:
        enriched_prompt = prompt

    return [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": enriched_prompt}
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
# CORE PIPELINE — Revised Architecture
# ─────────────────────────────────────────────

def generate_stream(user_input: str, history: list, last_query: str, mode: str, force_ai: bool = False):
    """
    NEW PIPELINE:
      1. Wikipedia   → returns directly for definition queries (concise factual)
      2. DuckDuckGo  → fetched as CONTEXT, passed to Groq (not returned raw)
      3. Groq        → ALWAYS runs (enriched with DDG context if available)

    force_ai=True skips Wikipedia + DDG entirely → pure Groq response.
    """
    clean_prompt = clean_query(user_input)

    if clean_prompt.lower() == "definition" and last_query:
        clean_prompt = f"define {last_query}"

    context = None

    if not force_ai:
        # ── Tier 1: Wikipedia for definitions only ──────────────────────────
        if is_definition_query(clean_prompt):
            topic = extract_topic(clean_prompt)
            wiki_result = wikipedia_search(topic)
            if wiki_result:
                logging.info("📖 Wikipedia result found — returning directly")
                yield f"📖 **Wikipedia:** {wiki_result}..."
                return

        # ── Tier 2: DuckDuckGo as context (NOT returned raw) ───────────────
        context = duckduckgo_search(clean_prompt)
        if context:
            logging.info("🌐 DDG context fetched — passing to Groq")
        else:
            logging.info("⚠️ No DDG context — Groq answering from training only")

    # ── Tier 3: Groq ALWAYS runs — enriched with context ───────────────────
    try:
        logging.info(f"🤖 Groq responding | mode={mode} | has_context={bool(context)} | force_ai={force_ai}")
        messages = build_messages(history, clean_prompt, mode, context)
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
    force_ai   = data.get("force_ai", False)

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
            "aqi":   aqi, "label": label, "color": color,
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




