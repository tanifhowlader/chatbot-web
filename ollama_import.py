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

# ✅ Block Wikipedia pages from unrelated fields
BLOCKED_WIKI_TOPICS = [
    "physics", "neuroscience", "mathematics", "spacetime", "minkowski",
    "relativity", "brain", "psychology", "astronomy", "quantum",
    "lorentz", "geometry", "latent diffusion", "mri"
]

# ✅ Guard: only answer if query touches these domains
ALLOWED_ENV_KEYWORDS = [
    "environment", "ecology", "climate", "marine", "ocean", "aquatic",
    "species", "habitat", "biodiversity", "water", "soil", "air",
    "pollution", "conservation", "monitoring", "ecosystem", "estuary",
    "oyster", "msx", "edna", "dna", "disease", "aquaculture", "fisheries",
    "wetland", "forest", "carbon", "emission", "wildlife", "river", "lake",
    "coastal", "algae", "sediment", "toxin", "pathogen", "microbe",
    "bacteria", "virus", "fungi", "parasite", "plankton", "benthic",
    "salinity", "temperature", "pH", "dissolved oxygen", "turbidity",
    "protected area", "mpa", "dfo", "cfia", "regulation", "policy",
    "surveillance", "detection", "sampling", "qpcr", "metabarcoding",
    "biomonitoring", "bioenvironmental", "spatiotemporal", "spread",
    "outbreak", "population", "abundance", "distribution", "mapping"
]

OFF_TOPIC_RESPONSE = (
    "⚠️ This chatbot is focused on **environmental science**, ecology, "
    "aquatic biology, and related fields.\n\n"
    "Please ask a question related to topics like:\n"
    "- 🦪 Oyster disease (MSX, Dermo)\n"
    "- 🧬 eDNA surveillance and monitoring\n"
    "- 🌊 Marine Protected Areas\n"
    "- 🌿 Ecosystems and biodiversity\n"
    "- 💧 Water quality and aquatic health\n"
    "- 📋 Environmental policy and regulation"
)

SYSTEM_PROMPTS = {
    "general": (
        "You are an expert environmental science assistant. "
        "ONLY answer questions related to environmental science, ecology, "
        "aquatic biology, climate, conservation, or closely related fields. "
        "If the question is unrelated (e.g. pure physics, neuroscience, mathematics, "
        "general medicine unrelated to environment), politely decline and redirect. "
        "If context from a web search is provided, use it to enrich your answer "
        "but always give a complete, detailed response. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    ),
    "research": (
        "You are a marine biologist specializing in eDNA surveillance, "
        "oyster diseases (MSX, Dermo), aquaculture monitoring, and bioenvironmental assessment. "
        "ONLY answer questions related to these fields or environmental science broadly. "
        "Respond with scientific precision, reference detection methodologies "
        "(qPCR, metabarcoding, eDNA filtration, histology), and suggest relevant "
        "monitoring techniques where applicable. "
        "If web search context is provided, incorporate it and expand with technical depth. "
        "After your response, on a new line write exactly: "
        "CITE: [APA-style suggested citation for the main topic discussed]"
    ),
    "policy": (
        "You are an environmental policy advisor specializing in marine protected areas, "
        "aquaculture regulation, DFO/CFIA frameworks, and environmental monitoring policy. "
        "ONLY answer questions related to environmental governance or science policy. "
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


def is_environmental_query(query: str) -> bool:
    """
    Returns True if the query contains at least one environmental keyword.
    Prevents physics, neuroscience, and other unrelated topics from passing through.
    """
    lowered = query.lower()
    return any(keyword in lowered for keyword in ALLOWED_ENV_KEYWORDS)


def wikipedia_search(topic: str) -> str | None:
    logging.info(f"🔍 Wikipedia search: {topic}")
    page = wiki.page(topic)
    if page.exists():
        summary_lower = page.summary[:300].lower()
        # ✅ Block off-topic Wikipedia pages
        if any(word in summary_lower for word in BLOCKED_WIKI_TOPICS):
            logging.info(f"⛔ Wikipedia blocked — off-topic content detected: {topic}")
            return None
        summary = re.sub(r"\[\[.*?\]\]", "", page.summary[:800]).strip()
        return summary
    return None


def duckduckgo_search(query: str) -> str | None:
    """
    Fetches top 3 DDG results and combines them as context for Groq.
    Never returned raw to the user.
    """
    logging.info(f"🌐 DDG context fetch: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                combined = " | ".join(
                    r["body"] for r in results if r.get("body")
                )
                return combined[:1200]
    except Exception as e:
        logging.error(f"❌ DuckDuckGo error: {e}")
    return None


def build_messages(history: list, prompt: str, mode: str = "general", context: str = None) -> list:
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])

    if context:
        enriched_prompt = (
            f"Context from web search (use this to inform and enrich your answer):\n"
            f"{context}\n\n"
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
# CORE PIPELINE
# ─────────────────────────────────────────────

def generate_stream(user_input: str, history: list, last_query: str, mode: str, force_ai: bool = False):
    """
    PIPELINE:
      0. Off-topic guard — reject non-environmental queries immediately
      1. Wikipedia       — returns directly for clear definition queries
      2. DuckDuckGo      — fetched silently as context for Groq
      3. Groq            — ALWAYS runs with enriched context + mode prompt
    """
    clean_prompt = clean_query(user_input)

    if clean_prompt.lower() == "definition" and last_query:
        clean_prompt = f"define {last_query}"

    # ── Step 0: Off-topic guard ─────────────────────────────────────────────
    if not force_ai and not is_environmental_query(clean_prompt):
        logging.info(f"⛔ Off-topic query rejected: {clean_prompt}")
        yield OFF_TOPIC_RESPONSE
        return

    context = None

    if not force_ai:
        # ── Step 1: Wikipedia (definitions only, returns directly) ──────────
        if is_definition_query(clean_prompt):
            topic = extract_topic(clean_prompt)
            wiki_result = wikipedia_search(topic)
            if wiki_result:
                logging.info("📖 Wikipedia result — returning directly")
                yield f"📖 **Wikipedia:** {wiki_result}..."
                return
            # If Wikipedia blocked or missing → fall through to DDG + Groq

        # ── Step 2: DDG as silent context for Groq ──────────────────────────
        context = duckduckgo_search(clean_prompt)
        if context:
            logging.info("🌐 DDG context fetched → passing to Groq")
        else:
            logging.info("⚠️ No DDG context → Groq answering from training only")

    # ── Step 3: Groq always responds ────────────────────────────────────────
    try:
        logging.info(f"🤖 Groq | mode={mode} | context={bool(context)} | force_ai={force_ai}")
        messages = build_messages(history, clean_prompt, mode, context)
        yield from groq_stream(messages)
    except Exception as e:
        logging.error(f"❌ Groq error: {e}")
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




