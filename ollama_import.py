import openai
import requests
import logging
from googlesearch import search
from newspaper import Article
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from colorama import init, Fore, Style
import os

# Set up OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')  # Secure your API key

# Function to communicate with OpenAI API
def chat_with_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if available and desired
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Initialize colorama and Flask
init(autoreset=True)
app = Flask(__name__)

# Set up logging for monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Conversation history to maintain context
assistant_convo = []

def search_or_not(prompt):
    """Determine if a web search is required."""
    response = chat_with_openai(
        f"Determine if a web search is required for the following user input:\nUser: {prompt}"
    )
    return 'true' in response.lower()

def generate_query(prompt):
    """Generate a search query based on user input."""
    response = chat_with_openai(
        f"Create a concise and relevant search query for the following user input:\nUser: {prompt}"
    )
    return response

def google_search(query, num_results=5):
    """Perform a Google search and return a list of URLs."""
    return list(search(query, num_results=num_results, stop=num_results, pause=2))

def scrape_webpage(url):
    """Scrape and extract text content from a given webpage URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.warning(f"Primary scrape failed: {e}. Using fallback.")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join([p.text for p in soup.find_all('p')])
        except Exception as fallback_error:
            logging.error(f"Fallback scrape failed: {fallback_error}")
            return None

def contains_relevant_data(page_text, prompt):
    """Check if the scraped content answers the user's question."""
    response = chat_with_openai(
        f"Does the following content answer the user's question?\nUser Question: {prompt}\nContent: {page_text}"
    )
    return 'true' in response.lower()

def ai_search(prompt):
    """Conduct a web search and return the most relevant content."""
    search_query = generate_query(prompt)
    logging.info(f"Search Query: {search_query}")
    search_results = google_search(search_query)

    for url in search_results:
        page_text = scrape_webpage(url)
        if page_text and contains_relevant_data(page_text, prompt):
            return page_text

    logging.warning("No relevant data found.")
    return "No relevant data found."

def generate_response(user_input):
    """Generate an AI response with or without additional web search context."""
    assistant_convo.append({'role': 'user', 'content': user_input})

    if search_or_not(user_input):
        context = ai_search(user_input)
        assistant_convo.pop()  # Remove the original user input to avoid duplication
        combined_prompt = f"SEARCH RESULT: {context}\nUSER: {user_input}"
        assistant_convo.append({'role': 'user', 'content': combined_prompt})

    prompt_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in assistant_convo])
    ai_reply = chat_with_openai(prompt_history)
    assistant_convo.append({'role': 'assistant', 'content': ai_reply})

    return ai_reply

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'Please enter a valid message.'})

    ai_response = generate_response(user_input)
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
