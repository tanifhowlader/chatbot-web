import os
import logging
import requests
from googlesearch import search
from newspaper import Article
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from colorama import init, Fore, Style
import openai

# Set OpenAI API key from environment variable
def set_openai_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key

set_openai_api_key()

# Function to communicate with OpenAI API using the latest interface
def chat_with_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Change to "gpt-4" if accessible
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error communicating with OpenAI: {e}")
        return "An error occurred while processing your request. Please try again later."

# Initialize colorama and Flask
init(autoreset=True)
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Store conversation history for context-aware responses
assistant_convo = []

def search_or_not(prompt):
    """Determine if a web search is required based on the user's input."""
    response = chat_with_openai(
        f"Determine if a web search is required for the following user input:\nUser: {prompt}"
    )
    return 'true' in response.lower()

def generate_query(prompt):
    """Generate a search query based on the user input."""
    response = chat_with_openai(
        f"Create a concise search query for the following input:\nUser: {prompt}"
    )
    return response

def google_search(query, num_results=5):
    """Perform a Google search and return a list of URLs."""
    try:
        return list(search(query, num_results=num_results, stop=num_results, pause=2))
    except Exception as e:
        logging.error(f"Google search failed: {e}")
        return []

def scrape_webpage(url):
    """Extract text content from a webpage."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.warning(f"Primary scrape failed: {e}. Using fallback method.")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join([p.text for p in soup.find_all('p')])
        except Exception as fallback_error:
            logging.error(f"Fallback scrape failed: {fallback_error}")
            return None

def contains_relevant_data(page_text, prompt):
    """Check if the extracted content answers the user's query."""
    response = chat_with_openai(
        f"Does the following content answer the user's question?\nUser Question: {prompt}\nContent: {page_text}"
    )
    return 'true' in response.lower()

def ai_search(prompt):
    """Conduct a search and return the most relevant content."""
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
    """Generate an AI response with or without web search context."""
    assistant_convo.append({'role': 'user', 'content': user_input})

    if search_or_not(user_input):
        context = ai_search(user_input)
        assistant_convo.pop()  # Remove duplicate user input
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
