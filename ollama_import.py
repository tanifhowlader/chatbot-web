import ollama
import requests
import logging
from googlesearch import search
from newspaper import Article
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from colorama import init, Fore, Style
import os

# Initialize colorama and Flask
init(autoreset=True)
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Store conversation history
assistant_convo = []

def search_or_not(prompt):
    """Decide if a web search is necessary."""
    response = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': "Determine if a web search is required."},
            {'role': 'user', 'content': prompt}
        ]
    )
    return 'true' in response['message']['content'].lower()

def generate_query(prompt):
    """Generate a search query from the user's input."""
    response = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': "Create a search query based on the user input."},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response['message']['content']

def google_search(query, num_results=5):
    """Perform a Google search and return URLs."""
    return list(search(query, num_results=num_results, stop=num_results, pause=2))

def scrape_webpage(url):
    """Scrape text from the given webpage URL."""
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
    """Check if the scraped content answers the user's query."""
    response = ollama.chat(
        model='llama3.2:latest',
        messages=[
            {'role': 'system', 'content': "Determine if this content answers the user's question."},
            {'role': 'user', 'content': f"User Question: {prompt}\nContent: {page_text}"}
        ]
    )
    return 'true' in response['message']['content'].lower()

def ai_search(prompt):
    """Conduct a web search and return relevant content."""
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
    """Generate AI response with or without search context."""
    assistant_convo.append({'role': 'user', 'content': user_input})

    if search_or_not(user_input):
        context = ai_search(user_input)
        assistant_convo.pop()  # Remove user's initial input
        combined_prompt = f"SEARCH RESULT: {context}\nUSER: {user_input}"
        assistant_convo.append({'role': 'user', 'content': combined_prompt})
    
    response = ollama.chat(
        model='llama3.2:latest',
        messages=assistant_convo
    )

    ai_reply = response['message']['content']
    assistant_convo.append({'role': 'assistant', 'content': ai_reply})
    return ai_reply

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    ai_response = generate_response(user_input)
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
