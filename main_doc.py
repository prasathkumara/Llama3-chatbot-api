from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

app = Flask(__name__)

CORS(app)

# Load Q&A Data from Excel
def load_qa_data(excel_file):
    df = pd.read_excel(excel_file)
    return df['Question'].tolist(), df['Answer'].tolist()

# Preprocess text
def preprocess_text(text):
    return ''.join(char for char in text.lower() if char not in string.punctuation)

# Find the most similar question using cosine similarity
def get_best_answer(user_question, questions, answers, threshold=0.5):
    user_question = preprocess_text(user_question)
    processed_questions = [preprocess_text(q) for q in questions]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_question] + processed_questions)
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_idx = cosine_sim.argmax()
    return answers[best_match_idx] if cosine_sim[best_match_idx] >= threshold else "not-found"

# Initialize Ollama LLaMA Model
model = OllamaLLM(model="llama3")
template = """
Answer the questions.

Here is the conversation history: {context}

Context from data: {context_data}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message').strip().lower()  # Normalize input

    # Predefined list of greeting keywords and responses
    greetings_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help you?",
    "thank you": "You're welcome! Is there anything else I can help with?",
    "bye": "Goodbye! Have a great day!"
}

# Check for greetings
    user_tokens = set(user_input.split())  # Split input into words
    for keyword, response in greetings_responses.items():
        if keyword in user_tokens:  # Check for exact word match
            return jsonify({'response': response})


    # Load Q&A Data
    questions, answers = load_qa_data("chatbot_doc.xlsx")
    
    # Get the best answer from Q&A data
    best_answer = get_best_answer(user_input, questions, answers)
    
    # Check if the best answer is the fallback response
    if best_answer == "not-found":
        # Directly return the fallback response
        return jsonify({'response': "Sorry, I don't have an answer for that."})
    else:
        # Generate the chatbot's response using LLaMA
        context = data.get('context', '')
        result = chain.invoke({
            "context": context, 
            "context_data": best_answer, 
            "question": user_input
        })
        return jsonify({'response': result})

if __name__ == '__main__':
     app.run(debug=False, host='0.0.0.0', port=5000)
