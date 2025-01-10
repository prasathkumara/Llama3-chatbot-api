from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

CORS(app)

# Load Q&A Data from Excel
def load_qa_data(excel_file):
    df = pd.read_excel(excel_file)
    return df['Question'].tolist(), df['Answer'].tolist()

# Find the most similar question using cosine similarity
def get_best_answer(user_question, questions, answers):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_question] + questions)
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_idx = cosine_sim.argmax()
    return answers[best_match_idx] if cosine_sim[best_match_idx] > 0.2 else "Sorry, I don't have an answer for that."

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
    user_input = data.get('message')
    
    # Load Q&A Data
    questions, answers = load_qa_data("chatbot_doc.xlsx")
    
    # Get the best answer from Q&A data
    best_answer = get_best_answer(user_input, questions, answers)
    
    # Generate the chatbot's response using LLaMA
    context = data.get('context', '')
    result = chain.invoke({"context": context, "context_data": best_answer, "question": user_input})
    
    # Return the result
    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(debug=True)
