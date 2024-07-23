from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model and vectorizer
with open('chatbot_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    user_message_transformed = vectorizer.transform([user_message])
    
    response = model.predict(user_message_transformed)
    return jsonify({'response': response[0]})

if __name__ == '__main__':
    app.run(debug=True)
