import os
from flask import Flask, render_template, request
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer

# --- CLOUD NLTK FIX ---
# This ensures NLTK downloads correctly on the server
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
# ----------------------

app = Flask(__name__)
# ... (rest of your existing app.py code) ...

# Initialize Flask app
app = Flask(__name__)
stemmer = LancasterStemmer()

# --- 1. LOAD THE BRAIN ---
# We load the files created by train.py
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = pickle.load(open("chatbot_model.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

# --- 2. TEXT PROCESSING FUNCTIONS ---
def clean_up_sentence(sentence):
    # Tokenize and stem the user input
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    # Convert sentence into a bag of words (0s and 1s)
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag).reshape(1, -1)

# --- 3. THE WEB ROUTES ---

# Route for the Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Route for the Bot Logic
@app.route("/get")
def get_bot_response():
    # Get the user's message from the website
    userText = request.args.get('msg')
    
    # Use the model to predict the tag (e.g., 'fees')
    results = model.predict(bow(userText, words))
    tag = results[0]
    
    # Pick a random response from intents.json for that tag
    for i in intents['intents']:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            
    return str(response)

# Start the server
if __name__ == "__main__":
    app.run(debug=True)