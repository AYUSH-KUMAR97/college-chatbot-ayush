import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

# 1. Load the data we saved during training
with open("intents.json") as file:
    data = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = pickle.load(open("chatbot_model.pkl", "rb"))

def clean_up_sentence(sentence):
    # Tokenize and stem the user's input
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    # Convert the user's sentence into 0s and 1s
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag).reshape(1, -1)

def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predict the intent
        results = model.predict(bow(inp, words))
        tag = results[0]

        # Find the response from our intents.json
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print("Bot: " + random.choice(responses))

chat()