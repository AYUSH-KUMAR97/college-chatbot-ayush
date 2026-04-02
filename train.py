import json
import nltk
import numpy as np
import pickle
import random
from nltk.stem import LancasterStemmer
from sklearn.svm import SVC

# 1. Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = LancasterStemmer()

# 2. Load the intents file
with open('intents.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# 3. Pre-process the data
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add to documents in our corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# 4. Create the 'training' data (Bag of Words)
training_bags = []
training_labels = []

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
    # Create the bag of words (1 if word exists, 0 if not)
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    training_bags.append(bag)
    training_labels.append(doc[1]) # The tag name

# 5. Build and Save the Model using Scikit-Learn (The Brain)
# We use an SVM classifier which is great for small text datasets
model = SVC(kernel='linear', probability=True)
model.fit(np.array(training_bags), np.array(training_labels))

# 6. Save everything for the chatbot script
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
pickle.dump(model, open('chatbot_model.pkl', 'wb'))

print("--- Success! ---")
print(f"Successfully processed {len(documents)} patterns.")
print(f"Model saved as chatbot_model.pkl")
print(f"Categories (Tags) trained: {classes}")