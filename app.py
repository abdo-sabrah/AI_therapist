#!/usr/bin/env python
# coding: utf-8

# In[13]:


from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import os
import random


# In[ ]:





app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to my Flask app!'


model = load_model('mental_chatbot_model_final.h5')

# Load the words and classes
words = pickle.load(open('word.pkl', 'rb'))
classes = pickle.load(open('classe.pkl', 'rb'))

# Load the intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    ERROR_THRESHOLD = 0.25
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "I'm sorry, I didn't understand that."
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json['message']
    ints = predict_class(user_input, model)
    res = getResponse(ints, intents)
    return jsonify({'response': res})

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=os.getenv('PORT', 8000))


# In[ ]:




