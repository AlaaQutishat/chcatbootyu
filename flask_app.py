#!/path/to/myenv/bin/python

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy as np
from flask import Flask, request
from keras.models import load_model
import json
import os
from flask import json
from werkzeug.exceptions import HTTPException


app = Flask(__name__)
my_dir = os.path.dirname(__file__)
intents_file_path = os.path.join(my_dir, 'intents.json')
model_file_path = os.path.join(my_dir, 'chatbot_model.h5')
words_file_path = os.path.join(my_dir, 'words.pkl')
classes_file_path = os.path.join(my_dir, 'classes.pkl')
intents = json.loads(open(intents_file_path).read())
words = pickle.load(open(words_file_path, 'rb'))
classes = pickle.load(open(classes_file_path, 'rb'))
model = load_model(model_file_path)
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
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = np.where(res > ERROR_THRESHOLD)[0]
    # sort by strength of probability
    results = sorted(results, key=lambda i: res[i], reverse=True)
    return_list = [{"intent": classes[i], "probability": str(res[i])} for i in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.route('/predict/<string:name>', methods=['GET'])
def predict(name):
    # sentence_words = lemmatizer.lemmatize("Hello".lower(), pos='v')
    p = bow(name, words, show_details=False)
    # res = model.predict(np.array([p]))[0]
    return  "" + str(model.predict(np.array([p])))

if __name__ == '__main__':
    app.run()