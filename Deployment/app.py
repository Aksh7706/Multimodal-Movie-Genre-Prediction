from __future__ import division, print_function
# coding=utf-8
import os
import re
import numpy as np
import pandas as pd
from string import punctuation

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from string import punctuation

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from tensorflow.python.framework.dtypes import _TYPE_TO_STRING
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

import cv2
import csv

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def resize_img(path):
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32)/255
        return img
    except Exception as e:
        print(str(e))
        return None

def clean_text(text):
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.lower().strip()
    text = ' '.join([i if i not in stop and i.isalpha()
                    else '' for i in text.lower().split()])
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
    text = re.sub(r"\s{2,}", " ", text)
    return text

def predict_helper(path, overview_txt):

    val_imgs = []

    img = resize_img(str(path))
    val_imgs.append(img)

    X_img_test = np.array(val_imgs)
    print(X_img_test.shape)

    # field names
    fields = ['overview']

    # data rows of csv file
    rows = [[overview_txt]]

    # name of csv file
    filename = "overview.csv"

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

    test = pd.read_csv("./overview.csv")
    dataset = pd.read_csv("./dataset/dataset_mod.csv")


    test['overview'] = test['overview'].astype(str)
    test['overview'] = test['overview'].apply(lambda text: clean_text(text))

    dataset['overview'] = dataset['overview'].astype(str)

    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = dataset['overview'].map(len).max()
    EMBEDDING_DIM = 300
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(dataset['overview'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Max len:', MAX_SEQUENCE_LENGTH)

    X_text_test = tokenizer.texts_to_sequences(test['overview'].values)
    X_text_test = pad_sequences(X_text_test, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of train tensor:', X_text_test.shape)

    word2vec_model = tf.keras.models.load_model(
        "./dataset/64_45.h5",
        custom_objects={'Functional': tf.keras.models.Model})

    out = word2vec_model.predict([X_img_test, X_text_test], batch_size=256)
    print(out)

    print(out.shape)

    y_pred = np.zeros(out.shape)
    y_pred[out > 0.5] = 1
    y_pred = np.array(y_pred)

    return y_pred.flatten()


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        desc = request.form.get("text")
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        output_g = predict_helper(file_path, desc)

        genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller',
                  'War', 'Western']
        result = ""   

        for i in range(len(output_g)):
            if output_g[i]:
                result+=genres[i] + " ,"
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
