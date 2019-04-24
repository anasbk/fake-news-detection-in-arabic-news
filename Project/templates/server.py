from flask import Flask, render_template, request
from werkzeug import secure_filename

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram


import json
from pprint import pprint

import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras import regularizers

import pandas as pd
import os
import glob 
import csv
from keras.models import load_model
import warnings

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    data = {}

    return render_template('index.html',data=data)

	
@app.route('/upload')
def upload():

    return render_template('upload.html')

@app.route('/synthesis', methods = ['GET', 'POST'])
def synthesis():
    if request.method == 'POST':
        print(request)

    return render_template('synthesis.html',data=data)
		
if __name__ == '__main__':
   app.run(debug = True)
