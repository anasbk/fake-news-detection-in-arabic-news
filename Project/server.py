# -*- coding: utf-8 -*-


from flask import Flask, render_template, request
from werkzeug import secure_filename

from keras import backend as K
import os
import pandas as pd
import itertools
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas
import pickle

from keras.models import load_model


def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_punctuations(text):
    translator = str.maketrans(' ', ' ', punctuations_list)
    return text.translate(translator)

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def removeDiacretics(news_list):

    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    temp_list = list()
    for news in news_list:
        text = re.sub(arabic_diacritics, '', news)
        temp_list.append(text)

    return temp_list



def normalizeDF(df):

    for index, row in df.iterrows():

        text = df.iloc[index].text
        if type(text) == str:

            text = removeDiacretics([text])

            word_tokens = word_tokenize(text[0])

            filtered_sentence = [w for w in word_tokens if not w in stop_words]

            stemmed_words = [ar_stemmer.stem(word) for word in filtered_sentence]

            stemmed_sentence = ' '.join(stemmed_words)

            df.iloc[index].text = stemmed_sentence

    return df


stop_words = set(stopwords.words('arabic'))



def normalizeArabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return(text)



from nltk.stem import SnowballStemmer

ar_stemmer = SnowballStemmer("arabic")


app = Flask(__name__)


@app.route('/')
def index():

    # these accuracies are real, writen directly just to make code easy
    data = {
        'acc1':88.54,
        'acc2':86.25,
        'acc3':85.96,
        'acc4':84.49
        }
	
    return render_template('index.html',data=data)


@app.route('/synthesis', methods = ['POST'])
def synthesis():
    
    news_input = request.form.get('txt')

    empty_list = list()
    empty_list.append(news_input)
    

    filtered_df = normalizeDF(pd.DataFrame(empty_list,columns=['text']))

    model1_name = "logistic_reg.pickle"
    model2_name = "nayve_bayes.pickle"
    model3_name = "random_forest.pickle"
    model4_name = "wembedlstm.h5"
    model5_name = "cnnmodel.h5"

    loaded_model1 = pickle.load(open("models/"+model1_name, 'rb'))
    loaded_model2 = pickle.load(open("models/"+model2_name, 'rb'))
    loaded_model3 = pickle.load(open("models/"+model3_name, 'rb'))

    K.clear_session()
    
    #loaded_model4 = load_model("models/"+model4_name)
    loaded_model5 = load_model("models/"+model5_name)
    
    tfidfvect = pickle.load(open("models/vectorizer.pickle","rb"))

    news_vector = tfidfvect.transform(filtered_df.values.astype('U')[0]).toarray()
  

    data = {
        'input':news_input,
        'pred1':loaded_model1.predict(news_vector)[0],
        'pred2':loaded_model2.predict(news_vector)[0],
        'pred3':loaded_model3.predict(news_vector)[0],
        'pred4':loaded_model4.predict(news_vector)[0],
        'pred5':loaded_model5.predict(news_vector)[0],
        }

    
    return render_template('synthesis.html',data=data)
		
if __name__ == '__main__':
   app.run(debug = True)
