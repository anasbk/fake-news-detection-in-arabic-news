import os
import pandas as pd
import itertools
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,SpatialDropout1D
from keras.layers.embeddings import Embedding




from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


data = "إن القراء يقرؤون القرآن قراءة جميلة"
stop_words = set(stopwords.words('arabic'))



from nltk.stem import SnowballStemmer
ar_stemmer = SnowballStemmer("arabic")

'''
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

'''

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

def normalizeArabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return(text)


'''
fake = []
for folder in os.listdir("FakeNews/data"):
    for file in os.listdir("FakeNews/data/"+str(folder)):

        with open('FakeNews/data/'+folder+"/"+file, "r", encoding="utf8") as f:
            txt = f.read().rstrip()
            print(file)
            fake.append((txt,folder))

            #preprocessing

df = pd.DataFrame(fake)
print(df.shape)

df.to_csv("dataset.csv", sep=',',index=False)

'''
trainDF = pd.read_csv("dataset.csv", delimiter=",", encoding='utf-8')


df = pd.DataFrame([['إِنَّ الْقُرَّاْءَ يَقْرَؤُوْنَ الْقُرْآنَ قِرَاْءَةً جَمِيْلَــــــة', 'hello'],['الْقُرْآن heereَ','abcdef']], columns=['text', 'B'])


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

filtered_df = normalizeDF(trainDF)



filtered_df.to_csv('to_train.csv', sep=',')



Y = pd.get_dummies(trainDF['label']).values

train_x, test_x, train_y, test_y = train_test_split(trainDF['text'], trainDF['label'])

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

'''
# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
print(train_y.shape)
test_y = encoder.fit_transform(test_y)
'''

max_fatures = 5000

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word')
tfidf_vect.fit(trainDF['text'].values.astype('U'))
xtrain_tfidf = tfidf_vect.transform(train_x.values.astype('U')).toarray()
xvalid_tfidf = tfidf_vect.transform(test_x.values.astype('U')).toarray()

# counter

count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 1))
count_train = count_vec.fit(trainDF['text'].values.astype('U'))
bag_of_words_train = count_vec.transform(train_x.values.astype('U')).toarray()
bag_of_words_test = count_vec.transform(train_x.values.astype('U')).toarray()

print(xtrain_tfidf.shape)
print(xvalid_tfidf.shape)

print(bag_of_words_train.shape)
print(bag_of_words_test.shape)

########################################################################################

# Fitting the Logistic Regression into the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain_tfidf, train_y)

# Predicting the test set results

tested_y = classifier.predict(xvalid_tfidf)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(tested_y, test_y)

def plot_confusion_matrix(cm, classes=['fake','not fake'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



############### 2


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(bag_of_words_train, train_y)

# Predicting the test set results

tested_y = classifier.predict(xvalid_tfidf)

# Making the Confusion Matrix

print(tested_y.shape)
print(test_y.shape)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(tested_y, test_y)

def plot_confusion_matrix(cm, classes=['fake','not fake'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


exit()

# Visualising the Training set results

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:,0].min() -1, stop = X_Set[:, 0].max() +1, step = 0.01),
                     np.arange(start = X_Set[:,1].min() -1, stop = X_Set[:, 1].max() +1, step = 0.01))

plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j,1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression ( Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

##################################################################

exit()


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = xtrain_tfidf.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


batch_size = 32
model.fit(xtrain_tfidf, train_y, epochs=7, batch_size=batch_size, verbose=2)

exit()

'''
# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'].values.astype('U'))
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x.values.astype('U'))
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(test_x.values.astype('U'))


'''


for i in range(1,10):
    print(xtrain_tfidf[i])
    print(train_y[i])

#xtrain_tfidf = np.reshape(xtrain_tfidf, (xtrain_tfidf.shape[0],xtrain_tfidf.shape[1],1))
#xvalid_tfidf = np.reshape(xvalid_tfidf, (xvalid_tfidf.shape[0], 1, xvalid_tfidf.shape[1]))


print(xtrain_tfidf.shape)
xtrain_tfidf=xtrain_tfidf[:, :, None]

print(xtrain_tfidf.shape)

# defining the LSTM model
model = Sequential()
model.add(LSTM(200, input_shape=( xtrain_tfidf.shape[1], xtrain_tfidf.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(xtrain_tfidf, train_y, validation_split=0.4, epochs=5)

xtrain_tfidf=xtrain_tfidf[:, :, None]

print(xtrain_tfidf.shape)



def build_model(hidden_layer_size=128, dropout=0.2, learning_rate=0.01, verbose=0):
    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,input_shape=(4426, 5000)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(4426, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate))
    if verbose:
        print('Model Summary:')
        model.summary()
    return model

model = build_model(verbose=1)

model_conv = create_conv_model()








# remove diacretics

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





print(type(news_list))


# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(news_list)

print(X.shape)

# print(vectorizer.get_feature_names().shape)


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations




exit()
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

print(remove_punctuations('rearb! arae!br!aer ?'))


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


parser = argparse.ArgumentParser(description='Pre-process arabic text (remove '
                                             'diacritics, punctuations, and repeating '
                                             'characters).')


