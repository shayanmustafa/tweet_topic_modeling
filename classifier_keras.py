# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 00:27:01 2019

@author: Shayan
"""

import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup



df = pd.read_excel('Twitter_timeline.xlsx', sheet_name=None, ignore_index=True, sort=True)
cdf = pd.concat(df.values(), ignore_index=True, sort=False)
cdf.drop(["id", "source", "created_at"], axis=1, inplace=True)
cdf["tags"].fillna(cdf[cdf.columns[2]], inplace=True)
#drop extra tag column
cdf.drop(cdf.columns[2], axis=1, inplace=True)
cdf.dropna(inplace=True)

cdf = cdf[cdf.tags != "RJ"]
cdf = cdf[cdf.tags != "Rj"]
cdf = cdf[cdf.tags != "ET"]
cdf = cdf[cdf.tags != "EH"]
cdf = cdf[cdf.tags != "RH"]
cdf = cdf[cdf.tags != "SI"]
#print(cdf.head(10))
words = cdf['text'].apply(lambda x: len(x.split(' '))).sum()
print(words)
my_tags = ['ST', 'PT', 'HT', 'BN', 'ED', 'SP', 'EN', 'SI', 'RE', 'GM', 'NW', 'WB']
#plt.figure(figsize=(10,4))
#cdf.tags.value_counts().plot(kind='bar');


def print_plot(index):
    example = cdf[cdf.index == index][['text', 'tags']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Tag:', example[1])
        
#print_plot(10)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    #text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
  

cdf['text'] = cdf['text'].apply(clean_text)

X = cdf.text
y = cdf.tags
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

train_size = int(len(cdf) * .7)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(cdf) - train_size))


train_posts = cdf['text'][:train_size]
train_tags = cdf['tags'][:train_size]

test_posts = cdf['text'][train_size:]
test_tags = cdf['tags'][train_size:]

max_words = 100000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

batch_size = 32
epochs = 5


# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])



for i in range(length):
    SVCvalues.append(train_model(svm.LinearSVC(C=ALPHAS[i]), xtrain_count, train_y, xvalid_count, valid_y))
    NBvalues.append(train_model(naive_bayes.MultinomialNB(alpha=ALPHAS[i]), xtrain_count, train_y, xvalid_count, valid_y))
    LogRegvalues.append(train_model(linear_model.LogisticRegression(C=ALPHAS[i], solver='lbfgs', multi_class='multinomial'), xtrain_count, train_y, xvalid_count, valid_y))
    print('Alpha = {:.2f}'
         .format(ALPHAS[i]))
    print ("Accuracy: {}%\n".format(round(NBvalues[i]*100, 3)))
    
print ("~ Using Naive Bayes ~ ")
accuracyNB = train_model(naive_bayes.MultinomialNB(alpha=0.1), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracyNB*100, 3)))    

#SVC
print()
print ("~ Using Linear SVC ~ ")
accuracySVC = train_model(svm.LinearSVC(C=0.1), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracySVC*100, 3)))

#SVC
print()
print ("~ Using Logistic Regression ~ ")
accuracySVC = train_model(linear_model.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial'), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(round(accuracySVC*100, 3)))


plt.style.use('ggplot')
plt.plot(ALPHAS, NBvalues,'r', label="Naive Bayes")
plt.plot(ALPHAS, SVCvalues, 'g', label="LinearSVC")
plt.plot(ALPHAS, LogRegvalues, 'b', label="Log Reg")
plt.legend()
plt.xlabel("ALPHAS")
plt.ylabel("Accuracy")


plt.show()