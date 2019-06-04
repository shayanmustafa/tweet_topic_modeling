# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:25:08 2019

@author: Shayan
"""

import pandas as pd
import sys
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

# Load spreadsheet with tweets of all users
df = pd.read_excel('Twitter_timeline.xlsx', sheet_name=None, ignore_index=True, sort=True)
cdf = pd.concat(df.values(), ignore_index=True, sort=False)

#print("Number of tweets before removing invalid data: {0}".format(cdf.shape[0]))
#drop unnecessary attributes
cdf.drop(["id", "source", "created_at"], axis=1, inplace=True)
#fill null columns of "tags"
cdf["tags"].fillna(cdf[cdf.columns[2]], inplace=True)
#drop extra tag column
cdf.drop(cdf.columns[2], axis=1, inplace=True)
cdf.dropna(inplace=True)
cdf = cdf[cdf.tags != "RJ"]
cdf = cdf[cdf.tags != "Rj"]
cdf = cdf[cdf.tags != "ET"]
cdf = cdf[cdf.tags != "EH"]
cdf = cdf[cdf.tags != "RH"]

tweet_text = cdf[['text', 'tags']]
tweet_text['id'] = tweet_text.index
documents = tweet_text
#print(documents.head())
#documents = documents.dropna(subset=['text'])

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(cdf['text'], cdf['tags'])

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(cdf['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
#print(xtrain_count[0])

#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(xtrain_count)
#X_valid_tfidf = tfidf_transformer.fit_transform(xvalid_count)

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text_to_preprocess):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text_to_preprocess, pos='v'))

def preprocess(text_to_preprocess):
    result = []
    for token in gensim.utils.simple_preprocess(text_to_preprocess):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_docs = documents['text'].map(preprocess)
cdf['text'] = processed_docs


def train_model(clf, x_train, y_train, x_test, y_test):
    model = clf.fit(x_train, y_train)    
    pred = model.predict(x_test)    
    return metrics.accuracy_score(pred, y_test)

#NB
accuracyNB = train_model(naive_bayes.MultinomialNB(alpha=0.1), xtrain_count, train_y, xvalid_count, valid_y)
print ("NB, Count Vectors: {}%".format(round(accuracyNB*100, 3)))

#SVC
accuracySVC = train_model(svm.LinearSVC(C=0.1), xtrain_count, train_y, xvalid_count, valid_y)
print ("SVC: {}%".format(round(accuracySVC*100, 3)))
