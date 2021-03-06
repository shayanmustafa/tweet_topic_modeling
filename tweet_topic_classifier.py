# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:25:08 2019

@author: Shayan, Kumail, Ehtasham
"""

import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
#import tensorflow as tf
#import numpy as np

#from sklearn.preprocessing import LabelBinarizer, LabelEncoder
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.preprocessing import text, sequence
#from keras import utils
#
#import nltk

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
cdf = cdf[cdf.tags != "RT"]

#tweet_text = cdf[['text', 'tags']]
#tweet_text['id'] = tweet_text.index
#documents = tweet_text

my_tags = ['ST', 'PT', 'HT', 'BN', 'ED', 'SP', 'EN', 'SI', 'RE', 'GM', 'NW', 'WB']

#print(documents.head())
#documents = documents.dropna(subset=['text'])

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text_to_preprocess):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text_to_preprocess, pos='v'))

def preprocess(text_to_preprocess):
    result = []
    for token in gensim.utils.simple_preprocess(text_to_preprocess):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


cdf['text'] = cdf['text'].map(preprocess)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(cdf['text'], cdf['tags'], random_state = 0)

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



def train_model(clf, x_train, y_train, x_test, y_test, verbose=False):
    clf = clf.fit(x_train, y_train)    
    pred = clf.predict(x_test)    
    
    if verbose:
        tweet = "This is a tweet about Science and Technology, wow!"
        print("Predicting tweet: {}".format(tweet))
        custom_pred = clf.predict(count_vect.transform([tweet]))
        print("Result: {}".format(encoder.inverse_transform(custom_pred)))
    
    return metrics.accuracy_score(pred, y_test)

#NBvalues = []
#SVCvalues = []
#LogRegvalues = []
#ALPHAS = [0.001, 0.005, 0.007, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4]
#length = len(ALPHAS)
#print("*******")
#for i in range(length):
#    SVCvalues.append(train_model(svm.LinearSVC(C=ALPHAS[i]), xtrain_count, train_y, xvalid_count, valid_y))
#    NBvalues.append(train_model(naive_bayes.MultinomialNB(alpha=ALPHAS[i]), xtrain_count, train_y, xvalid_count, valid_y))
#    LogRegvalues.append(train_model(linear_model.LogisticRegression(C=ALPHAS[i], solver='lbfgs', multi_class='multinomial'), xtrain_count, train_y, xvalid_count, valid_y))
#    print('Alpha = {:.2f}'
#         .format(ALPHAS[i]))
#    print ("Accuracy: {}%\n".format(round(NBvalues[i]*100, 3)))


def formatAccuracy(acc):
    return round(acc*100, 3)

#NB
print ("~ Using Naive Bayes ~ ")
NBModel = naive_bayes.MultinomialNB(alpha=0.1)
accuracyNB = train_model(NBModel, xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracyNB)))
NBModel = NBModel.fit(xtrain_count, train_y) 
pred = NBModel.predict(xvalid_count)
print(classification_report(valid_y, pred,target_names=my_tags))

#SVC
print()
print ("~ Using Linear SVC ~ ")
SVCModel = svm.LinearSVC(C=0.1)
accuracySVC = train_model(SVCModel, xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracySVC)))
SVCModel = SVCModel.fit(xtrain_count, train_y) 
pred = SVCModel.predict(xvalid_count)
print(classification_report(valid_y, pred,target_names=my_tags))

#LR
print()
print ("~ Using Logistic Regression ~ ")
LRModel = linear_model.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
accuracySVC = train_model(LRModel, xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracySVC)))
LRModel = LRModel.fit(xtrain_count, train_y) 
pred = LRModel.predict(xvalid_count)
print(classification_report(valid_y, pred,target_names=my_tags))

#RF
print()
print ("~ Using Random Forest Classifier ~")
RFModel = RandomForestClassifier(n_estimators=500, max_depth=200, random_state=0)
accuracyRF = train_model(RFModel, xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracyRF)))
RFModel = RFModel.fit(xtrain_count, train_y) 
pred = RFModel.predict(xvalid_count)
print(classification_report(valid_y, pred,target_names=my_tags))
    
#LDA
print()
print ("~ Using LDA ~ ")
LDAModel = LinearDiscriminantAnalysis()
accuracyLDA = train_model(LDAModel, xtrain_count.toarray(), train_y, xvalid_count.toarray(), valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracyLDA)))
LDAModel = LDAModel.fit(xtrain_count.toarray(), train_y) 
pred = LDAModel.predict(xvalid_count)
print(classification_report(valid_y, pred,target_names=my_tags))

#NN
print()
print ("~ Using NN ~ ")
NNModel = MLPClassifier(activation='relu', max_iter=800, solver='lbfgs', learning_rate_init=0.005, hidden_layer_sizes=(46, 44), random_state=1)
accuracyNN = train_model(NNModel, xtrain_count.toarray(), train_y, xvalid_count.toarray(), valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracyNN)))
NNModel = NNModel.fit(xtrain_count, train_y) 
pred = NNModel.predict(xvalid_count)
print(classification_report(valid_y, pred,target_names=my_tags))


#NBModel = naive_bayes.MultinomialNB(alpha=0.1).fit(xtrain_count, train_y)
#SVCModel = svm.LinearSVC(C=0.1).fit(xtrain_count, train_y)
#LRModel = linear_model.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial').fit(xtrain_count, train_y)
#LDAModel = LinearDiscriminantAnalysis().fit(xtrain_count.toarray(), train_y)
#RFModel = RandomForestClassifier(n_estimators=500, max_depth=200, random_state=0).fit(xtrain_count, train_y)

def majority_voting(x_train, y_train, x_test, y_test):    
    NBPredict = NBModel.predict(x_test)
    SVCPredict = SVCModel.predict(x_test)
    LRPredict = LRModel.predict(x_test)
    RFPredict = RFModel.predict(x_test)
    NNPredict = NNModel.predict(x_test)
#    
    votingPred = []
    
    for i in range(len(y_test)):
        for_pred = [NBPredict[i], LRPredict[i], SVCPredict[i], RFPredict[i], NNPredict[i]]
        highest = for_pred[0]
        count = 0
        for current_pred in for_pred: 
            new_count = 0
            for test_pred in for_pred:
                if current_pred == test_pred:
                    new_count = new_count + 1
            if new_count > count:
                highest = current_pred
                count = new_count
        votingPred.append(highest)
           
    return metrics.accuracy_score(votingPred, y_test)

def majorityVotingPredictor(inputX):
    NBPredict = NBModel.predict(count_vect.transform([inputX]))
    SVCPredict = SVCModel.predict(count_vect.transform([inputX]))
    LRPredict = LRModel.predict(count_vect.transform([inputX]))
    RFPredict = RFModel.predict(count_vect.transform([inputX]))
    NNPredict = NNModel.predict(count_vect.transform([inputX]))
    
    print("NB: {}".format(encoder.inverse_transform(NBPredict)))
    print("SVC: {}".format(encoder.inverse_transform(SVCPredict)))
    print("LR: {}".format(encoder.inverse_transform(LRPredict)))
    print("RF: {}".format(encoder.inverse_transform(RFPredict)))
    print("NN: {}".format(encoder.inverse_transform(NNPredict)))
    
    for_pred = [NBPredict, LRPredict, SVCPredict, RFPredict, NNPredict]
    highest = for_pred[0]
    count = 0
    for current_pred in for_pred: 
        new_count = 0
        for test_pred in for_pred:
            if current_pred == test_pred:
                new_count = new_count + 1
        if new_count > count:
            highest = current_pred
            count = new_count
    
    return encoder.inverse_transform(highest)
    
    

#SVC
print()
print ("~ Using Majority Voting ~ ")
votingAccuracy = majority_voting(xtrain_count, train_y, xvalid_count, valid_y)
print ("Accuracy: {}%".format(formatAccuracy(votingAccuracy)))


custom_input = "This is a tweet about science and technology!"
print("Predicting tweet: {}".format(custom_input))
result = majorityVotingPredictor(preprocess(custom_input))
print("Majority Voting: {}".format(result))
