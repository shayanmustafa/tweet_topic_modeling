# -*- coding: utf-8 -*-
"""
Created on Tue May 21 06:07:15 2019

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

# Load spreadsheet with tweets of all users
df = pd.read_excel('Twitter_timeline.xlsx', sheetname=None, ignore_index=True, sort=True)
cdf = pd.concat(df.values(), ignore_index=True, sort=False)

print("Number of tweets before removing invalid data: {0}".format(cdf.shape[0]))
#drop unnecessary attributes
cdf.drop(["id", "source", "created_at"], axis=1, inplace=True)
#fill null columns of "tags"
cdf["tags"].fillna(cdf[cdf.columns[2]], inplace=True)
#drop extra tag column
cdf.drop(cdf.columns[2], axis=1, inplace=True)
cdf.dropna(inplace=True)

tweet_text = cdf[['text']]
tweet_text['id'] = tweet_text.index
documents = tweet_text
#documents = documents.dropna(subset=['text'])


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
print(processed_docs[:5])