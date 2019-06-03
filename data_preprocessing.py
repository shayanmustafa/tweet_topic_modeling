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
import numpy as np
import nltk

# Load spreadsheet with tweets of all users
df = pd.read_excel('Twitter_timeline.xlsx', sheetname=None, ignore_index=True, sort=True)
cdf = pd.concat(df.values(), ignore_index=True, sort=False)
cdf.drop(cdf.columns[5],axis=1)
print(list(cdf))
