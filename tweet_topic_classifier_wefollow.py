import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Load spreadsheet with tweets of all users
tags = {}
tags['BN'] = 'business'
tags['EN'] = 'entertainment'
tags['ST'] = 'science_and_technology'
tags['PT'] = 'politics'
tags['ED'] = 'education'
tags['HT'] = 'health'
tags['RE'] = 'religion'
tags['SI'] = 'social_issues'
tags['SP'] = 'sports'

cdf = pd.DataFrame()

for tag in tags:
    df = pd.read_excel('wefollow/'+tags[tag]+'_timlines.xlsx', sheet_name=None, ignore_index=True, sort=True)
    temp_df = pd.concat(df.values(), ignore_index=True, sort=False)
    temp_df['tags'] = tag
    cdf = cdf.append(temp_df, ignore_index=True)

print("Number of tweets before removing invalid data: {0}".format(cdf.shape[0]))

cdf.drop(["id", "source", "created_at"], axis=1, inplace=True)
#cdf["tags"].fillna(cdf[cdf.columns[2]], inplace=True)
#cdf.drop(cdf.columns[2], axis=1, inplace=True)
#cdf.dropna(inplace=True)
#
tweet_text = cdf[['text', 'tags']]
tweet_text['id'] = tweet_text.index
documents = tweet_text
#print(documents.head())
#documents = documents.dropna(subset=['text'])

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
accuracyNB = train_model(naive_bayes.MultinomialNB(alpha=0.1), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracyNB)))

#SVC
print()
print ("~ Using Linear SVC ~ ")
accuracySVC = train_model(svm.LinearSVC(C=0.1), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracySVC)))

#LR
print()
print ("~ Using Logistic Regression ~ ")
accuracySVC = train_model(linear_model.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial'), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracySVC)))

#RF
print()
print ("~ Using Random Forest Classifier ~")
accuracyRF = train_model(RandomForestClassifier(n_estimators=500, max_depth=200, random_state=0), xtrain_count, train_y, xvalid_count, valid_y, verbose=True)
print ("Accuracy: {}%".format(formatAccuracy(accuracyRF)))

NBModel = naive_bayes.MultinomialNB(alpha=0.1).fit(xtrain_count, train_y)
SVCModel = svm.LinearSVC(C=0.1).fit(xtrain_count, train_y)
LRModel = linear_model.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial').fit(xtrain_count, train_y)
RFModel = RandomForestClassifier(n_estimators=500, max_depth=200, random_state=0).fit(xtrain_count, train_y)

def majority_voting(x_train, y_train, x_test, y_test):    
    NBPredict = NBModel.predict(x_test)
    SVCPredict = SVCModel.predict(x_test)
    LRPredict = LRModel.predict(x_test)
    RFPredict = RFModel.predict(x_test)
#    
    votingPred = []
    
    for i in range(len(y_test)):
        if NBPredict[i] == LRPredict[i] and NBPredict[i] == SVCPredict[i] and NBPredict[i] == RFPredict[i]:
            votingPred.append(NBPredict[i])
        elif NBPredict[i] == LRPredict[i] or NBPredict[i] == SVCPredict[i] or NBPredict[i] == RFPredict[i]:
            votingPred.append(NBPredict[i])
        elif LRPredict[i] == SVCPredict[i]:
            votingPred.append(LRPredict[i])
        else:
            votingPred.append(SVCPredict[i])
           
    return metrics.accuracy_score(votingPred, y_test)

def majorityVotingPredictor(inputX):
    NBPredict = NBModel.predict(count_vect.transform([inputX]))
    SVCPredict = SVCModel.predict(count_vect.transform([inputX]))
    LRPredict = LRModel.predict(count_vect.transform([inputX]))
    
    if NBPredict == LRPredict and NBPredict == SVCPredict:
        finalPred = NBPredict
    elif NBPredict == LRPredict or NBPredict == SVCPredict:
        finalPred = NBPredict
    elif LRPredict == SVCPredict:
        finalPred = LRPredict
    else:
        finalPred = SVCPredict
    
    return encoder.inverse_transform(finalPred)
    
    

#SVC
print()
print ("~ Using Majority Voting ~ ")
votingAccuracy = majority_voting(xtrain_count, train_y, xvalid_count, valid_y)
print ("Accuracy: {}%".format(formatAccuracy(votingAccuracy)))


custom_input = "Tweet about science and technology!"
result = majorityVotingPredictor(custom_input)
print("Predicting tweet: {}".format(custom_input))
print(result)
#plt.style.use('ggplot')
#plt.plot(ALPHAS, NBvalues,'r', label="Naive Bayes")
#plt.plot(ALPHAS, SVCvalues, 'g', label="LinearSVC")
#plt.plot(ALPHAS, LogRegvalues, 'b', label="Log Reg")
#plt.legend()
#plt.xlabel("ALPHAS")
#plt.ylabel("Accuracy")


#plt.show()