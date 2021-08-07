# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Natural Language Processing--Bag of Words Approach.
@author: Anand
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

#importing the dataset
dataset = pd.read_csv("../Data/Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
corpus = []

#Cleaning all the reviews---removing the stop words(except 'not'), remove special chars.
for i in  range(len(dataset)):
    review = dataset.iloc[i,0]
    review = review.lower()
    review = re.sub('[^a-zA-Z]', " ", review)    
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = " ".join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Building the model
nbc = GaussianNB()
nbc.fit(X_train,y_train)

#Predict using the model
y_pred = nbc.predict(X_test)

#Computing the accuracy and confusion matrix for the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

