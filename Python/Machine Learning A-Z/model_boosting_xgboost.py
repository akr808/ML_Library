# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Model Selection and Boosting- XGBoost
@author: Anand
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

dataset = pd.read_csv("../Data/Xgboost_Data.csv")


X = dataset.iloc[:,:-1] #Selecting all but the last columns of the dataset
y = dataset.iloc[:,-1] #Selecting the last column of the dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


#Training the XGBoost model
classifier = XGBClassifier()
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)

#Computing the accuracy and confusion matrix for the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


#Implement K-Fold Cross Validation#
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train)
print("Accuracy :{:.2f} % ".format(accuracies.mean() * 100))
print("Standard Deviation :{:.2f} % ".format(accuracies.std() * 100))