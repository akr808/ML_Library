# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Logistic Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("../Data/Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling the data
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#Building the Binomial Logistic Rgression
blr = LogisticRegression(random_state=0)
blr.fit(X_train, y_train)

#Predicting using the model
y_pred = blr.predict(X_test)

#Comparing the result with the expected value
result_comp = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

#Computing the accuracy and confusion matrix for the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#Visualising the model.
plt.scatter(X_train, blr.predict(X_train),color = "red")
plt.plot(X_test, blr.predict(X_test),color = "blue")
plt.show()
