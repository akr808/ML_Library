# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Regression Model Evaluation
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

dataset = pd.read_csv("../Data/Reg_Selection_Data.csv")

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#Spilitting test and retrain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)

#Model training
rfr =  RandomForestRegressor(n_estimators = 10)
rfr.fit(X, y)

#Model Prediction
y_pred = rfr.predict(X_test)

#Model r2 score computation
print("R2 Score fot the model is : " + str(r2_score(y_test, y_pred)))

#PLotting the data
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_test, rfr.predict(X_test), color = "blue")
plt.title("Decision Tree Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
