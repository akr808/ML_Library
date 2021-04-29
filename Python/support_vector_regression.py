# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Support Vector Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
dataset = pd.read_csv("../Data/Position_Salaries.csv")

X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1].values

#Feature Scaling
y = y.reshape(len(y),1)
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#SVR model building
svr = SVR(kernel = 'rbf',)
svr.fit(X,y)

#predicting using the model
x_test = sc_x.transform([[6.5]])
y_pred = svr.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)

#plotting the results
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(y),color = "red")
plt.plot(sc_x.inverse_transform(X),sc_y.inverse_transform(svr.predict(X)),color = "blue")
plt.show()
