# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Decision Tree Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv("../Data/Position_Salaries.csv")

X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]

#Creating the Decision Tree Regression model
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X, y)

#Predicting using the DT model
x_test = [[6.5]]
y_pred = dtr.predict(x_test)

#PLotting the data(hi res)
X_grid = np.arange(min(X.values), max(X.values), 0.0001)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, dtr.predict(X_grid), color = "blue")
plt.title("Decision Tree Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
