# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Random Forest Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
dataset = pd.read_csv("../Data/Position_Salaries.csv")

X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]

#Creating the Ranodom Forest Model
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
rfr.fit(X, y)

#Predicting using the model
x_test = [[6.5]]
y_pred = rfr.predict(x_test)

#PLotting the data(hi res)
X_grid = np.arange(min(X.values), max(X.values), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, rfr.predict(X_grid), color = "blue")
plt.title("Random Forest Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
