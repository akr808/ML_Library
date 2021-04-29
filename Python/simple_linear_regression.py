# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)" 
Topic: Simple Linear Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../Data/Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

#ploting the data
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, lr.predict(x_train), color = "blue")
plt.title("Salary Vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary Earned")
plt.show()


plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, lr.predict(x_train), color = "blue")
plt.title("Salary Vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary Earned")
plt.show()
