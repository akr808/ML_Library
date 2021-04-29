# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Polynomial Linear Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("../Data/Position_Salaries.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Simple Linear Model
lr = LinearRegression()
lr.fit(x, y)
x_test = np.array([[6.5]])
y_predict = lr.predict(x_test)

#Building Polynomial features with degree 2
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)

#Linear Reg with Polynomial
lr_poly_2 = LinearRegression()
lr_poly_2.fit(x_poly, y)
#y_predict_poly = lr_poly_2.predict(x_test)

plt.scatter(x,y,color = 'red')
plt.plot(x,lr.predict(x),color = "blue")
plt.plot(x,lr_poly_2.predict(x_poly),color = "green")
#Building Polynomial features with degree 4
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
plt.show()
lr_poly_2 = LinearRegression()
lr_poly_2.fit(x_poly, y)
#y_predict_poly = lr_poly_2.predict(x_test)

plt.scatter(x,y,color = 'red')
plt.plot(x,lr.predict(x),color = "blue")
plt.plot(x,lr_poly_2.predict(x_poly),color = "y")
#Predicting for the polynomial linear transformation
x_test_poly = poly_reg.fit_transform(x_test)
y_predict_poly = lr_poly_2.predict(x_test_poly)
plt.show()
