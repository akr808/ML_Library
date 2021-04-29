# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)" 
Topic: Multiple Linear Regression
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../Data/50_Startups.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
x = ct.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


mlr = LinearRegression()
mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

y_test = np.array(y_test)
np.set_printoptions(precision = 2)
concat_data = np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)),axis=1)