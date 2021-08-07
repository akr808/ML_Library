# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Artificial Neural Network(Using Tensor Flow 2.0)
@author: Anand
"""

import pandas as pd
import numpy as np
import tensorflow as tf
tf.config.list_physical_devices('GPU')

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#importing the dataset
dataset = pd.read_csv("../Data/Churn_Modelling.csv")
X = dataset.iloc[:,3:-1]
y = dataset.iloc[:,-1]

#Encoding the categorial values
le = LabelEncoder()
X.iloc[:,2] = le.fit_transform(X.iloc[:,2])

#Dummy Variable for categorical columns
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Split the data into train & test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)


#Feature Scling#
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the ANN#

#initialise the ANN
ann = tf.keras.models.Sequential()

#Adding the input layers
#ann.add(tf.keras.layers.Dense(units = 12, activation = 'relu')) # 1st hidden layer

#Adding the hidden layers
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # 1st hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # 2nd hidden layer

#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) # Output has binary values

#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the ANN
ann.fit(X_train, y_train, batch_size = 1, epochs = 100)

#y_predict = ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
y_predict = ann.predict((X_test))
y_predict = (y_predict > 0.5)
accuray = accuracy_score(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)