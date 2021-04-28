# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)" 
Topic: Data Pre-Processing
@author: Anand
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from  sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("./Data/Data.csv")
X = dataset.iloc[:,:-1] #Selecting all but the last columns of the dataset
Y = dataset.iloc[:,-1] #Selecting the last column of the dataset

#Handling missing data
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X.iloc[:,1:])
X.iloc[:,1:] = imputer.transform(X.iloc[:,1:])
#Additional steps to iloc while fit is added to both source and target slices

#Encoding the independend data=>Dummy Variable for categorical columns
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Encoding the dependent value
le = LabelEncoder()
Y = le.fit_transform(Y)


#Split the data into train & test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

'''
#Feature Scaling---applied after the slpit as otherwise there could be data leakage from the test set into the train set
#Standardisation technique ==== (x - mean(x)) / (std dev(x))
#Normlisation technique ==== (x - min(x)) / (max(x) - min(x))
Standardisation is used for general data irrecpective of the nature,
Normalisation is applied on data that is distributed normally
'''
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.fit_transform(X_test[:,3:])
