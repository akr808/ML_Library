# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: K-Nearest Neighbor
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

dataset = pd.read_csv("../Data/Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling the data
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#Build the KNN model
knn = KNeighborsClassifier(n_neighbors=5,   metric='minkowski' )
knn.fit(X_train, y_train)

#Predict using the model
y_pred = knn.predict(X_test)

#Comparing the result with the expected value
result_comp = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

#Computing the accuracy and confusion matrix for the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#Visualising the model.
#Visualising the training data
X_set, y_set = sc_x.inverse_transform(X_train),y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 10, stop =X_set[:,0].max() + 10, step = 1),
                      np.arange(start = X_set[:,1].min() - 1000, stop =X_set[:,1].max() + 1000, step = 1))
 
plt.contour(X1, X2, knn.predict(sc_x.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), 
            alpha =0.75, cmp = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-Nearest Neighbor-Train")
plt.xlabel("Age")
plt.ylabel("Estimated Salary") 
plt.legend()
plt.show()   

#Visualising the test data
X_set, y_set = sc_x.inverse_transform(X_test),y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 10, stop =X_set[:,0].max() + 10, step = 1),
                      np.arange(start = X_set[:,1].min() - 1000, stop =X_set[:,1].max() + 1000, step = 1))
 
plt.contour(X1, X2, knn.predict(sc_x.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), 
            alpha =0.75, cmp = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-Nearest Neighbor-Test")
plt.xlabel("Age")
plt.ylabel("Estimated Salary") 
plt.legend()
plt.show()   
