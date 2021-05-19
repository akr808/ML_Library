# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Dimentionality Reduction- LDA
@author: Anand
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

dataset = pd.read_csv("../Data/Wine.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


#Split the data into train & test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#Feature Scaling#
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Implement LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)




#Building the Binomial Logistic Rgression
blr = LogisticRegression(random_state=0)
blr.fit(X_train, y_train)

#Predicting using the model
y_pred = blr.predict(X_test)

#Computing the accuracy and confusion matrix for the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#Visualising the model.
#Visualising the training data
X_set, y_set = X_train,y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop =X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop =X_set[:,1].max() + 1, step = 0.01))
 
plt.contour(X1, X2, blr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            alpha =0.75, cmp = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title("Logistic Regression-Train")
plt.xlabel("PC1")
plt.ylabel("PC2") 
plt.legend()
plt.show()   

#Visualising the test data
X_set, y_set = X_test,y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop =X_set[:,0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop =X_set[:,1].max() + 1, step = 0.01))
 
plt.contour(X1, X2, blr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            alpha =0.75, cmp = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title("Logistic Regression-Test")
plt.xlabel("PC1")
plt.ylabel("PC2") 
plt.legend()
plt.show()   
