# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Model Selection and Boosting- Grid Search
@author: Anand
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from matplotlib.colors import ListedColormap

from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv("../Data/Social_Network_Ads.csv")


X = dataset.iloc[:,:-1] #Selecting all but the last columns of the dataset
y = dataset.iloc[:,-1] #Selecting the last column of the dataset

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



#Building the model
svm = SVC(kernel='rbf',random_state=0)
svm.fit(X_train, y_train)


#Implement Grid Search Cross Validation#
parameters = [{"C":[0.25,0.5,0.75, 1], "kernel": ["linear"]},
              {"C":[0.25,0.5,0.75, 1], "kernel": ["rbf"], "gamma" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] }]
grid_search = GridSearchCV(estimator = svm, param_grid = parameters, scoring='accuracy', cv = 10, n_jobs= -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Accuracy :{:.2f} % ".format(best_accuracy * 100))
print("Best Parameters : " , best_parameters)


#Predict using the model
y_pred = svm.predict(X_test)

#Computing the accuracy and confusion matrix for the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#Visualising the model.
#Visualising the training data
X_set, y_set = sc.inverse_transform(X_train),y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop =X_set[:,0].max() + 1, step = 0.5),
                      np.arange(start = X_set[:,1].min() - 1, stop =X_set[:,1].max() + 1, step = 0.5))
 
plt.contour(X1, X2, svm.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), 
            alpha =0.75, cmp = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Support Vector Machine-Train")
plt.xlabel("Age")
plt.ylabel("Estimated Salary") 
plt.legend()
plt.show()   

#Visualising the test data
X_set, y_set = sc.inverse_transform(X_test),y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop =X_set[:,0].max() + 1, step = 0.5),
                      np.arange(start = X_set[:,1].min() - 1, stop =X_set[:,1].max() + 1, step = 0.5))
 
plt.contour(X1, X2, svm.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), 
            alpha =0.75, cmp = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Support Vector Machine-Test")
plt.xlabel("Age")
plt.ylabel("Estimated Salary") 
plt.legend()
plt.show()   
