# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Hierarchical Clustering Algorithm
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

#importing the dataset
dataset = pd.read_csv("../Data/Mall_Customers.csv")
X = dataset.iloc[:,-2:].values

#Dendrogram for the plot

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.xlabel("Customers")
plt.ylabel("Distance") 
plt.title("Dendrogram") 
plt.show() 

#Building the Hierarchical clustering model
aggCluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = aggCluster.fit_predict(X)

#Plotting the scatter plot
colors = ("Green","pink","Orange","Blue","Red")
for i in range(5):
    rows = y_hc == i
    plt.scatter(X[rows,0],X[rows,1],c=colors[i], label = "Cluster " + str(i + 1))
#plt.scatter(aggCluster.cluster_centers_[:,0],aggCluster.cluster_centers_[:,1],c='yellow',s=100)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score") 
plt.title("Hierarchical Clustering") 
plt.legend()
plt.show() 