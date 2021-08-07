# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: K-Means Algorithm
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#importing the dataset
dataset = pd.read_csv("../Data/Mall_Customers.csv")
seed = 42

#Using only 2 columns of the dataset for clustering
X = dataset.iloc[:,-2:].values
WCSS = list()
no_of_cluster = list() 

#Computing the WCSS for each iteration of kmeans
for i in range(1,20):
    kmeans = KMeans(n_clusters=i, random_state=seed, init = 'k-means++')
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
    no_of_cluster.append(i)
    
plt.scatter(no_of_cluster,WCSS)
plt.plot(no_of_cluster,WCSS)
plt.show()


kmeans = KMeans(n_clusters=5, random_state=seed, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

#Method 1 to plot
# y = kmeans.labels_
# plt.scatter(X[:,0],X[:,1],cmap='viridis',c=y_kmeans)
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='red',s=100)
# plt.show()

#Method 2 to plot
colors = ("Green","pink","Orange","Blue","Red")
for i in range(5):
    rows = y_kmeans == i
    plt.scatter(X[rows,0],X[rows,1],c=colors[i], label = "Cluster " + str(i + 1))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',s=100)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score") 
plt.title("K-Means Clustering") 
plt.legend()
plt.show() 