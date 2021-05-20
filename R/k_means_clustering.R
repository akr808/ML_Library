#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: K-Means Clustering
#' @author: Anand


#Import the dataset#
dataset <- read.csv("../Data/Mall_Customers.csv")
dataset <- dataset[,4:5]

#Using the Elbow curve to find the optimum plot
set.seed(0)
wcss <- vector()
for(i in 1:10) wcss[i] = sum(kmeans(dataset, centers = i)$withins)
plot(1:10, wcss, type = 'b', xlab = "Number of cluster", ylab = "WCSS", main = "Elbow Graph")

#Creating K-Means after inferring from the above
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster


#Visualising the clusters#
library(cluster)
clusplot(x = dataset, lines = 0, clus = y_kmeans, shade = TRUE,
         color = TRUE, labels = 0, plotchar = FALSE, main = "K-Means Clustering", xlab = "Annual Income", ylab = "Spending Score")
