#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Hierarchical Clustering
#' @author: Anand


#Import the dataset#
dataset <- read.csv("../Data/Mall_Customers.csv")
X <- dataset[,4:5]


#Computing the dendrogram to find the optimal cluster#
dendrogram <- hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendrogram, main = "Dendrogram Plot", xlab = "Customers", ylab = "Distance")


#Creating the cluster using the info from the dendrogram#
hc <- hclust(dist(X, method = "euclidean"), method = "ward.D")
cluste_vector = cutree(hc, k = 5)


#Visualising the clusters#
library(cluster)
clusplot(x = X, lines = 0, clus = cluste_vector, shade = TRUE,
         color = TRUE, labels = 1, plotchar = FALSE, main = "Hierarchical Clustering", 
         xlab = "Annual Income", ylab = "Spending Score")

