#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Dimensionality Reduction--PCA
#' @author: Anand


#Import the dataset#
dataset <- read.csv("../Data/Wine.csv")


#Splitting the data into train and test#
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling#
training_set[,-14] = scale(training_set[,-14])
test_set[,-14] = scale(test_set[,-14])

#Applying PCA to the data#
library(caret)
library(e1071)
pca = preProcess(x = training_set[,-14],method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c("PC1", "PC2", "Customer_Segment")]
test_set = predict(pca, test_set)
test_set = test_set[c("PC1", "PC2", "Customer_Segment")]
  
#Logistic Regression Classifier#
classifier = svm(formula = Customer_Segment ~ ., family = binomial, 
                 data = training_set,
                 type = 'C-classification', 
                 kernel = 'radial')

#Predicting the probabilities
y_pred = predict(classifier, newdata = test_set[-3])

#Computiing the model characteristics#
cm = table(test_set[,3], y_pred)


y_pred = data.frame(actual = test_set$Purchased, predicted = ifelse(prob_pred > 0.5, 1, 0))



#Visualising the results#
# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, type = 'response', newdata = grid_set)
plot(set[, -3],
     main = 'SVM Classifier (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2,'blue3',ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, type = 'response', newdata = grid_set)
plot(set[, -3],
     main = 'SVM Classifier (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2,'blue3',ifelse(set[, 3] == 1, 'green4', 'red3')))