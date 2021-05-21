#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: XGBoost

#Import the dataset#
dataset <- read.csv("../Data/Churn_Modelling.csv")
dataset = dataset[,4:14]

#Encoding the categorical values#
levels <- unique(dataset$Geography)
labels <- 1 : (length(levels))
dataset$Geography = as.numeric(factor(dataset$Geography, levels = levels, labels = labels))
levels <- unique(dataset$Gender)
labels <- 0 : (length(levels) - 1)
dataset$Gender = as.numeric(factor(dataset$Gender, levels = levels, labels = labels))


#Splitting the data into train and test#
library(caTools)
set.seed(6)
split = sample.split(dataset$Exited, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Building the XGBoost#
library(xgboost)
classifier <- xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)


#Predicting the probabilities
y_pred = as.data.frame(predict(classifier, newdata = as.matrix(test_set[-11])))
y_pred$predict <- ifelse(y_pred$predict > 0.5, 1, 0)
#Computiing the model characteristics#
cm = table(test_set[,11], y_pred$predict)
