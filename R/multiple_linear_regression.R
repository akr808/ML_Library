#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Multiple Linear Regression
#' @author: Anand

#Import the dataset#
dataset <- read.csv("../Data/50_Startups.csv")


#Encode categorical values#
levels <- unique(dataset$State)
labels <- 1 : (length(levels))
dataset$State = factor(dataset$State, levels = levels, labels = labels)


#Splitting the data into train and test#
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Multiple Linear Regression#
regressor = lm(formula = Profit ~ ., data = training_set )
y_pred <- as.data.frame(predict(regressor, newdata = test_set))

summary(regressor)

#Backward Elimination method#Removed State, Administration, Marketing.Spend
regressor = lm(formula = Profit ~ R.D.Spend  , data = training_set )
summary(regressor)
y_pred <- as.data.frame(predict(regressor, newdata = test_set))