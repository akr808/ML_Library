#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Support Vector Regression
#' @author: Anand

#Import the dataset#
dataset <- read.csv("../Data/Position_Salaries.csv")
dataset <- dataset[2:3]

#Training the SVR model
library(e1071)
regressor = svm(formula = Salary ~ ., data = dataset, type = "eps-regression")
y_pred <- predict(regressor, newdata = data.frame(Level = 6.5))

#Visualising the results#
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = "green") +
  ggtitle("Sal Vs Level") +
  xlab("Years of Experience") +
  ylab("Salary")

