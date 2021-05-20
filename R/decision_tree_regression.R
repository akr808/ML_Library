#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Decision Tree Regression
#' @author: Anand

#Import the dataset#
dataset <- read.csv("../Data/Position_Salaries.csv")
dataset <- dataset[2:3]

#Training the model
library(rpart)
regressor = rpart(formula = Salary ~ ., data = dataset,control = rpart.control(minsplit = 1))
y_pred <- predict(regressor, newdata = data.frame(Level = 6.5))

#Visualising the results#
library(ggplot2)
x_grid =seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = "red") +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), colour = "blue") +
  ggtitle("Sal Vs Level") +
  xlab("Years of Experience") +
  ylab("Salary")

