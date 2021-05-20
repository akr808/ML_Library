#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Multiple Linear Regression
#' @author: Anand

#Import the dataset#
dataset <- read.csv("../Data/Position_Salaries.csv")

dataset <- dataset[2:3]

#Liner Regression for the same data
lin_reg <- lm(Salary ~ ., data = dataset)
summary(lin_reg)


#Polynomial Model#
dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4
poly_reg <- lm(formula = Salary ~ ., data = dataset)
summary(poly_reg)

#Predicting the result
test_data <-data.frame(Level = c(6.5))
y_pred_linear <- predict(lin_reg, newdata = test_data)
test_data$Level2 = test_data$Level ^ 2
test_data$Level3 = test_data$Level ^ 3 
test_data$Level4 = test_data$Level ^ 4 
y_pred_poly <- predict(poly_reg, newdata = test_data)

#Visualising the results#
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = "blue") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = "green") +
  ggtitle("Sal Vs Level") +
  xlab("Years of Experience") +
  ylab("Salary")
