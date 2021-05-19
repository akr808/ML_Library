#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Simple Linear Regression
#' @author: Anand

#Import the dataset#
dataset <- read.csv("../Data/Salary_Data.csv")

#Splitting the data into train and test#
library(caTools)
set.seed(123)
split = sample.split(dataset$YearsExperience, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = Salary ~ YearsExperience, data = training_set )
y_pred <- as.data.frame(predict(regressor, newdata = test_set))

#Visualising the results#
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), colour = "red") +
  geom_line(aes(training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = "blue") +
  ggtitle("Sal Vs Experience-Training") +
  xlab("Years of Experience") +
  ylab("Salary")
   
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), colour = "red") +
  geom_line(aes(training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = "blue") +
  ggtitle("Sal Vs Experience-Test Data") +
  xlab("Years of Experience") +
  ylab("Salary")