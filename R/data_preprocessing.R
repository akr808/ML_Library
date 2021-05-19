#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Data Preprocessing
#' @author: Anand


#Import the dataset#
dataset <- read.csv("../Data/Data.csv")

#Handling missing data#
dataset$Age = ifelse(is.na(dataset$Age), mean(dataset$Age, na.rm = TRUE), dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), mean(dataset$Salary, na.rm = TRUE), dataset$Salary)

#Encode categorical values#
levels <- unique(dataset$Country)
labels <- 1:length(levels)
dataset$Country = factor(dataset$Country, levels = levels, labels = labels)

levels <- unique(dataset$Purchased)
labels <- 0 : (length(levels) - 1)
dataset$Purchased = factor(dataset$Purchased, levels = levels, labels = labels)

#Splitting the data into train and test#
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling#
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
