#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Artificial Neural Network
#' @author: Anand


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

#Feature Scaling#
training_set[,-11] = scale(training_set[,-11])
test_set[,-11] = scale(test_set[,-11])

#Building the ANN#
library(h2o)
h2o.init(nthreads = -1)
classifier <- h2o.deeplearning(y = 'Exited', 
                               training_frame = as.h2o(training_set),
                               activation = "Rectifier", 
                               hidden = c(4,12), 
                               epochs = 100,
                               train_samples_per_iteration = -2)


#Predicting the probabilities
y_pred = as.data.frame(predict(classifier, newdata = as.h2o(test_set[-11])))
y_pred$predict <- ifelse(y_pred$predict > 0.5, 1, 0)
#Computiing the model characteristics#
cm = table(test_set[,11], y_pred$predict)
h2o.shutdown()
