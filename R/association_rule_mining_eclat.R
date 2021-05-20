#' 
#' ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
#' Topic: Association Rule Mining - Eclat
#' @author: Anand


#Import the dataset#
library(arules)
#Create the sparse matrix#
dataset <- read.transactions("../Data/Market_Basket_Optimisation.csv",sep = ',', rm.duplicates = TRUE)

#Frequency PLot for the transaction
itemFrequencyPlot(dataset, topN = 100)


#Building the apriori model
rules <- eclat(data = dataset, parameter = list(support = 0.05, minlen = 2) )

#Visualising the results#
inspect(sort(rules,by = 'support'))
