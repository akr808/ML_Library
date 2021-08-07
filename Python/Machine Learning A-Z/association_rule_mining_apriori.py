# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Association Rule Mining--Apriori
@author: Anand
"""

import pandas as pd

from apyori import apriori


#importing the dataset
dataset = pd.read_csv("../Data/Market_Basket_Optimisation.csv", header=None).values
transactions = []

for i in range(len(dataset)):
    transactions.append([str(j) for j in dataset[i,:]])
#Threshold values
min_tran_count = 3 #Number of transactions per day
min_supp = round(min_tran_count * 7 / len(dataset), 3)


#Building the apriori model
rules = apriori(transactions = transactions, min_support = min_supp, min_confidence = 0.2, min_lift = 3,max_length = 2)

#Fetching the rules
results = list(rules)

#Function to fetch the values from the results and put it as a dataframe and sort by the lift
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return pd.DataFrame(list(zip(lhs, rhs, supports, confidences, lifts)), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift']).nlargest(n=10, columns= "Lift")

rules_df = inspect(results)
