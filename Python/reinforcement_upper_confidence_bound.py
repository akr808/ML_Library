# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Reinforcement Learning--Upper Confidence Bound Algorithm.
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt

import math

#importing the dataset
dataset = pd.read_csv("../Data/Ads_CTR_Optimisation.csv")
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_rewards = 0

for i in range(N):
    ucb = []
    ad = 0
    max_upper_bound = 0
    for j in range(d):
        if number_of_selections[j] > 0:
            average_reward = sum_of_rewards[j] / number_of_selections[j]
            delta_j = math.sqrt(3/2 * math.log(i + 1) / number_of_selections[j])
            upper_bound = average_reward + delta_j
        else:
            upper_bound = 10 ** 1000
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = j            
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    sum_of_rewards[ad] += dataset.values[i, ad]
    total_rewards += dataset.values[i, ad]
    
plt.hist(ads_selected)
plt.title("Ad Selection Histogram")
plt.xlabel("Ad Identifier")
plt.ylabel("Number of selections")
plt.show()