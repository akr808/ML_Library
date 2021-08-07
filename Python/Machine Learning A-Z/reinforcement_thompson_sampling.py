# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Machine Learning A-Z (Codes and Datasets)"
Topic: Reinforcement Learning--Thompson Sampling Algorithm.
@author: Anand
"""

import pandas as pd
import matplotlib.pyplot as plt

import random

#importing the dataset
dataset = pd.read_csv("../Data/Ads_CTR_Optimisation.csv")
N = 10000
d = 10
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_reward = 0

for n in range(N):
    ad = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1 )
        if max_random < random_beta:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    total_reward += dataset.values[n,ad]   
    if dataset.values[n][ad]  == 1:
        number_of_rewards_1[ad] += dataset.values[n,ad] 
    else:
        number_of_rewards_0[ad] += dataset.values[n,ad] 
plt.hist(ads_selected)
plt.title("Ad Selection Histogram")
plt.xlabel("Ad Identifier")
plt.ylabel("Number of selections")
plt.show()
