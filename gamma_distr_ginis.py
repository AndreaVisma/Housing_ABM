# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:39:16 2021

@author: Andrea Vismara
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%% 

def gini_from_gamma(kappa, theta):
    
    b = np.empty(10000000)
    A = 0
    B = 0
    C = 0
    
    for i in range(len(b)):
        b[i] = np.random.gamma(kappa, theta)
        
    b = np.sort(b) 
    c = np.linspace(0, max(b), num = len(b))
    plt.plot(b)
    plt.plot(c)
    plt.show()
    for i in range(len(b)):  
        C += 1 * c[i]
        B += 1 * b[i]
        
    A = C - B    
    Gini = A / (A + B)
    
    return Gini

#%%


list_kappa = np.linspace(1, 10, num = 10)
list_theta = np.linspace(1, 10, num = 10)
gini_array = np.empty((len(list_kappa), len(list_theta)))

#%%
for i in range(len(list_kappa)):
    for t in range(len(list_theta)):
        kappa = list_kappa[i]
        theta = list_theta[t]
        gini_array[i, t] = gini_from_gamma(kappa, theta)

#%%

df = pd.DataFrame(data = gini_array, index = list_kappa, columns = list_theta).round(decimals =2)
# df['kappa'] = df['kappa'].round(decimals = 2)
# df['theta'] = df['theta'].round(decimals = 2)

#%%

ax, fig = plt.subplots()
ax = sns.heatmap(df, annot=True, fmt="g")
ax.set_title('gini coefficient from different specifications of gamma distribution')
ax.set_xlabel('theta')
ax.set_ylabel('kappa')