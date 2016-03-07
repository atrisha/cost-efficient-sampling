'''
Created on 2016-02-16

@author: Atri
'''
import numpy as np
from com.ase.extn.constants import configs
import random

def get_random_distribution(n_obs,categories):
    distr = np.zeros((1,categories))
    if configs.chi_sq_with_random is True:
        for i in range(n_obs):
            j = random.randint(1,30*100100)
            np.random.seed(j)
            val = np.random.randint(0,high=categories)
            distr[0][val] = distr[0][val] + 1
    else:
        for i in range(categories):
            distr[0][i] = int(n_obs) // int(categories)
    return distr    

