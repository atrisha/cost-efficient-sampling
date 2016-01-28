'''
Created on 2016-01-26

@author: Atri
'''
import numpy as np

k = np.empty((3,4))
k[0][0] = 10
k[1][0] = 20
l = range(0,k.shape[0])
print(k[0])