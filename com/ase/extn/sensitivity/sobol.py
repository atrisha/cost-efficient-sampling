'''
Created on Mar 10, 2016

@author: Atri
'''
from SALib.sample import sobol_sequence
import matplotlib.pyplot as plt
import numpy as np





x = range(0,1)
plt.axis([0, 1, 0,1])
plt.subplot(1,3,1)
param_values_0_1 = sobol_sequence.sample(100, 2)
for dots in param_values_0_1:
    plt.plot(dots[0],dots[1],'ro')
    
plt.subplot(1,3,2)
param_values_0_1 = sobol_sequence.sample(1000, 2)
for dots in param_values_0_1:
    plt.plot(dots[0],dots[1],'ro')
    
plt.subplot(1,3,3)
param_values_0_1 = sobol_sequence.sample(2000, 2)
for dots in param_values_0_1:
    plt.plot(dots[0],dots[1],'ro')        

'''
plt.axis([0, 1, 0,1])
plt.subplot(1,3,1)
param_values_0_1 = np.random.rand(100,2)
for dots in param_values_0_1:
    plt.plot(dots[0],dots[1],'ro')

plt.subplot(1,3,2)
param_values_0_1 = np.random.rand(1000,2)
for dots in param_values_0_1:
    plt.plot(dots[0],dots[1],'ro')
    
plt.subplot(1,3,3)
param_values_0_1 = np.random.rand(2000,2)
for dots in param_values_0_1:
    plt.plot(dots[0],dots[1],'ro')     
'''      
plt.show()