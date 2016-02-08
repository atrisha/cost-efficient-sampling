'''
Created on 2016-02-04

@author: Atri
'''

from SALib.sample import sobol_sequence
from com.ase.extn.cart import base
from com.ase.extn.constants import configs
import numpy as np


param_values = sobol_sequence.sample(10, 1)
param_values = np.append(param_values,1)
base.print_detail = False
sens_data = dict()
print("System-id : "+str(configs.system))
for values in param_values:
    if values <= 1 and values >=.1:
        configs.r = float(values)
        
        correlation_data = base.main()
        for keys in correlation_data:
            if correlation_data[keys]['selected'] is True:
                data_list = [keys,correlation_data[keys]['correlation'],correlation_data[keys]['optimal sample size'],correlation_data[keys]['accuracy'],
                             correlation_data[keys]['standard deviation'],correlation_data[keys]['total cost']]
                print(str(configs.r) + " - " + str(data_list))  
        sens_data[str(configs.r)] = data_list       
                    