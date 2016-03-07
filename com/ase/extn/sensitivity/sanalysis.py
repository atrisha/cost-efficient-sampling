'''
Created on 2016-02-04

@author: Atri
'''

from SALib.sample import sobol_sequence
from com.ase.extn.cart import base
from com.ase.extn.tway import twaysample
from com.ase.extn.constants import configs
import numpy as np
import collections
import matplotlib.pyplot as plt

sensitivity = 'r'

if sensitivity is 'r':
    param_values_0_1 = sobol_sequence.sample(10, 1)
    param_values_1_10 = (1 + (9*param_values_0_1))
    param_values = np.append(np.append(param_values_0_1,param_values_1_10),1)
    param_values.sort()
else:
    param_values = [2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3]

base.print_detail = False


sens_data_ff = collections.OrderedDict()
sens_data_2w = collections.OrderedDict()
sens_data_3w = collections.OrderedDict()
sens_data_pg = collections.OrderedDict()
print("System-id : "+str(configs.system))

for values in param_values:
    if values <= 10 and values >=.1:
        print(values)
        if sensitivity is 'r':
            configs.r = float(values)
        else:
            configs.th = values
        size_result,success,p_value,cost_result,opt_size_result_ff = base.projective(configs.system)  
        sens_data_ff[values] = cost_result,success,opt_size_result_ff       
        configs.tway = 2
        size_result,success,p_value,cost_result,opt_size_result_2w = twaysample.sample(configs.system)  
        sens_data_2w[values] = cost_result,success,opt_size_result_2w
        configs.tway = 3
        size_result,success,p_value,cost_result,opt_size_result_3w = twaysample.sample(configs.system)  
        sens_data_3w[values] = cost_result,success,opt_size_result_3w
        configs.strategy = 'progressive'
        data_list,opt_cost,real_cost = base.progressive(configs.system) 
        sens_data_pg[values] = (opt_cost,real_cost)

x_data,y_ff,y_2w,y_3w,y_re = [],[],[],[],[]        
for key in sens_data_ff:
    x_data.append(float(key))
    y_ff.append(sens_data_ff[key][0][2])
    print('feature frequencies: ',key,'-',sens_data_ff[key])
    y_2w.append(sens_data_2w[key][0][2])
    print('2way: ',key,'-',sens_data_2w[key])
    y_3w.append(sens_data_3w[key][0][2])
    print('3way: ',key,'-',sens_data_3w[key])
    y_re.append(sens_data_pg[key][1])
    print('Progressive,Real: ',key,'-',sens_data_pg[key])
    print()
    
plt.plot(x_data,y_ff,'go-')
i=0
for x in x_data:
    if x>=1:
        plt.text(x, y_ff[i], str(int(sens_data_ff[x][2])))
    i=i+1
plt.plot(x_data,y_2w,'bo-')
i=0
for x in x_data:
    if x>=1:
        plt.text(x, y_2w[i], str(int(sens_data_2w[x][2])))
    i=i+1
plt.plot(x_data,y_3w,'mo-')
i=0
for x in x_data:
    if x>=1:
        plt.text(x, y_3w[i], str(int(sens_data_3w[x][2])))
    i=i+1
plt.plot(x_data,y_re,'ro-')    
plt.show()        