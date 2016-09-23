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
    if configs.r_0_to_1 is True:
        param_values = np.append(param_values_0_1,1)
    else:
        param_values = np.append(param_values_1_10,1)
    param_values.sort()
else:
    param_values = 2 + sobol_sequence.sample(10, 1)
    param_values = np.append(param_values,3)
    param_values.sort()
    

base.print_detail = False

chart_type = 'cost'

sens_data_ff = collections.OrderedDict()
sens_data_2w = collections.OrderedDict()
sens_data_3w = collections.OrderedDict()
sens_data_pg = collections.OrderedDict()
print("System-id : "+str(configs.system))

plt.figure(figsize=(5,4.2))
plt.title(str(configs.system))
if sensitivity is 'r':
    plt.xlabel('Cost ratio (R)')
else:
    plt.xlabel('Training-testing split factor')
    
plt.ylabel('Total cost')

for values in param_values:
    if values <= 10 and values >=.1:
        print(values)
        if sensitivity is 'r':
            configs.r = float(values)
        else:
            if values==2 or values==3:
                configs.th = int(values)
            else:
                configs.th = float(values)
        ret = base.projective(configs.system)
        if ret is not None:
            size_result,success,p_value,cost_result,opt_size_result_ff = ret 
            sens_data_ff[values] = cost_result,success,opt_size_result_ff       
            
        configs.tway = 2
        ret = twaysample.sample(configs.system)
        if ret is not None:
            size_result,success,p_value,cost_result,opt_size_result_2w = ret
            sens_data_2w[values] = cost_result,success,opt_size_result_2w
            
        configs.tway = 3
        ret = twaysample.sample(configs.system)
        if ret is not None:
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

base_x = abs(max(x_data) - min(x_data))    
sl_2w = abs(y_2w[-1] - y_2w[0]) / base_x
sl_3w = abs(y_3w[-1] - y_3w[0]) / base_x
sl_re = abs(y_re[-1] - y_re[0]) / base_x
sl_ff = abs(y_ff[-1] - y_ff[0]) / base_x
print("********************** SLOPE (R :0 - 1): "+str(configs.system)+"*****************************")
print('2 way',sl_2w)
print('3 way',sl_3w)
print('ff',sl_ff)
print('Actual',sl_re)
err_2w,err_3w,err_ff = [],[],[]
scale = abs(max(y_2w + y_3w + y_ff + y_re) - min(y_2w + y_3w + y_ff + y_re))
err_2w_m,err_2w_v = np.mean(np.absolute(np.subtract(y_2w,y_re))),np.std(np.absolute(np.subtract(y_2w,y_re)))
err_3w_m,err_3w_v = np.mean(np.absolute(np.subtract(y_3w,y_re))),np.std(np.absolute(np.subtract(y_3w,y_re)))
err_ff_m,err_ff_v = np.mean(np.absolute(np.subtract(y_ff,y_re))),np.std(np.absolute(np.subtract(y_ff,y_re)))
print("********************** Erro MEAN,VARIANCE (R :0 - 1): "+str(configs.system)+"*****************************")
print('2 way',err_2w_m/scale,u"\u00B1",err_2w_v/scale)
print('3 way',err_3w_m/scale,u"\u00B1",err_3w_v/scale)
print('ff',err_ff_m/scale,u"\u00B1",err_ff_v/scale)

plt.plot(x_data,y_ff,'go-')
i=0
for x in x_data:
    '''plt.text(x, y_ff[i], str(int(sens_data_ff[x][2])))'''
    i=i+1
plt.plot(x_data,y_2w,'bo-')
i=0
for x in x_data:
    '''plt.text(x, y_2w[i], str(int(sens_data_2w[x][2])))'''
    i=i+1
plt.plot(x_data,y_3w,'mo-')
i=0
for x in x_data:
    '''plt.text(x, y_3w[i], str(int(sens_data_3w[x][2])))'''
    i=i+1
plt.plot(x_data,y_re,'ro-')    
plt.show()       

plt.clf()


plt.figure(figsize=(5,4.2))
plt.title(str(configs.system))

if sensitivity is 'r':
    plt.xlabel('Cost ratio (R)')
else:
    plt.xlabel('Training-testing split factor')
plt.ylabel('Accuracy')
x_data,y_ff,y_2w,y_3w,y_re = [],[],[],[],[]   
for key in sens_data_ff:
    x_data.append(float(key))
    y_ff.append(sens_data_ff[key][1])
    print('feature frequencies: ',key,'-',sens_data_ff[key])
    y_2w.append(sens_data_2w[key][1])
    print('2way: ',key,'-',sens_data_2w[key])
    y_3w.append(sens_data_3w[key][1])
    print('3way: ',key,'-',sens_data_3w[key])
    print()
    
plt.plot(x_data,y_ff,'go-')
i=0
for x in x_data:
    plt.text(x, y_ff[i], str(int(sens_data_ff[x][2])))
    i=i+1
plt.plot(x_data,y_2w,'bo-')
i=0
for x in x_data:
    plt.text(x, y_2w[i], str(int(sens_data_2w[x][2])))
    i=i+1
plt.plot(x_data,y_3w,'mo-')
i=0
for x in x_data:
    plt.text(x, y_3w[i], str(int(sens_data_3w[x][2])))
    i=i+1   
plt.show() 