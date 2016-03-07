'''
Created on 2016-02-13

@author: Atri
'''
import matplotlib.pyplot as plt
from com.ase.extn.constants import configs
import numpy as np

x_data,x_data_prog = [],[]
y_data,y_data_prog = [],[]
curr_system = ''
opt_size = 0
opt_accu = 0
opt_cost = 0
prog_data=[]
real_curve_pts = []


def plot_now():
    if configs.strategy == 'progressive':
        global y_data_prog,x_data_prog
        x = range(0,configs.details_map[curr_system][1] // configs.th) 
        if configs.plot_real_cost is True:
            plt.axis([0, configs.details_map[curr_system][1], min(y_data_prog)-100 ,max(y_data_prog)+100])
        else:
            plt.axis([0, configs.details_map[curr_system][1], min(y_data_prog) ,100])
        plt.xlabel('Sample Size')
        if configs.plot_real_cost is True:
            plt.ylabel('Cost')
        else:
            plt.ylabel('Accuracy')
        plt.plot(x_data_prog,y_data_prog,'ro-')
        if configs.plot_real_cost is True:
            plt.plot(opt_size,opt_cost,'bo')
            plt.annotate(s='n*:'+str(opt_size),xy=(opt_size,opt_cost))
        else:
            plt.plot(opt_size,opt_accu,'bo')
            plt.annotate(s='n*:'+str(opt_size),xy=(opt_size,opt_accu))
        plt.show()
    else:
        i = 0
        x = range(0,configs.details_map[curr_system][1]// configs.th) 
        
        f,sub_plots = plt.subplots(2, 5)
        for rows in sub_plots:
            for cols in rows:
                cols.axis([-10, configs.details_map[curr_system][1]// configs.th, 0 ,110])
                x_data,y_data = [],[]
                lambda_set_dict = prog_data[i][0]
                correlation_data = prog_data[i][1]
                for keys in lambda_set_dict:
                    x_data.append(keys)
                    err = float(lambda_set_dict[keys])
                    if err > 100:
                        y_data.append(0)
                    else:
                        y_data.append(100-err)
                for keys in correlation_data:
                    if correlation_data[keys]['accuracy'] is not None:
                        a = correlation_data[keys]['a']
                        b = correlation_data[keys]['b']
                        if keys == 'log':
                            if correlation_data[keys]['selected'] is True:
                                cols.plot(100-(a + (b * np.log(x))),'k')
                            else:
                                cols.plot(100-(a + (b * np.log(x))),'b')
                        elif keys == 'exp':
                            if correlation_data[keys]['selected'] is True:
                                cols.plot(100-(a * (np.power(b,x))),'k')
                            else:
                                cols.plot(100-(a * (np.power(b,x))),'g')
                        elif keys == 'weiss':
                            if correlation_data[keys]['selected'] is True:
                                cols.plot(100-(a + ((b*x)/(range(1,len(x)+1)))),'k')
                            else:
                                cols.plot(100-(a + ((b*x)/(range(1,len(x)+1)))),'c')
                        elif keys == 'power':
                            if correlation_data[keys]['selected'] is True:
                                cols.plot(100-(a * (np.power(x,b))),'k')
                            else:
                                cols.plot(100-(a * (np.power(x,b))),'m')
                cols.plot(x_data,y_data,'ro-')
                if configs.show_actual_lc is True:
                    if i < len(real_curve_pts):
                        cols.plot(real_curve_pts[i][0],real_curve_pts[i][1],'yo-')
                i=i+1
        plt.show()        
            