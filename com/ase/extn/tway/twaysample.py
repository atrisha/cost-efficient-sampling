'''
Created on 2016-02-11

@author: Atri
'''
import os
import numpy as np
import scipy.stats as sp
from com.ase.extn.constants import configs
from com.ase.extn.plots import plot
from sklearn import tree
from com.ase.extn.cart import base
import random


print_detail = configs.print_detail

def load_data(train):
    base_dir = configs.base_dir
    base_dir_in = configs.base_dir_tway_in
    base_dir_out = configs.base_dir_out
    system = configs.system
    file_name_train = system+'_'+str(configs.tway)+'_way_perf_train'
    file_name_test = system+'_'+str(configs.tway)+'_way_perf_test'
    fname = os.path.join(base_dir_in,file_name_train) if train is True else os.path.join(base_dir_in,file_name_test)
    num_features = range(0,configs.details_map[system][0])
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(str)
    return data

def load_perf_values(train):
    base_dir = configs.base_dir
    base_dir_in = configs.base_dir_tway_in
    base_dir_out = configs.base_dir_out
    system = configs.system
    file_name_train = system+'_'+str(configs.tway)+'_way_perf_train'
    file_name_test = system+'_'+str(configs.tway)+'_way_perf_test'
    fname = os.path.join(base_dir_in,file_name_train) if train is True else os.path.join(base_dir_in,file_name_test)
    data = np.loadtxt(fname,  delimiter=',', dtype=float,skiprows=1,usecols=(configs.details_map[system][0],))
    return data

def get_projected_accuracy(optimal_size,data_train,perf_values_train,data_test,perf_values_test,test_set_indices):
    results = np.empty((1,configs.repeat))
    for j in range(configs.repeat):
        np.random.seed(j)
        if optimal_size > data_train.shape[0]:
            if configs.fix_test_set is True:
                train_opt_indices = set(range(data_test.shape[0])) - set(test_set_indices)
                training_set_indices = np.random.choice(np.array(list(train_opt_indices)),(optimal_size-data_train.shape[0]),replace=False)
            else:
                training_set_indices = np.random.choice(data_test.shape[0],(optimal_size-data_train.shape[0]),replace=False)
                
            diff_indices = set(range(data_test.shape[0])) - set(training_set_indices)
            temp = data_test[training_set_indices]
            training_set = np.append(temp,data_train,0)
            
            if configs.fix_test_set is True:
                test_set_indices = test_set_indices
            else:
                test_set_indices = np.random.choice(np.array(list(diff_indices)),optimal_size,replace=False)
            test_set = data_test[test_set_indices]
            y = np.append(perf_values_test[training_set_indices],perf_values_train)
            
        else:
            training_set_indices = np.random.choice(data_train.shape[0],optimal_size,replace=False)
            training_set = data_train[training_set_indices]
            if configs.fix_test_set is True:
                test_set_indices = test_set_indices
            else:
                test_set_indices = np.random.choice(data_test.shape[0],optimal_size,replace=False)
            test_set = data_test[test_set_indices]
            y = perf_values_train[training_set_indices]
            
        X = training_set
        built_tree = base.cart(X, y)
        out = base.predict(built_tree, test_set, perf_values_test[test_set_indices])
        accu = base.calc_accuracy(out,perf_values_test[test_set_indices])
        if accu <= 100:
            results[0][j] = 100 - accu 
         
    mean = results.mean()
    sd = np.std(results)
    return (mean,sd)

def sample(system):
    configs.extend_lambda = False
    data_train = load_data(True)
    perf_values_train = load_perf_values(True)
    data_test = load_data(False)
    perf_values_test = load_perf_values(False)
    
    data_train[data_train == 'Y'] = 1
    data_train[data_train == 'N'] = 0
    data_train = data_train.astype(bool)    
    
    data_test[data_test == 'Y'] = 1
    data_test[data_test == 'N'] = 0
    data_test = data_test.astype(bool)
    
    repeat = configs.repeat
    if print_detail is True:
        print('Size of '+str(system)+' '+str(configs.tway)+'-way sample is: '+str(data_train.shape[0]))
    corr_list = []
    
    for s in range(repeat):
        if print_detail is True:
            print('Iteration',s)
        results = dict()
        j = random.randint(1,30*100100)
        if configs.fix_test_set is True:
            test_set_indices = np.random.choice(data_test.shape[0],configs.details_map[system][1] // configs.fix_test_ratio,replace=False)
        i=0
        while True:
            if i==data_train.shape[0]:
                break
            else:
                i=i+1
            curr_size = i
            np.random.seed(j)
            training_set_indices = np.random.choice(data_train.shape[0],curr_size,replace=False)
            training_set = data_train[training_set_indices]
            
            if configs.fix_test_set is True:
                test_set_indices = test_set_indices
            else:    
                test_set_indices = np.random.choice(data_test.shape[0],curr_size,replace=False)
            test_set = data_test[test_set_indices]
            
            X = training_set
            y = perf_values_train[training_set_indices]
            
            built_tree = base.cart(X, y)
            out = base.predict(built_tree, test_set, perf_values_test[test_set_indices])
            
            if curr_size in results:
                print('%%%%%%%%%%%%%%%%%%%% SHOCK!! &&&&&&&&&&&&&&&&&&&')
            else:
                accu = base.calc_accuracy(out,perf_values_test[test_set_indices])
                if accu <= 100:
                    results[curr_size] = accu
        result_in_cluster = base.check_result_cluster(results)        
        if configs.add_origin_to_lambda is True and result_in_cluster is True:
            results[0] = 100
        if configs.transform_lambda is True:
            results = base.transform_lambda_set(results)
        if print_detail is True:    
            print('Size of lambda set: '+ str(len(results)))    
        '''
        Transform the axes and calculate pearson correlation with
        each learning curve
        '''
        curve_data = base.transform_axes(base.smooth(base.dict_to_array(results)))
        parameter_dict = dict()
        correlation_data = dict()
        ''' keys here are individual curves for a given system. Values are 2-d array. x: transformed "no. of sample" values
        and y: transformed accuracy at that sample value'''
        for keys in curve_data:
            slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][configs.ignore_initial:,0],curve_data[keys][configs.ignore_initial:,1])
            if print_detail is True:
                print(keys,intercept,slope)
            value_a = base.get_intercept(intercept,keys)
            value_b = base.get_slope(slope,keys)
            parameter_dict[keys] = {'a' : value_a, 'b':value_b}
            value_r = configs.r
            value_s = configs.details_map[system][1]/3
            optimal_size = base.get_optimal(value_a,value_b,value_r,value_s,keys)
            estimated_error = 100
            weiss_within_range = True
            if keys == 'weiss' and (abs(value_a) + abs(value_b)) > 100:
                weiss_within_range = False
            if optimal_size <= (data_train.shape[0]+data_test.shape[0])//configs.th and optimal_size > 1 and weiss_within_range is True:
                mean_accu,sd = get_projected_accuracy(optimal_size,data_train,perf_values_train,data_test,perf_values_test,test_set_indices)
                r = configs.r
                th = configs.th
                total_cost = base.cost_eqn(th,optimal_size, 100-float(mean_accu), configs.details_map[system][1] // 3, r)
                estimated_error = base.get_error_from_curve(value_a,value_b,optimal_size,keys)
                estimated_cost = base.cost_eqn(th,optimal_size,estimated_error,configs.details_map[system][1] // 3, r)
            else:
                mean_accu,sd,total_cost,estimated_cost = (None,None,None,None)
            
            correlation_data[keys] = {'correlation' : rvalue,
                                      'p-value' : str(pvalue),
                                      'optimal sample size' :optimal_size,
                                      'accuracy' :mean_accu,
                                      'estimated accuracy': 100 - estimated_error,
                                      'standard deviation' :sd,
                                      'total cost' :total_cost,
                                      'estimated cost' : estimated_cost,
                                      'a' : value_a,
                                      'b' : value_b,
                                      'lambda size' : len(results)}
        selected_curve = base.select_curve(correlation_data)
        
        if print_detail is True:
            print()
            print('Detailed learning projections:')
            print('<curve-id> : {<details>}')
            print()
            
        for keys in correlation_data:
            if keys in selected_curve:
                correlation_data[keys]['selected'] = True
                if print_detail is True:
                    print(str(keys) +"**:"+str(correlation_data[keys]))
            else:
                correlation_data[keys]['selected'] = False
                if print_detail is True:
                    print(str(keys) +":"+str(correlation_data[keys]))
        if print_detail is True:            
            print("-----------------------------------------------")
            print()
        corr_list.append(correlation_data)
        if configs.plot is True and configs.sense_curve is True:
            plot.curr_system = system
            plot.prog_data.append((results,correlation_data))
        
    if configs.plot is True and configs.sense_curve is True:
        plot.plot_now()
        return base.mean_corr_list(corr_list)   
    else:
        return base.mean_corr_list(corr_list) 

                
'''sample(configs.system)'''