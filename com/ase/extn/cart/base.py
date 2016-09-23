'''
Created on 2016-01-23

@author: Atri
'''
import sys
import numpy as np
import scipy.stats as sp
import os
import time
import collections
import random
import math as math
from sklearn import tree
from sklearn.svm import SVR
from numpy import mean, dtype
from com.ase.extn.constants import configs
from pygments.lexers.inferno import LimboLexer
from com.ase.extn.plots import plot
from scipy.stats import norm
from scipy.stats import chisquare
from com.ase.extn.utils import commonutils

'''
Set 
strategy = projective|progressive
system = all|apache|bc|bj|llvm|sqlite|x264
'''
strategy = configs.strategy
system = configs.system

thismodule = sys.modules[__name__]
loc = configs.loc

base_dir = configs.base_dir
base_dir_in = configs.base_dir_in
base_dir_out = configs.base_dir_out

all_systems = configs.all_systems
print_detail = configs.print_detail
dynamic_override = False

'''
details_map holds the following data-
details_map = {<system-id> :[<no_of_features>,<size_of_sample_space>]}
'''
details_map = configs.details_map


def get_min_params(training_set_size):
    if training_set_size > 100:
        min_split = math.floor((training_set_size/100) + 0.5)
        min_bucket = math.floor(min_split/2)
    else:
        min_bucket = math.floor((training_set_size/10) + 0.5)
        min_split = 2 * min_bucket
    
    min_bucket=2 if min_bucket < 2 else min_bucket
    min_split=4 if min_split < 4 else min_split
    return [min_bucket,min_split]
        
   
def load_data():
    fname = os.path.join(base_dir_in,system)
    num_features = range(0,details_map[system][0])
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(str)
    return data

def load_perf_values():
    fname = os.path.join(base_dir_in,system)
    data = np.loadtxt(fname,  delimiter=',', dtype=float,skiprows=1,usecols=(details_map[system][0],))
    return data

def load_feature_names():
    fname = os.path.join(base_dir_in,system)
    f = open(fname).readline().rstrip('\n').split(',',details_map[system][0])
    return f[:len(f)-1]
    
def cart(X,y):
    training_set_size = X.shape[0]
    params = get_min_params(training_set_size)
    clf = tree.DecisionTreeRegressor(min_samples_leaf=params[0],min_samples_split=params[1])
    clf = clf.fit(X, y)
    return clf

def predict(clf,test_set,values):
    out = clf.predict(test_set) 
    return out

def calc_accuracy(pred_values,actual_values):
    return mean((abs(pred_values - actual_values)/actual_values)*100)

def all_true(in_list):
    for i in in_list:
        if not i:
            return False
    return True  

def progressive(system_val):
    global system
    system = system_val    
    if configs.strategy == 'progressive':
        configs.show_actual_lc = False
    
    if configs.plot is True or configs.plot_real_cost is True:
        plot.curr_system = system_val
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)
    repeat = configs.repeat
    if configs.th == 2 or configs.th == 3: 
        total_range = range((details_map[system][1]//10)//configs.th)
    else:
        total_range = range(int((details_map[system][1]//10)/configs.th))
    results = np.empty((len(total_range),repeat))
    data_list = []
    for j in range(repeat):
        for i in total_range:
            np.random.seed(j)
            if configs.fix_test_set is True:
                test_set_indices = np.random.choice(data.shape[0],details_map[system][1] // configs.fix_test_ratio,replace=False)
            curr_size = 10*(i+1)
            if configs.fix_test_set is True:
                train_opt_indices = set(range(data.shape[0])) - set(test_set_indices)
                training_set_indices = np.random.choice(np.array(list(train_opt_indices)),curr_size,replace=False)
            else:
                training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
                
            diff_indices = set(range(data.shape[0])) - set(training_set_indices)
            training_set = data[training_set_indices]
            if configs.fix_test_set is True:
                test_set_indices = test_set_indices
            else:
                test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
            test_set = data[test_set_indices]
            X = training_set
            y = perf_values[training_set_indices]
            if configs.model is 'cart':
                built_tree = cart(X, y)
                out = predict(built_tree, test_set, perf_values[test_set_indices])
            else:
                clf = SVR(C=1.0, epsilon=0.2)
                clf.fit(X, y)
                out = predict(clf, test_set, perf_values[test_set_indices])
            results[i][j] = calc_accuracy(out,perf_values[test_set_indices])
        if print_detail is True:    
            print('['+system+']' + " iteration :"+str(j+1))
    print()
    out_file = open(os.path.join(base_dir_out,system)+"_out_"+strategy+"_"+str(configs.model),'w')
    out_file.truncate()
    cost_prev = sys.maxsize
    size_prev = 0
    acc_prev = 0
    opt_cost = 0
    cost_list = []
    if configs.show_actual_lc is True:
        local_xdata = []
        local_ydata = []
    opt_found = False
    for i in range(results.shape[0]):
        size = (i+1)*10
        error = mean(results[i])
        out_file.write(str(size)+","+ str(error))
        out_file.write('\n')
        if configs.plot is True or configs.plot_real_cost is True:
            plot.x_data_prog.append(size)
            if configs.show_actual_lc is True:
                local_xdata.append(size)
            if configs.plot_real_cost is False:
                if error > 100:
                    plot.y_data_prog.append(100-100)
                    if configs.show_actual_lc is True:
                        local_ydata.append(100-100)
                else:
                    plot.y_data_prog.append(100-error)
                    if configs.show_actual_lc is True:
                        local_ydata.append(100-error)
                        
        if configs.calc_prog_opt is True:
            R = configs.r
            S = configs.details_map[system][1]//3    
            cost_curr = cost_eqn(configs.th,size,error,S,R)
            cost_list.append(cost_curr)
            if configs.plot_real_cost is True:
                plot.y_data_prog.append(cost_curr)
            if cost_curr > cost_prev and opt_found is False:
                plot.opt_size = size_prev
                plot.opt_accu = acc_prev
                plot.opt_cost = cost_prev
                opt_cost = cost_prev
                opt_found = True
            else:
                cost_prev = cost_curr
                size_prev = size
                acc_prev = 100-error    
        if configs.show_actual_lc is True:
            data_list.append((local_xdata,local_ydata))        
    real_cost = min(cost_list)
    plot.real_min_cost = real_cost
    if configs.print_detail is True:
        print('Accuracy at optimal:',acc_prev)
    if configs.plot is True and configs.strategy == 'progressive':
        plot.plot_now()   
    return data_list,opt_cost,real_cost
            
        
def transform_axes(results):
    curve_data = dict()
    original = np.copy(results)
    if len(original) > 0:
        results[:,0] = np.log(original[:,0])
        results[:,1] = original[:,1]
        curve_data['log'] = np.copy(results)
        
        results[:,0] = original[:,0]/(original[:,0]+1)
        results[:,1] = original[:,1]
        curve_data['weiss'] = np.copy(results)
        
        results[:,0] = original[:,0]
        results[:,1] = np.log(original[:,1])
        curve_data['exp'] = np.copy(results)
        
        results[:,0] = np.log(original[:,0])
        results[:,1] = np.log(original[:,1])
        curve_data['power'] = np.copy(results)
    return curve_data

def dict_to_array(dict_struct):
    dictlist = []
    for key, value in dict_struct.items():
        if isinstance(value, collections.Iterable):
            value = value
        dictlist.append([key,value])
    return np.array(dictlist)

def smooth(result_array):
    if len(result_array) > 0:
        fault_rates = result_array[:,1]
        for i in range(1, len(fault_rates)-1):
            fault_rates[i] = (fault_rates[i-1] + fault_rates[i] + fault_rates[i+1])/3    
        result_array[:,1] = fault_rates
        return result_array
    else:
        return result_array

def get_projected_accuracy(size,repeat,data,perf_values,test_ind_in):
    results = np.empty((1,repeat))
    for j in range(repeat):
        np.random.seed(j)
        if configs.fix_test_set is True:
            train_opt_indices = set(range(data.shape[0])) - set(test_ind_in)
            training_set_indices = np.random.choice(np.array(list(train_opt_indices)),size,replace=False)
        else:
            training_set_indices = np.random.choice(data.shape[0],size,replace=False)
        diff_indices = set(range(data.shape[0])) - set(training_set_indices)
        training_set = data[training_set_indices]
        if configs.fix_test_set is True:
            test_set_indices = test_ind_in
        else:
            test_set_indices = np.random.choice(np.array(list(diff_indices)),size,replace=False)
        test_set = data[test_set_indices]
        
        X = training_set
        y = perf_values[training_set_indices]
        if configs.model is 'cart':
            built_tree = cart(X, y)
            out = predict(built_tree, test_set, perf_values[test_set_indices])
        else:
            clf = SVR(C=1.0, epsilon=0.2)
            clf.fit(X, y)
            out = predict(clf, test_set, perf_values[test_set_indices])
        accu = calc_accuracy(out,perf_values[test_set_indices])
        if accu <=100:
            results[0][j] = 100 - accu
    mean = results.mean()
    sd = np.std(results)
    return (mean,sd)
        
def get_optimal(a,b,r,s,curve):
    if curve=='log':
        n = -(r*s*b)/configs.th
    elif curve=='weiss':
        if b > 0:
            n = -1
        else:
            n = np.power(((-r*s*b)/configs.th),0.5)
    elif curve=='power':
        if b < 0:
            n = np.power((-configs.th/(r*s*a*b)),(1/(b-1)))
        else:
            n = -1
    elif curve=='exp':
        if b > 1:
            n = -1
        else :
            n = math.log((-configs.th/(r*s*(a*(np.log(b))))),b)    
    return n

def get_error_from_curve(a,b,n,curve):
    if curve == 'log':
        return a + (b*np.log(n))
    elif curve == 'weiss':
        return a + (b*n /(n+1))
    elif curve == 'power':
        return a*(np.power(n,b))
    elif curve == 'exp':
        return a*(np.power(b,n))

def cost_eqn(th,n,e,s,r):
    return (th*n + (e*r*s))

def get_intercept(intercept,curve):
    if curve=='power' or curve=='exp':
        return np.exp(intercept)
    else:
        return intercept

def get_slope(slope,curve):
    if curve=='exp':
        return np.exp(slope)
    else:
        return slope

def get_next_size(curve,array,curve_array,index):
    i = curve_array.index(curve)
    s = array[i]
    temp_list = []
    for v in array:
        if v<s and v>0:
            temp_list.append(v)
    if len(temp_list) == 0:
        return 0
    else:
        return max(temp_list)
    
        

def select_curve_dynamic(correlation_data,data,perf_values,parameter_dict,results,test_set_indices):
    ''' First, we transform the correlation_data structure from key: {a: a1, b:b1...} format to an array where
    keys(curves) are individual rows and columns are detailed data of that curve'''
    trans_array = np.empty([len(correlation_data),len(next(iter(correlation_data.values())))])
    curve_array = []
    index = dict()
    i=0
    for values in next(iter(correlation_data.values())):
        index[values] = i
        i = i+1
    i=0
    for keys in correlation_data:
        value_dict = correlation_data[keys]
        curve_array.append(keys)
        for values in value_dict:
            trans_array[i][index[values]] = value_dict[values]
        i=i+1
    
    lambda_size = len(results)
    ''' Select the current chosen curve'''
    curve = select_curve(correlation_data)
    if curve is not None:
        size_to_sample = get_next_size(curve[0],trans_array[:,index['optimal sample size']],curve_array,index)
        if print_detail is True:
            print('Updated size : '+ str(size_to_sample))
    else:
        size_to_sample = len(results)
    if size_to_sample > lambda_size:
        lims = [len(results),size_to_sample]
        added_results = build_data_points(results,1, data, perf_values, False,lims,test_set_indices)
        results = None
        results = added_results
        if configs.smooth is True:
            curve_data = transform_axes(smooth(dict_to_array(results)))
        else:
            curve_data = transform_axes(dict_to_array(results))
        correlation_data = dict()
        ''' keys here are individual curves for a given system. Values are 2-d array. x: transformed "no. of sample" values
        and y: transformed accuracy at that sample value'''
        for keys in curve_data:
            slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][configs.ignore_initial:,0],curve_data[keys][configs.ignore_initial:,1])
            value_a = get_intercept(intercept,keys)
            value_b = get_slope(slope,keys)
            value_r = configs.r
            value_s = details_map[system][1]/3
            optimal_size = get_optimal(value_a,value_b,value_r,value_s,keys)
            if optimal_size <= data.shape[0]//configs.th and optimal_size > 1:
                mean_accu,sd = get_projected_accuracy(optimal_size,configs.repeat,data,perf_values,test_set_indices)
                proj_err = get_error_from_curve(parameter_dict[keys]['a'], parameter_dict[keys]['b'], optimal_size, keys)
                diff = abs((100-mean_accu) - proj_err)
                total_cost = cost_eqn(configs.th,optimal_size, 100-float(mean_accu), details_map[system][1] // 3, configs.r)
                estimated_error = get_error_from_curve(value_a,value_b,optimal_size,keys)
                estimated_cost = cost_eqn(configs.th,optimal_size,estimated_error,details_map[system][1] // 3, configs.r)
                correlation_data[keys] = {'correlation' : rvalue,
                                      'p-value' : str(pvalue),
                                      'optimal sample size' :optimal_size,
                                      'accuracy' :mean_accu,
                                      'estimated accuracy': 100-estimated_error,
                                      'standard deviation' :sd,
                                      'total cost' :total_cost,
                                      'estimated cost' : estimated_cost,
                                      'diff' : diff,
                                      'a' : value_a,
                                      'b' : value_b,
                                      'lambda size' : len(results)}
                
            else:
                mean_accu,sd,total_cost = (None,None,None)
            
        if configs.dynamic_recursive_curve_selection is True:    
            selected_curve = select_curve_dynamic(correlation_data, data, perf_values, parameter_dict, results, test_set_indices)
        else:
            selected_curve = select_curve(correlation_data)[0]
        '''selected_curve_2 = select_curve_diff_error(correlation_data,data,perf_values,parameter_dict,results)'''
        if print_detail is True:
            print('corr: ' + str(selected_curve))
            print('original: ' +str(curve))
        return selected_curve,added_results
    else:
        return curve,results
    
def select_curve(correlation_data):
    curve = []
    min_corr = configs.min_corr
    for keys in correlation_data:
        if float(correlation_data[keys]['correlation']) < min_corr and correlation_data[keys]['accuracy'] is not None:
            min_corr = float(correlation_data[keys]['correlation'])
    for keys in correlation_data:
        if float(correlation_data[keys]['correlation']) == min_corr and min_corr!=0:
            if configs.curve_selection is not 'composite' and len(curve) == 0:
                curve.append(keys)
            elif configs.curve_selection is 'composite':
                curve.append(keys)
    return curve    

def select_curve_composite(correlation_data):
    curves = []
    min_corr = configs.min_corr
    for keys in correlation_data:
        if float(correlation_data[keys]['correlation']) < min_corr and correlation_data[keys]['accuracy'] is not None:
            min_corr = float(correlation_data[keys]['correlation'])
    for keys in correlation_data:
        if float(correlation_data[keys]['correlation']) == min_corr or abs(float(correlation_data[keys]['correlation']) - min_corr) < 0.05:
            curves.append(keys)
    return curves 

def select_curve_diff_error(correlation_data,data,perf_values,parameter_dict,results):
    curve = ''
    min_diff = 100
    for keys in correlation_data:
        if float(correlation_data[keys]['diff']) < min_diff and correlation_data[keys]['diff'] is not None:
            min_diff = float(correlation_data[keys]['diff'])
    for keys in correlation_data:
        if float(correlation_data[keys]['diff']) == min_diff:
            curve = keys
    return curve    

def extend_lambda_set(results):
    extension_end = configs.details_map[system][1] // configs.th
    entension_start = len(results)+1
    accuracy = results[len(results)-1]
    for ex in range(entension_start,extension_end):
        results[ex] = accuracy
    return results

def transform_lambda_set(results):
    error_list = list(results.values())
    error_list.sort(key=None, reverse=True)
    i=0
    transformed_results = results.copy()
    for keys in results:
        transformed_results[keys] = error_list[i]
        i=i+1
    return transformed_results
    

def build_data_points(results,repeat,data,perf_values,stop_by_freq,lims,test_ind_in):
    '''
    Initialise frequency table values to a 'ridiculous' number for all mandatory features
    if data is all true for a feature <=> mandatory <=> frequency table value = sys.maxsize 
    (These kind of hacks causes funny bugs!!)
    '''
    freq_table = np.empty([2,details_map[system][0]])
    for k in range(details_map[system][0]):
        if all_true(data[:,k]==1):
            freq_table[:,k]=sys.maxsize
    
    
    
    if lims is not None:
        i = lims[0]
    else:
        i = 0
    j = random.randint(1,30*100100)
    
    while True:
        '''print('['+system+']'+' running size :'+ str(i+1))'''
        curr_size = (i+1)
        np.random.seed(j)
        if configs.fix_test_set is True:
            train_opt_indices = set(range(data.shape[0])) - set(test_ind_in)
            training_set_indices = np.random.choice(np.array(list(train_opt_indices)),curr_size,replace=False)
        else:
            training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
        diff_indices = set(range(data.shape[0])) - set(training_set_indices)
        training_set = data[training_set_indices]
        if configs.fix_test_set is True:
            test_set_indices = test_ind_in
        else:
            test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
        test_set = data[test_set_indices]
        X = training_set
        y = perf_values[training_set_indices]
        if configs.model is 'cart':
            built_tree = cart(X, y)
            out = predict(built_tree, test_set, perf_values[test_set_indices])
        else:
            clf = SVR(C=1.0, epsilon=0.2)
            clf.fit(X, y)
            out = predict(clf, test_set, perf_values[test_set_indices])
        if curr_size in results:
            '''results[curr_size].append(calc_accuracy(out,perf_values[test_set_indices]))'''
            print('%%%%%%%%%%%%%%%%%%%% SHOCK!! &&&&&&&&&&&&&&&&&&&')
        else:
            accu = calc_accuracy(out,perf_values[test_set_indices])
            if accu <= 100:
                results[curr_size] = accu 
        
        if stop_by_freq is True:
            '''
            Update frequency table based on training set feature activation/de-activation
            We are refreshing the values in each iteration instead of making it incremental.
            This is in-efficient but keeps thing simple.
            '''
            for k in range(details_map[system][0]):
                if not freq_table[0][k]==sys.maxsize:
                    active_count = np.count_nonzero(training_set[:,k])
                    deactive_count = training_set.shape[0] - active_count
                    freq_table[0][k] = active_count
                    freq_table[1][k] = deactive_count
                else:
                    continue
                    
            '''
            We are done if the frequency table values hits the threshold
            '''
            if np.all(freq_table>=configs.projective_feature_threshold):
                break
            i=i+1
        else:
            i=i+1
            if i > lims[1]:
                break
    result_in_cluster = check_result_cluster(results)
    if configs.add_origin_to_lambda is True and result_in_cluster is True:
        results[0] = 100
    if configs.transform_lambda is True:
        transformed_results = transform_lambda_set(results)
        if configs.extend_lambda is True:
            extended_lambda = extend_lambda_set(transformed_results)
            return extended_lambda
        return transformed_results
    else:
        return results

def check_result_cluster(results):
    if len(results) > 0:
        min_error = min(results.values())
        max_error = max(results.values())
        if abs(max_error - min_error) < 20 and min_error < 70:
            return True
        else:
            return False
    else:
        return False
                    
def projective(system_val):
    if print_detail is True:
        print('System-id : '+system_val)
        print('R value : '+str(configs.r))
        print('th value : '+str(configs.th))
    global system
    if configs.plot is not True:
        configs.show_actual_lc = False
    system = system_val
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)    
    repeat = configs.repeat
    corr_list = []
    if configs.plot is True and configs.show_actual_lc is True:
        real_curve_points = progressive(system_val)
        plot.real_curve_pts = real_curve_points[0]
    for s in range(repeat):
        if print_detail is True:
            print('Running iteration :' +str(s))
        if configs.fix_test_set is True:
            test_set_indices = np.random.choice(data.shape[0],details_map[system][1] // configs.fix_test_ratio,replace=False)
        else:
            test_set_indices = []
        results = dict()
        results = build_data_points(results,repeat, data, perf_values, True,None,test_set_indices)
        
        if print_detail is True:
            print('Size of lambda set: '+ str(len(results)))    
        '''
        Transform the axes and calculate pearson correlation with
        each learning curve
        '''
        if configs.smooth is True:
            curve_data = transform_axes(smooth(dict_to_array(results)))
        else:
            curve_data = transform_axes(dict_to_array(results))
        parameter_dict = dict()
        correlation_data = dict()
        ''' keys here are individual curves for a given system. Values are 2-d array. x: transformed "no. of sample" values
        and y: transformed accuracy at that sample value'''
        for keys in curve_data:
            
            slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][configs.ignore_initial:,0],curve_data[keys][configs.ignore_initial:,1])
            value_a = get_intercept(intercept,keys)
            value_b = get_slope(slope,keys)
            parameter_dict[keys] = {'a' : value_a, 'b':value_b}
            value_r = configs.r
            value_s = details_map[system][1]/3
            optimal_size = get_optimal(value_a,value_b,value_r,value_s,keys)
            estimated_error = 100
            weiss_within_range = True
            if keys == 'weiss' and (abs(value_a) + abs(value_b)) > 100:
                weiss_within_range = False
            if optimal_size <= data.shape[0]//configs.th and optimal_size > 1 and weiss_within_range is True:
                mean_accu,sd = get_projected_accuracy(optimal_size,repeat,data,perf_values,test_set_indices)
                r = configs.r
                th = configs.th
                total_cost = cost_eqn(th,optimal_size, 100-float(mean_accu), details_map[system][1] // 3, r)
                estimated_error = get_error_from_curve(value_a,value_b,optimal_size,keys)
                estimated_cost = cost_eqn(th,optimal_size,estimated_error,details_map[system][1] // 3, r)
            else:
                mean_accu,sd,total_cost,estimated_cost,optimal_size = (None,None,None,None,None)
            
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
                                      'stderr' : stderr,
                                      'lambda size' : len(results)}
            
        if configs.curve_selection == 'dynamic':
            selected_curve,results = select_curve_dynamic(correlation_data,data,perf_values,parameter_dict,results,test_set_indices)
        elif configs.curve_selection == 'static':
            selected_curve = select_curve(correlation_data)
        else:
            selected_curve = select_curve_composite(correlation_data)
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
                if print_detail is True and float(correlation_data[keys]['correlation']) < configs.min_corr:
                    print(str(keys) +":"+str(correlation_data[keys]))
        if print_detail is True:            
            print("-----------------------------------------------")
            print()
        corr_list.append(correlation_data)
        if configs.plot is True and configs.sense_curve is True:
            plot.prog_data.append((results,correlation_data))
            plot.curr_system = system_val
            
    if configs.plot is True and configs.sense_curve is True:
        p_value = mean_corr_list(corr_list)
        plot.plot_now()
    else:
        p_value = mean_corr_list(corr_list)         
    return p_value     
    
def mean_corr_list(corr_list):
    summary = dict()
    corr_summary = dict()
    cost_list = []
    lambda_size_list = []
    optimal_sample_list = []
    accu_list = []
    for entry in corr_list:
        for keys in entry:
            if keys in corr_summary:
                if entry[keys]['correlation'] is not None and entry[keys]['accuracy'] is not None: 
                    corr_value = float(entry[keys]['correlation'])
                    corr_summary[keys].append(corr_value)
            else:
                if entry[keys]['correlation'] is not None and entry[keys]['accuracy'] is not None:
                    corr_summary[keys] = [float(entry[keys]['correlation'])]
                    
            if entry[keys]['selected'] is True:
                if keys in summary:
                    summary[keys] = summary[keys]+1
                else:
                    summary[keys] = 1
                if entry[keys]['total cost'] is not None:
                    cost_list.append(float(entry[keys]['total cost']))
                    optimal_sample_list.append(float(entry[keys]['optimal sample size']))
                    accu_list.append(float(entry[keys]['accuracy']))
                    
        if len(list(entry.values())) > 0:
            lambda_size_list.append(int(list(entry.values())[0]['lambda size']))    
    if len(cost_list) > 0:
        if configs.show_box_pl is False:        
            cost_result = np.percentile(cost_list,25),np.percentile(cost_list,75),np.median(cost_list)
        else:
            cost_result = cost_list
        size_result = np.percentile(lambda_size_list,25),np.percentile(lambda_size_list,75),np.median(lambda_size_list)
        opt_size_result = np.mean(optimal_sample_list)
        accu_result = np.mean(accu_list)
        if print_detail is True:
            print('Cost - 25-pc,75-pc,Med : ',cost_result)
            print(summary)
        actual_obs = dict_to_array(summary)[:,1].astype(int)
        success = str(sum(summary.values())) + '/' + str(configs.repeat)
        exp_obs = commonutils.get_random_distribution(sum(summary.values()), len(dict_to_array(summary)[:,1]))[0].astype(int)
        chisq,p = chisquare(actual_obs,exp_obs)
        p = p,summary,exp_obs
        if print_detail is True:
            print('p-value',p)
        for keys in corr_summary:
            mu, std = norm.fit(corr_summary[keys])
            '''corr_summary[keys].append('Mean : ' + str(mu) + ', std : '+ str(std)+ ', min : '+ str(max(corr_summary[keys]))+ ', med : '+ str(np.median(corr_summary[keys])))'''
        return size_result,accu_result,p,cost_result,opt_size_result
    else:
        return None
    

def main():           
    if system=='all':
        for i in all_systems:
            func = getattr(thismodule, strategy)
            func(i)
    else:
        func = getattr(thismodule, strategy)
        return func(system)     



