'''
Created on 2016-01-23

@author: Atri
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import math as math
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
from numpy import mean

thismodule = sys.modules[__name__]

base_dir = 'C:\\Users\\Atri\\juno_high\\ASE_extn\\com\\ase\\extn\\cart\\data\\'
base_dir_in = base_dir+'input\\'
base_dir_out = base_dir+'output\\'

system = 'apache'
all_systems = ['apache','bc','bj','llvm','sqlite','x264']
details_map = {"apache" : [9,192], "llvm" : [11,1024], "x264" : [16,1152], "bc" : [18,2560], "bj" : [26,180], "sqlite" : [39,4553]}
strategy_1 = "progressive"

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
    fname = base_dir_in+system
    num_features = range(0,details_map[system][0]-1)
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(str)
    return data

def load_perf_values():
    fname = base_dir_in+system
    data = np.loadtxt(fname,  delimiter=',', dtype=float,skiprows=1,usecols=(details_map[system][0],))
    return data

def load_feature_names():
    fname = base_dir_in+system
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

def progressive(system_val):
    global system
    system = system_val    
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)
    repeat = 30
    total_range = range((details_map[system][1]//10)//2)
    results = np.empty((len(total_range),repeat))
    for j in range(repeat):
        for i in total_range:
            np.random.seed(j)
            curr_size = 10*(i+1)
            training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
            diff_indices = set(range(data.shape[0])) - set(training_set_indices)
            training_set = data[training_set_indices]
            test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
            test_set = data[test_set_indices]
            X = training_set
            y = perf_values[training_set_indices]
            built_tree = cart(X, y)
            out = predict(built_tree, test_set, perf_values[test_set_indices])
            results[i][j] = calc_accuracy(out,perf_values[test_set_indices])
        print('['+system+']' + "done iteration :"+str(j))
    print()
    out_file = open(base_dir_out+system+"_out_"+strategy_1,'w')
    out_file.truncate()
    
    for i in range(results.shape[0]):
        out_file.write(str((i+1)*10)+","+ str(mean(results[i])))
        out_file.write('\n')


def projective(system_val):
    global system
    system = system_val
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)    
    i=0
    repeat = 10
    freq_table = np.array(2,details_map[system])
    for j in range(repeat):
        while True:
            curr_size = (i+1)
            np.random.seed(j)
            training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
            diff_indices = set(range(data.shape[0])) - set(training_set_indices)
            training_set = data[training_set_indices]
            test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
            test_set = data[test_set_indices]
            X = training_set
            y = perf_values[training_set_indices]
            built_tree = cart(X, y)
            out = predict(built_tree, test_set, perf_values[test_set_indices])
            
if system=='all':
    for i in all_systems:
        func = getattr(thismodule, strategy_1)
        func(i)
else:
    func = getattr(thismodule, strategy_1)
    func(system)    
   