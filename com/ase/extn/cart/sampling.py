'''
Created on 2016-01-26

@author: Atri
'''

from .cart import *;
import numpy as np

def main():    
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)
    limit = 3
    total_range = range(1,details_map[system][1]//10)
    for i in total_range:
        if i > limit:
            break
        curr_size = 10*i
        training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
        diff_indices = set(range(data.shape[0])) - set(training_set_indices)
        training_set = data[training_set_indices]
        test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
        test_set = data[test_set_indices]
        X = training_set
        y = perf_values[training_set_indices]
        built_tree = cart(X, y)
        out = predict(built_tree, test_set, perf_values[test_set_indices])
        print("Running with size :" + curr_size)
        print(calc_accuracy(out,perf_values[test_set_indices]))
        print()
    
main()