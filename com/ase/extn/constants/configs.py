'''
Created on 2016-02-01

@author: Atri
'''
import os
import sys

thismodule = sys.modules[__name__]
loc = os.path.dirname(__file__)

base_dir = os.path.join(loc,'data')
base_dir_in = os.path.join(base_dir,'input')
base_dir_tway_in = os.path.join(base_dir_in,'tway')
base_dir_out = os.path.join(base_dir,'output')



details_map = {"apache" : [9,192], "llvm" : [11,1024], "x264" : [16,1152], "bc" : [18,2560], "bj" : [26,180], "sqlite" : [39,4553]}
all_systems = ['apache','bc','bj','llvm','x264','sqlite']
'''all_systems = ['sqlite']'''

'''Strategy is progressive|projective'''
strategy = 'projective'
system = 'apache'

''' This adds a,b to the correaltion data structure'''
track_detail = True

''' Whether to use a fix test et size or a varying one'''
fix_test_set = True
fix_test_ratio = 3

''' Plot cost instead of accuracy ; Works only with progressive'''
plot_real_cost = False
'''' Highlight the optimal sample size'''
calc_prog_opt = True

''' For projective sampling, this will show the actual learning curve too'''
show_actual_lc = True
''' For projective sampling, whether to show only the selected learning curve'''
show_all_lc = False
''' Used to show box plots for cost. Set this to true and run execute.py. Please do set it to false for all other scenarios'''
show_box_pl = False

smooth = True
''' Plot the data '''
plot = True
print_detail = True

transform_lambda = False
add_origin_to_lambda = True
extend_lambda = False

projective_feature_threshold = 2
repeat = 10

''' Which prediction model to use'''
model = 'cart'

min_corr = 0

tway = 2

sense_curve = True
ignore_initial = 0

chi_sq_with_random = True

''' Options are dynamic|static|composite'''
curve_selection = 'static'
dynamic_recursive_curve_selection = False

r_0_to_1 = False



''' r = Cost of prediction error / Cost of measurement'''
r = 1
th = 2