'''
Created on 2016-02-17

@author: Atri
'''
from com.ase.extn.constants import configs
from com.ase.extn.cart import base
from com.ase.extn.tway import twaysample
import os

def run():
    configs.print_detail = False
    configs.plot = False
    configs.fix_test_set = True
    configs.fix_test_ratio = 3
    configs.smooth = True
    configs.transform_lambda = False
    configs.add_origin_to_lambda = True
    configs.extend_lambda = False
    configs.projective_feature_threshold = 5
    configs.repeat = configs.repeat
    configs.min_corr = 0
    configs.curve_selection = 'static'
    out_file = open(configs.base_dir_out+"_result_transf_"+str(configs.transform_lambda)+'_smooth_'+str(configs.smooth)+'_'+str(configs.repeat),'w')
    configs.chi_sq_with_random = True
    out_file.truncate()
    for system_key in configs.all_systems:
        for i in range(3,7):
            configs.projective_feature_threshold = i
            print(str(system_key)+"-feature-frequencies-"+str(i)+": ")
            out_file.write(str(system_key)+"-feature-frequencies-"+str(i)+": ")
            configs.system = system_key
            size_result,success,p_value,cost_result,opt_size_result = base.projective(system_key)
            if p_value is not None:
                print(str(p_value))
                out_file.write(str(p_value))
            else:
                print('None')
                out_file.write('None')
            print()
            out_file.write('\n')
            out_file.write('Cost Result : ')
            out_file.write(str(cost_result))
            out_file.write('\n')
            out_file.write('Size : ')
            out_file.write(str(size_result))
            print()
            print('Cost Result : ',str(cost_result))
            print()
            print()
            out_file.write('\n')
            out_file.write('Success rate : ')
            out_file.write(str(success))
            print()
            print('Success rate : ',str(success))
            print()
            out_file.write('\n')
        
        
        out_file.write('\n')
        configs.tway = 2
        print(str(system_key)+"-2way: ")
        out_file.write(str(system_key)+"-2way: ")
        size_result,success,p_value,cost_result,opt_size_result = twaysample.sample(system_key)
        if p_value is not None:
            print(str(p_value))
            out_file.write(str(p_value))
        else:
            print('None')
            out_file.write('None')
        print()
        out_file.write('\n')
        out_file.write('Cost Result : ')
        out_file.write(str(cost_result))
        print()
        print('Cost Result : ',str(cost_result))
        print()
        print()
        out_file.write('\n')
        out_file.write('Size : ')
        out_file.write(str(size_result))
        out_file.write('\n')
        out_file.write('Success rate : ')
        out_file.write(str(success))
        print()
        print('Success rate : ',str(success))
        print()
        out_file.write('\n')
        out_file.write('\n')
        configs.tway = 3
        out_file.write(str(system_key)+"-3way: ")
        size_result,success,p_value,cost_result,opt_size_result = twaysample.sample(system_key)
        if p_value is not None:
            print(str(p_value))
            out_file.write(str(p_value))
        else:
            print('None')
            out_file.write('None')
        print()
        out_file.write('\n')
        out_file.write('Cost Result : ')
        out_file.write(str(cost_result))
        out_file.write('\n')
        out_file.write('Size : ')
        out_file.write(str(size_result))
        print()
        print('Cost Result : ',str(cost_result))
        print()
        out_file.write('\n')
        print()
        out_file.write('Success rate : ')
        out_file.write(str(success))
        print()
        print('Success rate : ',str(success))
        print()
        out_file.write('\n')
        print("---------------------------------------------")
        out_file.write('\n')
        out_file.write("---------------------------------------------")
        out_file.write('\n')
    
run()       
        
        