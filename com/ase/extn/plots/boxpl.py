'''
Created on Mar 18, 2016

@author: Atri
'''

import matplotlib.pyplot as plt
from com.ase.extn.constants import configs
import numpy as np
import collections


progressive_cost = 0
global_min_cost = 0
data = collections.OrderedDict()
system_id = ''
lims_min, lims_max =[],[]
lambda_size = collections.OrderedDict()

def setup(system):
    global progressive_cost
    global data
    global global_min_cost
    global system_id
    global lims_max
    global lims_min
    global lambda_size
    progressive_cost,global_min_cost = 0,0
    data = collections.OrderedDict()
    system_id = system
    lims_min, lims_max =[],[]
    lambda_size = collections.OrderedDict()

def append(a,b):
    return str(a) + '(' + str(int(b)) + ')'

def show():
    fig, ax1 = plt.subplots(figsize=(8,4.2))
    fig.canvas.set_window_title(system_id)
    plt.title(system_id)
    plt.xlabel('Sampling strategy')
    plt.ylabel('Total cost')
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    bp = plt.boxplot(list(data.values()), notch=0, sym='', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    lims_max.append(bp['whiskers'][1].get_ydata()[1])
    lims_min.append(bp['whiskers'][0].get_ydata()[0])
    randomDists = map(append,list(data.keys()),list(lambda_size.values()))
    xtickNames = plt.setp(ax1, xticklabels=randomDists)
    plt.setp(xtickNames, rotation=45, fontsize=12)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    if ymax > progressive_cost and ymin < progressive_cost:
        plt.axhline(y=progressive_cost,c='m')
    else:
        x = xmax - 1.5
        y = ymax - (0.05 * (ymax - ymin))
        plt.text(x, y, 'Cost (progressive) =' + str(int(progressive_cost)),fontsize=8)
        
    if ymax > global_min_cost and ymin < global_min_cost:
        plt.axhline(y=global_min_cost,c='b')
    else:
        x = xmax - 1.5
        y = ymax - (0.1 * (ymax - ymin))
        plt.text(x, y, 'Global minimum =' + str(int(global_min_cost)), fontsize=8)    
    
    
    plt.show()

