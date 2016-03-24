'''
Created on 2016-02-13

@author: Atri
'''
import matplotlib.pyplot as plt
import numpy as np

x = range(0,500)
plt.axis([0, 500, 1,100])
a = 58.3045513580588217
b = 0.9885676984176499
print((a * (np.power(b,x))))
plt.plot(100-(a * (np.power(b,x))),'g')
plt.plot(100,72,marker='^',markersize=20,c='k')
plt.ylabel('some numbers')


plt.show()