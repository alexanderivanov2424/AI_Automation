'''

Run multiple simulations one after another and save results.

'''
#TODO Threading/make parallel

import os
command = "python3 -m peak_clustering.peak_clustering -d"

start = .03
increment = .001
stop = .07
while start <=.07:
    os.system(command + str(start))
    start += increment
    start = float('%.3f'%(start))
