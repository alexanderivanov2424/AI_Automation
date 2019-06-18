'''

Run multiple simulations one after another and save results.

'''
#TODO Threading/make parallel

import os
command = "python3 -m algorithms.PSG --graphics --video -s"


for s in range(10):
    os.system(command + str(s))
