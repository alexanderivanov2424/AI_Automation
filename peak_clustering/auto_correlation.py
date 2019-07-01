


from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib


import numpy as np
import math


"""
Testing auto_correlation

"""


"""
Load Data
"""
dataGrid = DataGrid_TiNiSn_500C()

def autocorr(d1,d2):
    x1 = dataGrid.data_at_loc(d1)[:,1]
    x2 = dataGrid.data_at_loc(d2)[:,1]
    result = np.correlate(x1, x2, mode='full')
    return result[int(result.size/2):]



for i in range(1,178):
    for j in range(1,178):
        plt.plot(autocorr(i,j))
        plt.draw()
        plt.pause(.1)
        plt.cla()
