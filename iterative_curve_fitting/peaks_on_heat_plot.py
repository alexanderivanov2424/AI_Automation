"""
Copy from peak_fitting (small changes for shortness)

Plot the detected peaks as points on top of the peak heat map plot

"""
from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from peak_fitting.fit_data_sample import fit_curves_to_data, get_peak_indices

import matplotlib.pyplot as plt

import numpy as np
import math
import imageio
import os

"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/iterative_curve_fitting_save_test/"
regex = """params_(?P<num>.*?).csv"""
peakGrid = DataGrid(data_dir,regex)


layers = []
for i in range(len(dataGrid.row_sums)):
    layers.append(range(dataGrid.row_starts[i],dataGrid.row_sums[i]+1))



# grid locations to plot
#locations = range(82,96+1)
skip = 0
for locations in layers:
    skip += 1
    if skip < 7: # skip this many cross sections of the grid
        continue

    lst = []
    for L in locations:
            lst.append(dataGrid.data[L][:,1])
    im = np.array(lst)
    im_log = np.log(im + 1)
    im_sqrt = np.sqrt(im)

    SCALE = 10
    plt.imshow(np.repeat(im_log,SCALE,axis=0))
    for i,L in enumerate(locations):
        X = dataGrid.data_at_loc(L)[:,0]
        for peak in peakGrid.data_at_loc(L)[:,1]:
            p =np.argmin(np.abs(X - peak))
            plt.scatter(p,i*SCALE + SCALE//2,marker='x',color="red",s=5)

    plt.gca().invert_yaxis()
    #plt.xlim((0,400))
    plt.show()
    #plt.draw()
    #plt.pause(.001)
locations = range(82,96+1)


lst = []
for L in locations:
        lst.append(dataGrid.data[L][:,1])
im = np.array(lst)
im_log = np.log(im + 1)
im_sqrt = np.sqrt(im)

plt.imshow(im_log)
plt.gca().invert_yaxis()
plt.show()
