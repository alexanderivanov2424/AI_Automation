'''
Script for plotting spectra at several data values to
visually compare differences.
'''


from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math


dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/iterative_curve_fitting_save_test/"
regex = """params_(?P<num>.*?).csv"""
peakGrid = DataGrid(data_dir,regex)

# grid locations to plot
locations = [152,151,150,149,148,147,137,136,123]

#how much to shift each grid location vertically
#(makes it easier to see peaks)
#shifts = [0,100,200,300,400]
shifts = [100 * i for i in range(len(locations))]

colors = cm.get_cmap("viridis")

for i,k in enumerate(locations):
    y = dataGrid.data[k][:,1]
    if len(shifts) == len(locations):
        y = y + shifts[i]
    x = dataGrid.data[k][:,0]
    plt.plot(x,y,label=str(k),color=colors(i/len(locations)))
    for peak in peakGrid.data_at_loc(k):
        if len(shifts) == len(locations):
            plt.plot(peak[1], shifts[i],color=colors(i/len(locations)),marker="o")
        else:
            plt.plot(peak[1],shifts[i],color=colors(i/len(locations)),marker="o")
plt.show()
