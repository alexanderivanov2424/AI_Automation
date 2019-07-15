
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

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
#data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

ShowBBA = True

if not ShowBBA:
    data_dir = "/home/sasha/Desktop/peakData_temp/"
    for loc in range(1,dataGrid.size+1):
        file = data_dir + str(loc) + ".txt"
        try:
            peaks = eval(open(file).read())
        except:
            print("Generating " + str(loc))
            X = dataGrid.data_at_loc(loc)[:,0]
            Y = dataGrid.data_at_loc(loc)[:,1]

            #curve_params = fit_curves_to_data(X,Y)
            peaks = get_peak_indices(X,Y)
            open(file,'w+').write(str(peaks))
        peakGrid.data[loc] = peaks







layers = []
for i in range(len(dataGrid.row_sums)):
    layers.append(range(dataGrid.row_starts[i],dataGrid.row_sums[i]+1))



# grid locations to plot
#locations = range(82,96+1)
for locations in layers:
    lst = []
    for L in locations:
            lst.append(dataGrid.data[L][:,1])
    im = np.array(lst)
    im_log = np.log(im + 1)
    im_sqrt = np.sqrt(im)

    SCALE = 20
    plt.imshow(np.repeat(im,SCALE,axis=0))
    if ShowBBA:
        for i,L in enumerate(locations):
            X = dataGrid.data_at_loc(L)[:,0]
            for peak in peakGrid.data_at_loc(L)[:,1]:
                p =np.argmin(np.abs(X - peak))
                plt.scatter(p,i*SCALE + SCALE//2,marker='x',color="red",s=5)
    else:
        for i,L in enumerate(locations):
            for p in peakGrid.data_at_loc(L):
                plt.scatter(p,i*SCALE + SCALE//2,marker='x',color="red",s=5)
    plt.gca().invert_yaxis()
    #plt.xlim((0,400))
    plt.show()
    #plt.draw()
    #plt.pause(.3)
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
