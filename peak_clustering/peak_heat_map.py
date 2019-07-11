

from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import math


dataGrid = DataGrid_TiNiSn_500C()

layers = []
for i in range(len(dataGrid.row_sums)):
    layers.append(range(dataGrid.row_starts[i],dataGrid.row_sums[i]+1))



# grid locations to plot
#locations = range(82,96+1)
for locations in layers:
    lst = []
    for L in locations:
            lst.append(dataGrid.data[L][120:160,1])
    im = np.array(lst)
    im_log = np.log(im + 1)
    im_sqrt = np.sqrt(im)

    plt.imshow(im_log)
    plt.gca().invert_yaxis()
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
