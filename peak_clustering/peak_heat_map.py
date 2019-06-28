

from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import math


dataGrid = DataGrid_TiNiSn_500C()


# grid locations to plot
locations = range(82,96+1)

im = []
for L in locations:
    for k in range(10):
        im.append(dataGrid.data[L][:,1])

plt.imshow(im)
plt.show()
