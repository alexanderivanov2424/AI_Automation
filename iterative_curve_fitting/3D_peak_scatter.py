
"""
Plot peaks in 3D space. 2 axies for spacial dimensions and one for Q space.


"""


#3D plot import
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np


from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

data_dir = "/home/sasha/Desktop/iterative_curve_fitting_save_test/"
regex = """params_(?P<num>.*?).csv"""
peakGrid = DataGrid(data_dir,regex)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for loc in peakGrid.data.keys():
    x,y = peakGrid.coord(loc)
    for i,peak in enumerate(peakGrid.data_at_loc(loc)[:,1]):
        if peakGrid.data_at_loc(loc)[i,0] > 10:
            ax.scatter([x], [y], [peak], marker='o',color = 'red')
    #xs = [x for p in peaks]
    #ys = [y for p in peaks]


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Q')
#ax.set_xlim3d(0, 150)
#ax.set_ylim3d(0, 150)
#ax.set_zlim3d(0, 1000)

plt.show()
