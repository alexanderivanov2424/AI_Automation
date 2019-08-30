

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA


import numpy as np
import math
import random



"""
Plotting the unique peak reductions in the wafer.
This is used to check the consistency of the peak finding algorithm.



each color corresponds to a unique peak vector

(This illustrates that every single grid location is its own region)
"""



"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

"""
Create a list of peaks in the form [x,y,p]
"""
SCALE = 100
def to_point(x,y,p):
    return [(x-1)/15.,(y-1)/15.,SCALE*float(p)/5]

peaks = []
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    [peaks.append(to_point(x,y,p)) for p in peakGrid.data_at_loc(k)[:,1]]


"""
DBSCAN PEAK CLUSTERING
"""
X = np.array(peaks)

clustering = DBSCAN(eps=0.25, min_samples=5).fit(X)

C = len(set(clustering.labels_).difference(set([-1])))

"""
REDUCE DIMENSIONS BASED ON PEAK CLUSTERING
"""
M = np.zeros(shape=(peakGrid.size,C))

for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    V = np.zeros(shape=C)
    for i,p in enumerate(peakGrid.data_at_loc(k)[:,1]):
        loc = clustering.labels_[peaks.index(to_point(x,y,p))]
        if loc == -1:
            continue
        M[k-1,loc] = 1#peakGrid.data_at_loc(k)[i,3]

M_copy = M.copy()
for i in range(len(M_copy)):
    for j in range(i+1,len(M_copy)):
        if j >= len(M_copy):
            continue
        if np.array_equal(M_copy[i],M_copy[j]):
            M_copy = np.delete(M_copy,j,axis=0)
            j -= 1
C = len(M_copy)
hues = [float(float(x)/float(i)) for x in range(1,C+1)]
random.shuffle(hues)


grid = np.zeros(shape = (15,15,3))
for val in range(1,178):
    x,y = dataGrid.coord(val)
    for i in range(len(M_copy)):
        if np.array_equal(M[val-1],M_copy[i]):
            grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[i],1,1])
plt.imshow(grid)
for j in range(dataGrid.size):
    x,y = dataGrid.coord(j+1)
    plt.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)
plt.gca().invert_yaxis()
plt.show()
