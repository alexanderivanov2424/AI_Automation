

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import math
import random



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


"""
PCA ON REDUCED DIFFRACTION DATA
"""


pca = PCA(n_components = 'mle',svd_solver='full').fit_transform(M)
pca = PCA(n_components = 20,svd_solver='full').fit_transform(M)
print(len(M[0]))
print(len(pca[0]))

def get_cluster_grids(i):
    agg = AgglomerativeClustering(n_clusters=i).fit(pca)
    #i = max(agg.labels_)+1

    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

    cluster_grid = np.zeros(shape = (15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        if cluster == -1:
            continue
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

    """
    Find max peak count/ max peak width for each cluster
    """
    max_over_locations = lambda locs : np.amax([len(peakGrid.data[L][:,1]) for L in locs])
    Peak_max = [max_over_locations(np.where(agg.labels_==L)[0] + 1) for L in range(i)]

    max_over_locations = lambda locs : np.nanmax([np.nanmax(peakGrid.data[L][:,2].astype(np.float)) for L in locs])
    Width_max = [max_over_locations(np.where(agg.labels_==L)[0] + 1) for L in range(i)]


    peak_grid = np.zeros(shape =(15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = len(peakGrid.data_at_loc(val)[:,1])/Peak_max[cluster]
        peak_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([1,1,k])

    width_grid = np.zeros(shape =(15,15))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = max(peakGrid.data_at_loc(val)[:,2].astype(np.float))/Width_max[cluster]
        width_grid[y-1][15-x] = k

    return cluster_grid,peak_grid,width_grid

"""
Plotting
"""
for i in range(10,30):

    fig = plt.figure(figsize=(15,8))
    cg,pg,wg = get_cluster_grids(i)

    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(cg)
    for j in range(dataGrid.size):
        x,y = dataGrid.coord(j+1)
        ax1.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)
    ax1.axis("off")
    ax1.invert_yaxis()
    ax1.title.set_text(i)


    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(pg)
    ax2.axis("off")
    ax2.invert_yaxis()
    ax2.title.set_text(i)

    ax2 = fig.add_subplot(1,3,3)
    ax2.imshow(wg)
    ax2.axis("off")
    ax2.invert_yaxis()
    ax2.title.set_text(i)

    plt.show()
