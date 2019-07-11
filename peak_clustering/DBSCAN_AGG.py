

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib


from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import math

"""
################################
Perform clustering using peak based dimension reduction and then L1 similarity
in the agglomerative clustering algorithm.
################################

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
Cluster the peaks into C clusters
"""
X = np.array(peaks)

clustering = DBSCAN(eps=0.25, min_samples=10).fit(X)

C = len(set(clustering.labels_).difference(set([-1])))

"""
Create a map of each grid location to a Vector of peaks in C dimensions
"""
M = np.zeros(shape=(peakGrid.dims[0],dataGrid.dims[1],C))

i = 0
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    V = np.zeros(shape=C)
    for i,p in enumerate(peakGrid.data_at_loc(k)[:,1]):
        loc = clustering.labels_[peaks.index(to_point(x,y,p))]
        if loc == -1:
            continue
        M[x-1,y-1,loc] = peakGrid.data_at_loc(k)[i,3]


"""
Similarity function
"""
def similarity(d1,d2):
    x,y = peakGrid.coord(d1)
    a = M[x-1,y-1]
    x,y = peakGrid.coord(d2)
    b = M[x-1,y-1]
    #return abs(len(a[a!=0]) - len(b[b!=0]))
    return np.mean(np.abs(a-b))


    #return math.sqrt(np.sum(np.square(a-b)))
    #return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)


size = peakGrid.size

"""
Connectivity Matrix
"""
K_Matrix = np.zeros(shape=(size,size))
for x in dataGrid.data.keys():
    for N in dataGrid.neighbors(x).values():
        K_Matrix[x-1,N-1] = 1


"""
Similarity Matrix
"""
D = np.ones(shape=(size,size))
for x in range(size):
    for y in range(size):
        D[x,y] = similarity(x+1,y+1)


"""
Grid Clustering based on similarity matrix
"""

def get_cluster_grids(i):
    agg = AgglomerativeClustering(n_clusters=i,affinity='precomputed',linkage='complete').fit(D)
    #agg =  DBSCAN(eps=1.5, min_samples=3).fit(M)
    #agg = AgglomerativeClustering(n_clusters=i).fit(M)

    #i = max(agg.labels_)+1

    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

    cluster_grid = np.zeros(shape = (15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])


    peak_max_counts = np.zeros(i)
    for val in range(1,178):
        cluster = agg.labels_[val-1]
        peak_max_counts[cluster] = max(peak_max_counts[cluster],len(peakGrid.data_at_loc(val,True)[:,1]))

    peak_grid = np.zeros(shape =(15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = len(peakGrid.data_at_loc(val,True)[:,1])/peak_max_counts[cluster]
        peak_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([1,1,k])



    width_max_counts = np.zeros(i)
    for val in range(1,178):
        cluster = agg.labels_[val-1]
        width_max_counts[cluster] = max(width_max_counts[cluster],max(peakGrid.data_at_loc(val,True)[:,2].astype(np.float)))

    width_grid = np.zeros(shape =(15,15))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = max(peakGrid.data_at_loc(val,True)[:,2].astype(np.float))/width_max_counts[cluster]
        width_grid[y-1][15-x] = k

    return cluster_grid, peak_grid, width_grid


"""
Plotting
"""
for i in range(10,30):

    fig = plt.figure(figsize=(15,8))
    cg,pg,wg = get_cluster_grids(i)

    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(cg)
    for j in range(dataGrid.size):
        x,y = dataGrid.coord(j+1)
        ax1.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)
    ax1.axis("off")
    ax1.invert_yaxis()
    ax1.title.set_text(i)

    """
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
    """

    plt.show()
