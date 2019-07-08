

from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import math
import sys


dataGrid = DataGrid_TiNiSn_500C()

"""
Improve Data Contrast
"""

grid_scale = 1/1000
peak_scale = 1/10

points = []
for x in range(1,15):
    for y in range(1,15):
        for q in range(dataGrid.data_length):
            if dataGrid.in_grid(x,y):
                points.append([x*grid_scale,y*grid_scale,dataGrid.data_at(x,y)[q,0],dataGrid.data_at(x,y)[q,1]*peak_scale])

X = np.array(points)
clustering = DBSCAN(eps=0.05, min_samples=1).fit(X)
num_clusters = len(set(clustering.labels_).difference(set([-1])))


data_clusters = np.zeros(shape=(15,15,dataGrid.data_length))

for i,p in enumerate(points):
    x = int(p[0]/grid_scale)
    y = int(p[1]/grid_scale)
    q = int(np.where(dataGrid.data_at(x,y)[:,0] == p[2])[0][0])
    data_clusters[15-x,y,q] = int(clustering.labels_[i])
"""
hues = [float(float(x)/float(i)) for x in range(1,num_clusters+1)]
get_hue = lambda l : 0 if l == -1 else hues[l]+1
get_hue_V = np.vectorize(get_hue)
"""
im = data_clusters[:,8,:]
im = np.repeat(im,10, axis=0)
plt.imshow(im)
plt.show()


"""
Threshold
"""

grid_scale = 1/100
peak_thresh = 100

points = []
for x in range(1,15):
    for y in range(1,15):
        for q in range(dataGrid.data_length):
            if dataGrid.in_grid(x,y) and dataGrid.data_at(x,y)[q,1] > peak_thresh:
                points.append([x*grid_scale,y*grid_scale,dataGrid.data_at(x,y)[q,0]])

X = np.array(points)
clustering = DBSCAN(eps=0.05, min_samples=1).fit(X)
num_clusters = len(set(clustering.labels_).difference(set([-1])))


data_clusters = np.zeros(shape=(15,15,dataGrid.data_length))

for i,p in enumerate(points):
    x = int(p[0]/grid_scale)
    y = int(p[1]/grid_scale)
    q = int(np.where(dataGrid.data_at(x,y)[:,0] == p[2])[0][0])
    data_clusters[15-x,y,q] = int(clustering.labels_[i])
"""
hues = [float(float(x)/float(i)) for x in range(1,num_clusters+1)]
get_hue = lambda l : 0 if l == -1 else hues[l]+1
get_hue_V = np.vectorize(get_hue)
"""
im = data_clusters[:,8,:]
im = np.repeat(im,10, axis=0)
plt.imshow(im)
plt.show()
