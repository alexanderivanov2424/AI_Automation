"""
Clustering algorithm on the actual heat map

"""

from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import random
import math
import sys


dataGrid = DataGrid_TiNiSn_500C()

fig = plt.figure(figsize=(15,5))

# grid locations to plot
locations = range(82,96+1)

lst = []
for L in locations:
    lst.append(dataGrid.data[L][:,1])
im = np.array(lst)
im = im[:,0:200]
x = plt.subplot(2,1,1)
x.imshow(im)

points = []
for x in range(len(im)):
    for y in range(len(im[0])):
        P = im[x,y]
        #P = math.sqrt(im[x,y])
        for i in range(int(P)):
            xr = random.random()
            yr = random.random()
            points.append([x+xr,y+yr])
"""
for p in points:
    plt.plot(p[0],p[1],'o',color="black")
    if random.random() < .05:
        plt.draw()
        plt.pause(.0001)
plt.show()
"""

X = np.array(points)
clustering = DBSCAN(eps=0.16, min_samples=10).fit(X)
clustering.labels_ = GaussianMixture(n_components=10,covariance_type="tied").fit_predict(X)
#clustering = AgglomerativeClustering(n_clusters=5,linkage="single").fit(X)
#num_clusters = len(set(clustering.labels_).difference(set([-1])))

"""
for i in set(clustering.labels_):
    if len(np.where(clustering.labels_ == i)[0]) < 20:
        clustering.labels_[clustering.labels_==i] = -1
        for j in range(len(clustering.labels_)):
            if clustering.labels_[j] > i:
                clustering.labels_[j] -= 1
"""
data_clusters = np.zeros(shape=(15,dataGrid.data_length))

for i,p in enumerate(points):
    x = int(p[0])
    q = int(p[1])
    data_clusters[x,q] = int(clustering.labels_[i])
"""
hues = [float(float(x)/float(i)) for x in range(1,num_clusters+1)]
get_hue = lambda l : 0 if l == -1 else hues[l]+1
get_hue_V = np.vectorize(get_hue)
"""
im = data_clusters[:,0:200]
x = plt.subplot(2,1,2)
x.imshow(im)
plt.show()
#im = np.repeat(im,10, axis=0)
#plt.imshow(im)
#plt.show()
sys.exit()

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
#im = np.repeat(im,10, axis=0)
plt.imshow(im)
plt.show()
