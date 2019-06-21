
from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C


from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import os
import pandas as pd
import sys



#folder with data files

dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/saveTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

max_peaks = 0
for k in peakGrid.data.keys():
    max_peaks = max(max_peaks,len(peakGrid.data[k]))
print("Maximum number of peaks: ",max_peaks)
pca = PCA(n_components=25)
X_red = pca.fit_transform(dataGrid.get_data_array())


def similarity_vector(A,B):
    cosine =  np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    return cosine

def similarity(d1,d2):
    a = X_red[d1-1]#dataGrid.data_at_loc(d1)[:,1]
    b = X_red[d2-1]#dataGrid.data_at_loc(d2)[:,1]
    #a = np.log(a+100)
    #b = np.log(b+100)
    return np.mean(np.abs(a-b))
    return math.sqrt(np.sum(np.square(a-b)))


size = dataGrid.size

K_Matrix = np.zeros(shape=(size,size))
for x in range(1,size+1):
    K_Matrix[x-1,x-1] = 1
    for N in dataGrid.neighbors(x).values():
        K_Matrix[x-1,N-1] = 1
        for N2 in dataGrid.neighbors(N).values():
            K_Matrix[x-1,N2-1] = 1

D = np.ones(shape=(size,size))
for x in range(size):
    for y in range(size):
        D[x,y] = similarity(x+1,y+1)




#calculate i clusters and create grid visuals and center points

def get_cluster_grids(i):
    agg = AgglomerativeClustering(n_clusters=i,affinity='precomputed',linkage='complete')
    agg.fit(D)

    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

    cluster_grid = np.zeros(shape = (15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])


    return cluster_grid

C = len(np.unique(D))
print(C)
start = 6
end = 14
fig = plt.figure()
fig.tight_layout()
for i in range(start,end):
    cg = get_cluster_grids(i)
    ax = fig.add_subplot(1,end-start,i-start+1)
    ax.imshow(cg)
    ax.invert_yaxis()
    ax.title.set_text(i)
    ax.axis("off")

k=.01
plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
#plt.savefig("/home/sasha/Desktop/Peak_Clustering_Images/clust-" + str(delta) + "-" + str(C) + ".png")
plt.show()
plt.close()
