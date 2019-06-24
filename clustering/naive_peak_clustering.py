
from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C



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



import argparse

parser = argparse.ArgumentParser(description='Run Peak Clustering')
parser.add_argument('-d','--delta', type=float, default=.1,
                    help='peak shift allowance')
args = parser.parse_args()


#folder with data files

dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/saveTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

# dataGrid.data[0][1:,1] is peak locations

# dataGrid.data[0][1:,3] is peak intensity


##################################
# LOAD PEAK DATA
'''
k1 = 97
k2 = 134

X = dataGrid.data[k1][:,0]
Y = dataGrid.data[k1][:,1]
plt.plot(X,Y)

X = dataGrid.data[k2][:,0]
Y = dataGrid.data[k2][:,1]
plt.plot(X,Y+100)

print(peak_lists[k1-1])
print(peak_lists[k2-1])
plt.show()
'''
#check for missing files
#print(len([i+1 for i,x in enumerate(peak_lists) if x == []]))
#print([i+1 for i,x in enumerate(peak_lists) if x == []])
#sys.exit()
##################################


def similarity_vector(A,B):
    cosine =  np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    return cosine

delta = args.delta #.05
size = 50
def similarity(d1,d2):

    differences = 0
    for i,p in enumerate(peakGrid.data_at_loc(d1)[1:,1]):
        if float(peakGrid.data_at_loc(d1)[i+1,3]) < size:
            continue
        found = False
        for t in peakGrid.data_at_loc(d2)[1:,1]:
            if abs(float(t)-float(p)) < delta:
                found = True
        if not found:
            differences += 1
    for i,p in enumerate(peakGrid.data_at_loc(d2)[1:,1]):
        if float(peakGrid.data_at_loc(d2)[i+1,3]) < size:
            continue
        found = False
        for t in peakGrid.data_at_loc(d1)[1:,1]:
            if abs(float(t)-float(p)) < delta:
                found = True
        if not found:
            differences += 1
    return differences/4


size = dataGrid.size

K_Matrix = np.zeros(shape=(size,size))
for x in range(1,size+1):
    for N in dataGrid.neighbors(x).values():
        K_Matrix[x-1,N-1] = 1



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

cg = get_cluster_grids(10)
plt.imshow(cg)
plt.gca().invert_yaxis()
plt.axis("off")
plt.title(C)

#k=.01
#plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
#plt.savefig("/home/sasha/Desktop/Peak_Clustering_Images/clust-" + str(delta) + "-" + str(C) + ".png")
plt.show()
plt.close()
