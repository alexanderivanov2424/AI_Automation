
from data_loading.data_grid_TiNiSn import DataGrid



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

data_dir = "/home/sasha/Desktop/saveTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
dataGrid = DataGrid(data_dir,regex)

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
    pA, _ = find_peaks(A)
    pB, _ = find_peaks(B)
    p = np.append(pA,pB,axis=0)
    cosine =  np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    peaks = np.dot(A[p],B[p])/np.linalg.norm(A[p])/np.linalg.norm(B[p])
    return cosine

delta = args.delta #.05

#cosine similarity function using two grid positions
def similarity(d1,d2):
    '''
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    return similarity_vector(a,b)
    '''
    #return 100 - abs(len(peak_lists[d1-1]) - len(peak_lists[d2-1]))

    differences = 0
    for p in peak_lists[d1-1]:
        found = False
        for t in peak_lists[d2-1]:
            if abs(float(t)-float(p)) < delta:
                found = True
        if not found:
            differences += 1
    for p in peak_lists[d2-1]:
        found = False
        for t in peak_lists[d1-1]:
            if abs(float(t)-float(p)) < delta:
                found = True
        if not found:
            differences += 1
    return differences/4


points = [[6,1]]
for val in range(2,178):
    x,y = dataGrid.coord(val)
    points = np.append(points,[[x,y]],axis=0)

size = len(points)
'''
K_Matrix = np.zeros(shape=(size,size))
for x in range(1,size+1):
    for N in dataGrid.neighbors(x).values():
        K_Matrix[x-1,N-1] = 1
'''


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

cg = get_cluster_grids(C)
plt.imshow(cg)
plt.gca().invert_yaxis()
plt.axis("off")
plt.title(C)

#k=.01
#plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
#plt.savefig("/home/sasha/Desktop/Peak_Clustering_Images/clust-" + str(delta) + "-" + str(C) + ".png")
plt.show()
plt.close()
