"""
Custom Clustering algorithm to cluster based on density plot

"""

from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import random
import math
import sys



dataGrid = DataGrid_TiNiSn_500C()



def contrast(arr,k):
    ret = arr.copy()
    for i in range(len(arr)):
        a = max(0,i-k)
        b = min(len(arr)-1,i+k)
        ret[i] += arr[i] - np.median(arr[a:b])
    return ret

def smooth(arr,k):
    ret = arr.copy()
    for i in range(len(arr)):
        a = max(0,i-k)
        b = min(len(arr)-1,i+k)
        ret[i] = np.mean(arr[a:b])
    return ret
"""
arr = dataGrid.data[1][:,1]
for i in range(5):
    plt.cla()
    arr = contrast(arr,30)
    arr = np.clip(arr,0,1000)
    #arr = smooth(arr,2)
    plt.plot(arr)
    plt.draw()
    plt.pause(.1)

plt.show()
"""


# grid locations to plot
locations = range(82,96+1)

lst = []
for L in locations:
    arr = dataGrid.data[L][:,1]
    arr = contrast(arr,30)
    arr = contrast(arr,30)
    lst.append(arr)
im = np.array(lst)
im = im[:,0:400]


def detect_peaks(image):
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks

peaks = detect_peaks(im)

ax = plt.subplot(1,1,1)
for x in range(peaks.shape[0]):
    for y in range(peaks.shape[1]):
        if peaks[x][y] and not im[x][y] <= 0:
            ax.plot(y,x*5,'x',color="red")

ax.imshow(np.log(np.repeat(im,5,axis=0)))

plt.show()
