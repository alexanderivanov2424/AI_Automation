

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from peak_fitting.fit_data_sample import fit_curves_to_data, get_peak_indices

import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA


import numpy as np
import math
import imageio
import os



"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
#data_dir = "/home/sasha/Desktop/peakTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)


data_dir = "/home/sasha/Desktop/peakData_temp/"
for loc in range(1,dataGrid.size+1):
    file = data_dir + str(loc) + ".txt"
    try:
        peaks = eval(open(file).read())
    except:
        print("Generating " + str(loc))
        X = dataGrid.data_at_loc(loc)[:,0]
        Y = dataGrid.data_at_loc(loc)[:,1]

        #curve_params = fit_curves_to_data(X,Y)
        peaks = get_peak_indices(X,Y)
        open(file,'w+').write(str(peaks))
    peakGrid.data[loc] = peaks


# grid locations to plot
locations =[85,69,53,39]
#locations = range(82,96+1)
#locations = [4,11,21,33,45,60,75,90,105,120,134,147,159,169,176]
locations = [1,2,3]
#how much to shift each grid location vertically
#(makes it easier to see peaks)
shifts = [150*i for i,v in enumerate(locations)]
#shifts = [0,100,200,300,500,600,700,800,1000,1100,1200,1300,1500,1600,1700,1800]


"""
Load and Cluster peaks
"""

SCALE = 1000
def to_point(x,y,p):
    return [(x-1)/15.,(y-1)/15.,SCALE*float(p)/5]

peaks = []
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    X = dataGrid.data_at_loc(k)[:,0]
    [peaks.append(to_point(x,y,X[p])) for p in peakGrid.data_at_loc(k)]

X = np.array(peaks)
#clustering = SpectralClustering(n_clusters=C,assign_labels="discretize",random_state=0).fit(X)

#clustering = AgglomerativeClustering(n_clusters=C,linkage='average').fit(X)

clustering = DBSCAN(eps=0.25, min_samples=5).fit(X)

#clustering = OPTICS().fit(X)
#clustering = Birch(branching_factor=5, n_clusters=None, threshold=0.5).fit(X)



"""
Plotting
"""
for i,k in enumerate(locations):
    x,y = dataGrid.coord(k)
    Y = dataGrid.data_at_loc(k,True)[:,1]
    if len(shifts) == len(locations):
        Y = Y + shifts[i]
    X = dataGrid.data_at_loc(k,True)[:,0]
    plt.plot(X,Y,label=str(k))
    for i in peakGrid.data_at_loc(k):
        try:
            L = str(clustering.labels_[peaks.index(to_point(x,y,X[i]))])
        except:
            L = "x"
        plt.text(X[i],Y[i],L)
plt.legend()

plt.show()
