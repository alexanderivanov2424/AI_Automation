'''
Script for plotting spectra at several data values to
visually compare differences.
'''


from data_loading.data_grid import DataGrid

from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math




regex_500 = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
regex_600 = """TiNiSn_600C_Y20190219_14x14_t65_(?P<num>.*?)_bkgdSub_1D.csv"""


data_path ="/path/to/data/here/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
data_path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"

dataGrid = DataGrid(data_path,regex_500)

data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.05/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

# grid locations to plot
locations =[117,45,94]
#locations = range(82,96+1)
#locations = [4,11,21,33,45,60,75,90,105,120,134,147,159,169,176]

#how much to shift each grid location vertically
#(makes it easier to see peaks)
shifts = [100*i for i,v in enumerate(locations)]
#shifts = [0,100,200,300,500,600,700,800,1000,1100,1200,1300,1500,1600,1700,1800]


"""
Load and Cluster peaks
"""

SCALE = 100
def to_point(x,y,p):
    return [(x-1)/15.,(y-1)/15.,SCALE*float(p)/5]

peaks = []
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    [peaks.append(to_point(x,y,p)) for p in peakGrid.data_at_loc(k)[:,1]]

X = np.array(peaks)

#clustering = SpectralClustering(n_clusters=C,assign_labels="discretize",random_state=0).fit(X)

#clustering = AgglomerativeClustering(n_clusters=C,linkage='average').fit(X)

clustering = DBSCAN(eps=0.25, min_samples=10).fit(X)

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
    for xc in peakGrid.data_at_loc(k)[:,1]:
        for i,X_nearest in enumerate(X):
            if X_nearest >= float(xc):
                try:
                    L = str(clustering.labels_[peaks.index(to_point(x,y,xc))])
                except:
                    L = "x"
                plt.text(X_nearest,Y[i],L)
                break
plt.legend()

plt.show()
