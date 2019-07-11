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
locations =[85,69,53,39]
#locations = range(82,96+1)
#locations = [4,11,21,33,45,60,75,90,105,120,134,147,159,169,176]
locations = [1]
#how much to shift each grid location vertically
#(makes it easier to see peaks)
shifts = [150*i for i,v in enumerate(locations)]
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

clustering = DBSCAN(eps=0.25, min_samples=5).fit(X)

#clustering = OPTICS().fit(X)

#clustering = Birch(branching_factor=5, n_clusters=None, threshold=0.5).fit(X)


"""
Create a map of each grid location to a Vector of peaks in C dimensions
"""
C = len(set(clustering.labels_).difference(set([-1])))
M = np.zeros(shape=(peakGrid.dims[0],dataGrid.dims[1],C))

i = 0
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    V = np.zeros(shape=C)
    for i,p in enumerate(peakGrid.data_at_loc(k)[:,1]):
        loc = clustering.labels_[peaks.index(to_point(x,y,p))]
        M[x-1,y-1,loc] = peakGrid.data_at_loc(k)[i,3]

"""
Similarity function
"""
def similarity(d1,d2):
    x,y = peakGrid.coord(d1)
    a = M[x-1,y-1]
    x,y = peakGrid.coord(d2)
    b = M[x-1,y-1]
    a[a==0] = 100000
    b[b==0] = 100000
    return 1/(np.mean(np.abs(a-b))+1)


"""
Print Pair-wise similarity
"""
for i in range(len(locations)):
    for j in range(i+1,len(locations)):
        print(str(locations[i]) + "  " + str(locations[j]) + ": " + str(similarity(i,j)))



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
