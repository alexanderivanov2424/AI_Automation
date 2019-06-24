'''
Script for plotting spectra at several data values to
visually compare differences.
'''


from data_loading.data_grid import DataGrid


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


data_dir = "/home/sasha/Desktop/saveTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

# grid locations to plot
locations =[38,37,50,49,48]

#how much to shift each grid location vertically
#(makes it easier to see peaks)
shifts = [100*i for i,v in enumerate(locations)]



peaks = []
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    [peaks.append([x/100.,y/100.,float(p)]) for p in peakGrid.data[k][1:,1]]

C = 57
X = np.array(peaks)
'''
clustering = SpectralClustering(n_clusters=C,
        assign_labels="discretize",
        random_state=0).fit(X)
'''
clustering = AgglomerativeClustering(n_clusters=C,linkage='average').fit(X)


def similarity_vector(A,B):
    return np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)

#cosine similarity function using two grid positions
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    return similarity_vector(a,b)


for i,k in enumerate(locations):
    x,y = dataGrid.coord(k)
    Y = dataGrid.data[k][:,1]
    if len(shifts) == len(locations):
        Y = Y + shifts[i]
    X = dataGrid.data[k][:,0]
    plt.plot(X,Y,label=str(k))
    for xc in peakGrid.data[k][1:,1]:
        for i,X_nearest in enumerate(X):
            if X_nearest >= float(xc):
                try:
                    L = str(clustering.labels_[peaks.index([x/100.,y/100.,float(xc)])])
                except:
                    L = "x"
                plt.text(X_nearest,Y[i],L)
                break
plt.legend()


for i in range(len(locations)):
    for j in range(i+1,len(locations)):
        print("Similarity " + str(locations[i]) + "," + str(locations[j]) + ": " + str(similarity(locations[i],locations[j])))

plt.show()
