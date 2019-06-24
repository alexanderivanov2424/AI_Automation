

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import SpectralClustering

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import math




dataGrid = DataGrid_TiNiSn_500C()

data_dir = "/home/sasha/Desktop/saveTest/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""
peakGrid = DataGrid(data_dir,regex)

start = 140
stop = 152

size = 10
G = np.zeros(shape=((stop-start+1)*size,dataGrid.data_length))

for l,i in enumerate(range(start,stop+1)):
    G[l*size:(l+1)*size] = np.log(dataGrid.data_at_loc(i)[:,1]+100)
#plt.imshow(G)
#plt.show()


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



M = np.zeros(shape=(peakGrid.dims[0],dataGrid.dims[1],C))

i = 0
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    V = np.zeros(shape=C)
    for peak_int in peakGrid.data[k][1:,3]:
        V[clustering.labels_[i]] = peak_int
        i+=1

    M[x-1,y-1] = V
'''
i = 0
for k in dataGrid.data.keys():
    plt.plot(dataGrid.data[k][:,0],dataGrid.data[k][:,1])
    print(peakGrid.data[k][1:,1])
    x,y = peakGrid.coord(k)
    print(clustering.labels_[i:i+len(peakGrid.data[k])])
    plt.show()
'''

def similarity(d1,d2):
    x,y = peakGrid.coord(d1)
    a = M[x-1,y-1]
    x,y = peakGrid.coord(d2)
    b = M[x-1,y-1]
    a = np.log(a+1)
    b = np.log(b+1)
    return np.mean(np.abs(a-b))
    #return math.sqrt(np.sum(np.square(a-b)))
    #return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)


size = peakGrid.size

K_Matrix = np.zeros(shape=(size,size))
for x in range(1,size+1):
    for N in peakGrid.neighbors(x).values():
        K_Matrix[x-1,N-1] = 1


D = np.ones(shape=(size,size))
for x in range(size):
    for y in range(size):
        D[x,y] = similarity(x+1,y+1)

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

for i in range(3,15):
    cg = get_cluster_grids(i)
    plt.imshow(cg)
    for j in range(dataGrid.size):
        x,y = dataGrid.coord(j+1)
        plt.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title(i)
    plt.show()
