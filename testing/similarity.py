
from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import math



#folder with data files

dataGrid = DataGrid_TiNiSn_500C()


'''
for k in dataGrid.data.keys():
    x = dataGrid.data[k][:,1]
    plt.plot(x)
    peaks,_ = find_peaks(x,height=100)
    plt.plot(peaks,x[peaks], "x")
    plt.show()
'''

#cosine similarity function using two grid positions
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    pa, _ = find_peaks(a)
    pb, _ = find_peaks(b)
    p = np.append(pa,pb,axis=0)
    #cosine =  np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
    peaks = np.dot(a[p],b[p])/np.linalg.norm(a[p])/np.linalg.norm(b[p])
    return peaks

#create grid
grid = np.zeros(shape=(15,15))

#calculate similarity values for grid
for val in range(1,178):
    x,y = dataGrid.coord(val)
    #keys = ['up','left']
    keys = ['up', 'left', 'right', 'down']
    neigh = [dataGrid.neighbors(val)[k] for k in dataGrid.neighbors(val).keys() if k in keys]
    sim_values = [similarity(val,n) for n in neigh]
    if len(sim_values) == 0:
        grid[y-1][15-x] = 1
        continue
    grid[y-1][15-x] = np.min(sim_values)


def getPointsX(x,list):
    if len(list) == 0:
        return []
    dir = list.pop(0)
    if dir in ['u','d','s']:
        return [x] + getPointsX(x,list)
    elif dir == 'l':
        return [x-1] + getPointsX(x-1,list)
    else:
        return [x+1] + getPointsX(x+1,list)

def getPointsY(y,list):
    if len(list) == 0:
        return []
    dir = list.pop(0)
    if dir in ['l','r','s']:
        return [y] + getPointsY(y,list)
    elif dir == 'd':
        return [y-1] + getPointsY(y-1,list)
    else:
        return [y+1] + getPointsY(y+1,list)

l1 = list('suuluuulululllldldll')
l1x = getPointsX(9.5,l1.copy())
l1y = getPointsY(-.5,l1.copy())

l2 = list('sululull')
l2x = getPointsX(12.5,l2.copy())
l2y = getPointsY(1.5,l2.copy())

l3 = list('sululululllldldldl')
l3x = getPointsX(9.5,l3.copy())
l3y = getPointsY(4.5,l3.copy())

l4 = list('sululululuuuuuluulu')
l4x = getPointsX(13.5,l4.copy())
l4y = getPointsY(2.5,l4.copy())

l5 = list('sldl')
l5x = getPointsX(9.5,l5.copy())
l5y = getPointsY(8.5,l5.copy())

l6 = list('sullllld')
l6x = getPointsX(8.5,l6.copy())
l6y = getPointsY(8.5,l6.copy())

l7 = list('sulluuuu')
l7x = getPointsX(6.5,l7.copy())
l7y = getPointsY(9.5,l7.copy())

l8 = list('sdddldlll')
l8x = getPointsX(3.5,l8.copy())
l8y = getPointsY(13.5,l8.copy())

l9 = list('sddldl')
l9x = getPointsX(1.3,l9.copy())
l9y = getPointsY(9.5,l9.copy())

grid[grid==0] = np.nan
plt.figure(1)
plt.imshow(grid)
#x,y = dataGrid.coord(15)
#plt.plot([x-1],[y-1],marker='o', markersize=3, color="red")
#plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.plot(l1x,l1y,color="red")
plt.plot(l2x,l2y,color="red")
plt.plot(l3x,l3y,color="red")
plt.plot(l4x,l4y,color="red")
plt.plot(l5x,l5y,color="red")
plt.plot(l6x,l6y,color="red")
plt.plot(l7x,l7y,color="red")
plt.plot(l8x,l8y,color="red")
plt.plot(l9x,l9y,color="red")




from sklearn.cluster import AgglomerativeClustering

points = [[6,1]]
for val in range(2,178):
    x,y = dataGrid.coord(val)
    points = np.append(points,[[x,y]],axis=0)

size = len(points)
D = np.ones(shape=(size,size))
for x in range(size):
    for y in range(size):
        D[x,y] = 1 - similarity(x+1,y+1)

agg = AgglomerativeClustering(n_clusters=10, affinity='precomputed',linkage='complete')
agg.fit(D)

colors = [[.5,0,.5],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1],[.5,.5,.5],[.5,1,.5],[1,.5,0]]
#colors = ['red','gray','gold','chartreuse','lightseagreen','deepskyblue','navy','m','sandybrown','palevioletred']


cluster_grid = np.zeros(shape = (15,15,3))
for val in range(1,178):
    x,y = dataGrid.coord(val)
    cluster_grid[y-1][15-x] = colors[agg.labels_[val-1]]

plt.figure(2)
plt.imshow(cluster_grid)
plt.gca().invert_yaxis()
plt.show()

'''
label = 5
for i,l in enumerate(agg.labels_):
     if l == label:
         x = points[i][0]
         y = points[i][1]
         plt.plot(dataGrid.data_at(x,y)[:,1])
         plt.show()
'''


'''
#K-Means clustering

#general parameters
data = dataGrid.data
n_clusters = 3
X = np.stack([data[k][:,1].ravel() for k in data.keys()])


#K-Means with cosine distance metric

from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
kclusterer = KMeansClusterer(n_clusters, distance=cosine_distance, repeats=25)
kmeans = kclusterer.cluster(X, assign_clusters=True)

#fill grid with clusters for visual
k_grid = np.zeros(shape = (15,15))
for val in range(1,177):
    x,y = dataGrid.coord(val)
    k_grid[x-1][y-1] = kmeans[val] + 1


#K-Means with euclidean distance metric

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

#fill grid with clusters for visual
k_grid_2 = np.zeros(shape = (15,15))
for val in range(1,177):
    x,y = dataGrid.coord(val)
    k_grid_2[x-1][y-1] = kmeans[val] + 1


#plot both clustrings
plt.subplot(211)
plt.imshow(k_grid)

plt.subplot(212)
plt.imshow(k_grid_2)
plt.show()

'''
