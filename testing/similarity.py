
from data_loading.data_grid import DataGrid

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import math



#folder with data files
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
dataGrid = DataGrid(path)


#cosine similarity function using two grid positions
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)

#create grid
grid = np.zeros(shape=(15,15))

#calculate similarity values for grid
for val in range(1,177):
    x,y = dataGrid.coord(val)
    #keys = ['up','left']
    keys = ['up', 'left', 'right', 'down']
    neigh = [dataGrid.neighbors(val)[k] for k in dataGrid.neighbors(val).keys() if k in keys]
    sim_values = [similarity(val,x) for x in neigh]
    if len(sim_values) == 0:
        grid[x-1][y-1] = 1
        continue
    grid[x-1][y-1] = 1 - np.amin(sim_values)


#normalize grid based on min
# values range (0, max - min]
#this shows the relitive similarity values
min = np.min(grid.ravel()[np.nonzero(grid.ravel())])
min_array = np.full(grid.shape,min*.99)
#grid = np.clip(grid - min_array,0,1)
grid = np.clip(grid,min,1)

#show similarity plot
plt.imshow(grid)
plt.show()


'''
#K-Means clustering

#general parameters
n_clusters = 2
X = np.stack([data[k][:,1].ravel() for k in data.keys()])


#K-Means with cosine distance metric

from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
kclusterer = KMeansClusterer(n_clusters, distance=cosine_distance, repeats=25)
kmeans = kclusterer.cluster(X, assign_clusters=True)

#fill grid with clusters for visual
k_grid = np.zeros(shape = (15,15))
for val in range(1,177):
    x,y = coord(val)
    k_grid[x-1][y-1] = kmeans[val] + 1


#K-Means with euclidean distance metric

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

#fill grid with clusters for visual
k_grid_2 = np.zeros(shape = (15,15))
for val in range(1,177):
    x,y = coord(val)
    k_grid_2[x-1][y-1] = kmeans[val] + 1


#plot both clustrings
plt.subplot(211)
plt.imshow(k_grid)

plt.subplot(212)
plt.imshow(k_grid_2)
plt.show()
'''
