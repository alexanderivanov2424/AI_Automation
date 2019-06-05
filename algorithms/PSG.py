
from data_loading.data_grid import DataGrid
from utils.utils import plotDataGrid, interpolateData


from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import numpy as np




#folder with data files
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
dataGrid = DataGrid(path)

fig, ax = plt.subplots(nrows=1, ncols=2)

M = np.empty(shape=(dataGrid.size,dataGrid.data_length))
G = np.full(shape=dataGrid.size,fill_value = .99)
S = set()


#cosine similarity function using two grid positions
def similarity(d1,d2):
    return np.dot(M[d1],M[d2])/np.linalg.norm(M[d1])/np.linalg.norm(M[d2])

#Note: cells numbering starts at 1
data_range = range(1,dataGrid.size+1)

def blur(G):
    T = np.zeros(shape=dataGrid.dims)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i)
        T[x-1][y-1] = v
    T_blurred = gaussian_filter(T, sigma=5)
    final = np.empty(shape=G.shape)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i)
        final[i] = T_blurred[x-1][y-1]
    return final

for n in range(20):

    MIN = np.full(shape=G.shape,fill_value = .95)
    blurred = blur(G-MIN)
    G_norm = blurred / np.sum(blurred)

    cells = np.random.choice(data_range, 1, p=G_norm)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=G_norm)

    C = cells[0]
    S.add(C)

    M[C-1] = dataGrid.data_at_loc(C)[:,1]  #"taking a measurement"

    sim_list = []

    for K in dataGrid.neighbors(C).values():
        if not K in S:
            M[K-1] = dataGrid.data_at_loc(K)[:,1]  #"taking a measurement"
            S.add(K)
        sim_list = [similarity(C-1,K-1)] + sim_list


    G[C-1] = max(sim_list)

    plotDataGrid(ax[0],G,dataGrid)
    plotDataGrid(ax[1],G_norm,dataGrid)
    plt.draw()
    plt.pause(.001)

full_data = interpolateData(M,dataGrid)
