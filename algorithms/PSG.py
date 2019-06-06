
from data_loading.data_grid import DataGrid
from utils.utils import plotDataGrid, interpolateData, similarity
from utils.utils import getSimilarityMatrix, clipSimilarityMatrix


from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import numpy as np




#folder with data files
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
dataGrid = DataGrid(path)

fig, ax = plt.subplots(nrows=2, ncols=3)

k = .95

power = 12

blur_const = 3

M = np.empty(shape=(dataGrid.size,dataGrid.data_length))
G = np.full(shape=dataGrid.size,fill_value = .1)
S = set()


#cosine similarity function using two grid positions
def get_similarity(d1,d2):
    return similarity(M[d1],M[d2])


def blur(G):
    T = np.zeros(shape=dataGrid.dims)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i)
        T[x-1][y-1] = v
    T_blurred = gaussian_filter(T, sigma=blur_const)
    final = np.empty(shape=G.shape)
    for i,v in enumerate(G):
        x,y = dataGrid.coord(i)
        final[i] = T_blurred[x-1][y-1]
    return final

for n in range(30):

    blurred = blur(np.power(G,power))
    G_norm = blurred / np.sum(blurred)

    #Note: cells numbering starts at 1
    data_range = range(1,dataGrid.size+1)

    cells = np.random.choice(data_range, 1, p=G_norm)
    while cells[0] in S:
        cells = np.random.choice(data_range, 1, p=G_norm)

    C = cells[0]
    S.add(C)

    M[C-1] = dataGrid.data_at_loc(C)[:,1]  #"taking a measurement"

    def update(C):
        sim_list = []
        for K in dataGrid.neighbors(C).values():
            if not K in S:
                M[K-1] = dataGrid.data_at_loc(K)[:,1]  #"taking a measurement"
                S.add(K)
            sim_list = [get_similarity(C-1,K-1)] + sim_list
        G[C-1] = max(sim_list)

    update(C)

    for K in dataGrid.neighbors(C).values():
        sur = True
        for K_2 in dataGrid.neighbors(K).values():
            if not K_2 in S:
                sur = False
                break
        if sur:
            update(K)

    plotDataGrid(ax[0,0],np.power(G,power),dataGrid)
    plotDataGrid(ax[0,1],G_norm,dataGrid)

    full_data = interpolateData(M,dataGrid)
    exp_data = clipSimilarityMatrix(getSimilarityMatrix(full_data,dataGrid))
    ax[0,2].imshow(exp_data)

    measured_points = np.zeros(dataGrid.dims)
    for s in S:
        x,y = dataGrid.coord(s)
        measured_points[x-1,y-1] = 1
    ax[1,0].imshow(measured_points)

    plt.draw()
    plt.pause(.5)

full_data = interpolateData(M,dataGrid)
exp_data = clipSimilarityMatrix(getSimilarityMatrix(full_data,dataGrid))
true_data = clipSimilarityMatrix(getSimilarityMatrix(dataGrid.get_data_array(),dataGrid))

print(np.square(np.subtract(exp_data, true_data)).mean())

plt.figure()
plt.imshow(exp_data)

plt.figure()
plt.imshow(true_data)
plt.show()
