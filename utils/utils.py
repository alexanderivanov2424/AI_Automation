
from data_loading.data_grid import DataGrid

import matplotlib.pyplot as plt
import numpy as np



def plotDataGrid(ax,sim_array,dataGrid):
    grid = np.zeros(shape=dataGrid.dims)
    for i,v in enumerate(sim_array):
        x,y = dataGrid.coord(i)
        grid[x-1][y-1] = v
    ax.imshow(grid)

#Note empty data starts with a zero reading
def interpolateData(measurement_array,dataGrid):
    full_data = np.empty(shape=measurement_array.shape)
    no_data = np.zeros(shape=len(measurement_array))
    for i,x in enumerate(measurement_array):
        if x[0] == 0.:
            no_data[i] = 1

    searched = set() #indexed at 1
    cur = set() #indexed at 1
    avg = set() # indexed at 0
    for i,x in enumerate(no_data):
        if x == 1:
            searched.clear()
            cur.clear()
            avg.clear()
            cur.add(i+1)
            while len(avg) < 4:
                next_cur = set()
                for C in cur:
                    for K in dataGrid.neighbors(C).values():
                        if not K in searched:
                            next_cur.add(K)
                            searched.add(K)
                            if no_data[K-1] == 0:
                                avg.add(K-1)
                searched = searched.union(cur)
                cur = next_cur

            full_data[i] = np.mean(measurement_array[list(avg)],axis=0)
    return full_data


def getSimilarityMatrix(measurement_array,dataGrid)
