
from data_loading.data_grid import DataGrid

import matplotlib.pyplot as plt
import numpy as np
import csv


def plotDataGrid(ax,sim_array,dataGrid):
    grid = np.zeros(shape=dataGrid.dims)
    for i,v in enumerate(sim_array):
        x,y = dataGrid.coord(i+1)
        grid[x-1][y-1] = v
    ax.imshow(trim_outside_grid(grid,dataGrid))

#Note empty data starts with a zero reading
def interpolateData(measurement_array,count,dataGrid):


    full_data = measurement_array.copy()

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
            while len(avg) < count:
                next_cur = set()
                for C in cur:
                    for K in dataGrid.neighbors(C).values():
                        if len(avg) >= count:
                            break
                        if not K in searched:
                            next_cur.add(K)
                            searched.add(K)
                            if no_data[K-1] == 0:
                                avg.add(K-1)
                searched = searched.union(cur)
                cur = next_cur

            full_data[i] = np.mean(measurement_array[list(avg)],axis=0)
    return full_data

def interpolateDataAvg(measurement_array):
    full_data = measurement_array.copy()
    avg = np.mean([x for x in full_data if not x[0] == 0.],axis=0)
    for i,x in enumerate(full_data):
        if x[0] == 0.:
            full_data[i] = avg
    return full_data

def similarity(V1,V2):
    return np.dot(V1,V2)/np.linalg.norm(V1)/np.linalg.norm(V2)

def getDissimilarityMatrix(measurement_array,dataGrid,keys = ['up', 'left', 'right', 'down']):
    #create grid
    grid = np.zeros(shape=dataGrid.dims)

    #calculate similarity values for grid
    for i,val in enumerate(measurement_array):
        x,y = dataGrid.coord(i+1)
        neigh = [dataGrid.neighbors(i+1)[k] for k in dataGrid.neighbors(i+1).keys() if k in keys]
        sim_values = [similarity(val,measurement_array[x-1]) for x in neigh]
        if len(sim_values) == 0:
            grid[x-1][y-1] = 1
            continue
        grid[x-1][y-1] = 1 - np.amin(sim_values)
    return grid

def clipSimilarityMatrix(data):
    min = np.min(data.ravel()[np.nonzero(data.ravel())])
    min_array = np.full(data.shape,min)
    return np.clip(data - min_array,0,1)

def trim_outside_grid(data,dataGrid):
    arr = data.copy()
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if not dataGrid.in_grid(x+1,y+1):
                arr[x,y] = np.nan
    return arr


def dict_to_csv(dict,path,file_name):
    with open(path + file_name + ".csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for k, v in dict.items():
           writer.writerow([k, v])

def csv_to_dict(path,file_name):
    with open(path + file_name + ".csv") as csv_file:
        reader = csv.reader(csv_file)
        dict = dict(reader)
        return dict
