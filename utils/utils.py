
from data_loading.data_grid import DataGrid

from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

#Note empty data starts with a zero reading
def interpolateDataCubic(measurement_array,dataGrid):
    full_data = measurement_array.copy()

    dims = dataGrid.dims
    grid_x,grid_y = np.mgrid[0:dims[0],0:dims[1]]

    points = []
    values = []

    for i,val in enumerate(full_data):
        if not val[0] == 0.:
            x,y = dataGrid.coord(i+1)
            points.append([x-1,y-1])
            values.append(val)
    values = np.array(values)
    for i in range(dataGrid.data_length):
        data = griddata(points, values[:,i], (grid_x, grid_y), method='cubic')
        nearest = griddata(points, values[:,i], (grid_x, grid_y), method='nearest')
        for j in range(len(full_data)):
            x,y = dataGrid.coord(j+1)
            v = data[x-1,y-1]
            if v != v or v == np.NINF or v == np.inf:
                v = nearest[x-1,y-1]
            if v != v or v == np.NINF or v == np.inf:
                v = 0
            full_data[j,i] = v
    return full_data

def interpolateDataNearest(measurement_array,dataGrid):
    full_data = measurement_array.copy()

    dims = dataGrid.dims
    grid_x,grid_y = np.mgrid[0:dims[0],0:dims[1]]

    points = []
    values = []

    for i,val in enumerate(full_data):
        if not val[0] == 0.:
            x,y = dataGrid.coord(i+1)
            points.append([x-1,y-1])
            values.append(val)
    values = np.array(values)

    for i in range(dataGrid.data_length):
        data = griddata(points, values[:,i], (grid_x, grid_y), method='nearest')
        for j in range(len(full_data)):
            x,y = dataGrid.coord(j+1)
            v = data[x-1,y-1]
            if v != v or v == np.NINF or v == np.inf:
                v = 0
            full_data[j,i] = v
    return full_data

def interpolateDataLinear(measurement_array,dataGrid):
    full_data = measurement_array.copy()

    dims = dataGrid.dims
    grid_x,grid_y = np.mgrid[0:dims[0],0:dims[1]]
    points = []
    values = []
    for i,val in enumerate(full_data):
        if not val[0] == 0.:
            x,y = dataGrid.coord(i+1)
            points.append([x-1,y-1])
            values.append(val)
    values = np.array(values)

    for i in range(dataGrid.data_length):
        data = griddata(points, values[:,i], (grid_x, grid_y), method='linear')
        nearest = griddata(points, values[:,i], (grid_x, grid_y), method='nearest')
        for j in range(len(full_data)):
            x,y = dataGrid.coord(j+1)
            v = data[x-1,y-1]
            if v != v or v == np.NINF or v == np.inf:
                v = nearest[x-1,y-1]
            if v != v or v == np.NINF or v == np.inf:
                v = 0
            full_data[j,i] = v
    return full_data

def interpolateDataAvg(measurement_array):
    full_data = measurement_array.copy()
    avg = np.mean([x for x in full_data if not x[0] == 0.],axis=0)
    for i,x in enumerate(full_data):
        if x[0] == 0.:
            full_data[i] = avg
    return full_data

def getDissimilarityMatrix(M, metric, dataGrid, keys = ['up', 'left', 'right', 'down']):
    grid = np.zeros(shape=dataGrid.dims)

    #calculate similarity values for grid
    for i,val in enumerate(M):
        x,y = dataGrid.coord(i+1)
        neigh = [dataGrid.neighbors(i+1)[k] for k in dataGrid.neighbors(i+1).keys() if k in keys]
        sims = [metric.similarity(val,M[x-1]) for x in neigh]
        if len(sims) == 0:
            grid[x-1][y-1] = 1
            continue
        grid[x-1][y-1] = 1 - np.amin(sims)
    return grid

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
