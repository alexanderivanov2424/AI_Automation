'''
UTILS

General class to hold utility methods.
 - interpolation
 - dissimilarity matrix
 - saving/loading dict
'''
from data_loading.data_grid import DataGrid

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

def interpolateDataCubic(measurement_array,dataGrid):
    """
    Interpolate measurement array with cubic splines along
    each of the spectra dimensions.

    dataGrid is passed to find spectra neighbors

    Note: Values outside the bounds are filled with the nearest neighbor
    Note: Emtpy data is assumed to start with 0
    """
    full_data = measurement_array.copy()

    dims = dataGrid.dims
    grid_x,grid_y = np.mgrid[0:dims[0],0:dims[1]]

    points = []
    values = []

    # get x,y coordinates for each spectra
    for i,val in enumerate(full_data):
        if not val[0] == 0.:
            x,y = dataGrid.coord(i+1)
            points.append([x-1,y-1])
            values.append(val)

    #interpolate slices
    #each slice is an interpolation over the entire grid in a
    #single spectra dimention
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
    """
    Interpolate measurement array with nearest neighbor method along
    each of the spectra dimensions.

    dataGrid is passed to find spectra neighbors

    Note: Emtpy data is assumed to start with 0
    """
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


#interpolate a measurement array with linear splines
def interpolateDataLinear(measurement_array,dataGrid):
    """
    Interpolate measurement array with linear splines along
    each of the spectra dimensions.

    dataGrid is passed to find spectra neighbors

    Note: Values outside the bounds are filled with the nearest neighbor
    Note: Emtpy data is assumed to start with 0
    """
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
    """
    Interpolate measurement array by replacing empty values with average

    dataGrid is passed to find spectra neighbors

    Note: Emtpy data is assumed to start with 0
    """
    full_data = measurement_array.copy()
    avg = np.mean([x for x in full_data if np.any(x)],axis=0)
    for i,x in enumerate(full_data):
        if x[0] == 0.:
            full_data[i] = avg
    return full_data

#generate a dissimilarity matrix from a measurement array
#keys - the directions in which similarity is computed
def getDissimilarityMatrix(M, metric, dataGrid, keys = ['up', 'left', 'right', 'down']):
    """
    Generate dissimilarity matrix from measurement arry.

    # M - measurement array
    # metric - similarity metric object
    # dataGrid - DataGrid object
    # keys - [optional] specify specific neighbors to consider for similarity
    """
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
    """
    trim grid border
    the values outside the dataGrid are set to nan
    """
    arr = data.copy()
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if not dataGrid.in_grid(x+1,y+1):
                arr[x,y] = np.nan
    return arr

def dict_to_csv(dict,path):
    """
    write a dictionary to a file
    """
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for k, v in dict.items():
           writer.writerow([k, v])

def csv_to_dict(path,file_name):
    """
    load a dictionary from a file
    """
    with open(path + file_name + ".csv") as csv_file:
        reader = csv.reader(csv_file)
        dict = dict(reader)
        return dict
