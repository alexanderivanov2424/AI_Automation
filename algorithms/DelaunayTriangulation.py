#Delaunay Triangulation

from data_loading.data_grid import DataGrid
from utils.utils import plotDataGrid, interpolateData, similarity, interpolateDataAvg
from utils.utils import getSimilarityMatrix, clipSimilarityMatrix

from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
import imageio
import numpy as np
import time
import math
import random



#seed algorithm
seed = 0
np.random.seed(seed)

#set up DataGrid object
path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
dataGrid = DataGrid(data_path,regex)


#Plotting
fig, ax = plt.subplots(nrows=2, ncols=2)
true_data = clipSimilarityMatrix(getSimilarityMatrix(dataGrid.get_data_array(),dataGrid))
exp_data = np.zeros(true_data.shape)

#CONSTANTS
NUMBER_OF_SAMPLES = 100
p = .2

#DATA STRUCTURES
M = np.empty(shape=(dataGrid.size,dataGrid.data_length))
S = set()

tri = None

lengths = {}
similarities = {}


def key_from_edge(a,b):
    x1 = int(tri.points[a][0])
    y1 = int(tri.points[a][1])
    d1 = dataGrid.grid_num(x1+1,y1+1)
    x2 = int(tri.points[b][0])
    y2 = int(tri.points[b][1])
    d2 = dataGrid.grid_num(x2+1,y2+1)
    key = ""
    if d2 < d1:
        key = str(d2) + "," + str(d1)
    else:
        key = str(d1) + "," + str(d2)
    return key, d1, d2

def add_edge(a,b):
    key, d1, d2 = key_from_edge(a,b)
    if key in lengths:
        return
    lengths[key] = math.sqrt(np.sum(np.square(tri.points[a] - tri.points[b])))
    similarities[key] = (1 - similarity(M[d1-1],M[d2-1]))**10

def update_edge_values():
    for x in tri.simplices:
        add_edge(x[0],x[1])
        add_edge(x[1],x[2])
        add_edge(x[0],x[2])

def edge_center(a,b):
    return (np.add(tri.points[a],tri.points[b]))//2



Initial_Points = []
C_list = [1,14,164,172]
for C in C_list:
    x,y = dataGrid.coord(C)
    M[C-1] = dataGrid.data_at(x,y)[:,1]
    Initial_Points.append((x-1,y-1))
    S.add(C)


tri = Delaunay(Initial_Points,incremental=True)

update_edge_values()

while len(S) < NUMBER_OF_SAMPLES:
    P = 1
    if random.random() < p:
        cell = np.random.choice(range(1,dataGrid.size+1),1)
        while cell[0] in S:
            cell = np.random.choice(range(1,dataGrid.size+1),1)
        P = cell[0]
    else:
        max_v = -1000000
        max_p1 = None
        max_p2 = None
        for x in tri.simplices:
            for e in [(x[0],x[1]),(x[1],x[2]),(x[0],x[2])]:
                k, _, _ = key_from_edge(e[0],e[1])
                v = similarities[k]/lengths[k]
                if v > max_v:
                    max_p1 = e[0]
                    max_p2 = e[1]
                    max_v = v

        center = edge_center(max_p1,max_p2)
        P = dataGrid.grid_num(int(center[0]),int(center[1]))
        if P in S or P < 1 or P >= dataGrid.size:
            cell = np.random.choice(range(1,dataGrid.size+1),1)
            while cell[0] in S:
                cell = np.random.choice(range(1,dataGrid.size+1),1)
            P = cell[0]

    M[P-1] = dataGrid.data_at_loc(P)[:,1]
    S.add(P)
    x,y = dataGrid.coord(P)
    tri.add_points([(x-1,y-1)])
    update_edge_values()


    #Plotting
    full_data = interpolateData(M,3,dataGrid)
    exp_data = clipSimilarityMatrix(getSimilarityMatrix(full_data,dataGrid))

    ax[0,1].imshow(exp_data)
    ax[1,1].imshow(true_data)

    measured_points = np.zeros(dataGrid.dims)
    for s in S:
        x,y = dataGrid.coord(s)
        measured_points[x-1,y-1] = 1
    ax[1,0].imshow(measured_points)


    plt.draw()
    plt.pause(.01)

plt.show()
