
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

data_dir = "/home/sasha/Desktop/iterative_curve_fitting_save_test/"
regex = """params_(?P<num>.*?).csv"""
peakGrid = DataGrid(data_dir,regex)


used_points = set() #dictionary of used points


def get_adjacent_points(x,y,Q,I,FWHM):
    def dir(dx,dy):
        if not peakGrid.in_grid(x+dx,y+dy):
            return None
        i = np.argmin(peakGrid.data_at(x+dx,y+dy)[:,2] - Q)
        P = peakGrid.data_at(x+dx,y+dy)[i]
        if Q-.5 < P[1] and P[1] < Q+.5:# and I * .9 < P[0] and P[0] < I*1.1:
            return [x+dx,y+dy,P[1],P[0],P[5]]
        return None
    points = []
    for direction in [[0,-1],[0,1],[-1,0],[1,0]]:
        p = dir(*direction)
        if not p == None:
            points.append(p)
    return points


def find_max_peak():
    max_P = []
    max_I = 0
    for loc in peakGrid.data.keys():
        x,y = peakGrid.coord(loc)
        for P in peakGrid.data_at_loc(loc):
            if (x,y,P[1]) in used_points:
                continue
            if P[0] < max_I:
                continue
            max_I = P[0]
            max_P = [x,y,P[1],P[0],P[5]]
    return max_P


def link_layer():
    P0 = find_max_peak()
    Border = [P0]
    Layer = [P0]
    used_grid_locs = set()
    used_grid_locs.add((Border[0][0],Border[0][1]))

    while len(Border) > 0:
        new_Border = []
        for B in Border:
            points = get_adjacent_points(*B)
            for point in points:
                if (point[0],point[1]) in used_grid_locs:
                    continue
                new_Border.append(point)
                Layer.append(point)
                used_grid_locs.add((point[0],point[1]))
        Border = new_Border

    for point in Layer:
        used_points.add((point[0],point[1],point[2]))
    return Layer
    #add to used points


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    layer = link_layer()
    xs = []
    ys = []
    qs = []
    for P in layer:
        xs.append(P[0])
        ys.append(P[1])
        qs.append(P[2])
    ax.scatter(xs,ys,qs, marker='o',alpha=1)



ax.set_xlim3d(0, 15)
ax.set_ylim3d(0, 15)
ax.set_zlim3d(0, 6)
plt.show()
