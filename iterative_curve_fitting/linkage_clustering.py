"""
Link peaks together based on Q, FWHM, and grid location
to identify phase transitions.

Produces "layers" where a given peak propagates through a portion of the wafer.

3D and 2D visuals are produced

"""


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

data_dir = "/home/sasha/Desktop/iterative_curve_fitting_save_test/"
regex = """params_(?P<num>.*?).csv"""
peakGrid = DataGrid(data_dir,regex)


used_points = set() #dictionary of used points

"""
Get the adjacent peaks that "connect" to a given peak
"""
def get_adjacent_points(x,y,Q,I,FWHM):
    def dir(dx,dy):
        if not peakGrid.in_grid(x+dx,y+dy):
            return None
        i = np.argmin(np.abs(peakGrid.data_at(x+dx,y+dy)[:,1] - Q))
        P = peakGrid.data_at(x+dx,y+dy)[i]
        #FWHM = .005
        if Q-FWHM/4< P[1] and P[1] < Q+FWHM/4:
        #if I*.9 < P[0] and P[0] < I*1.1:
            return [x+dx,y+dy,P[1],P[0],P[5]]
        return None
    points = []
    for direction in [[0,-1],[0,1],[-1,0],[1,0]]:
        p = dir(*direction)
        if not p == None:
            points.append(p)
    return points

"""
Find the largest intensity peak that does not yet belong to a layer
"""
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

"""
Link a full layer and return associated peaks
"""
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

# 2D

"""
for i in range(100):
    layer = link_layer()
    max_I = np.max([P[3] for P in layer])
    if len(layer) < 10 or len(layer) == 177:
        continue
    xs = []
    ys = []
    qs = []
    for P in layer:
        xs.append(P[0])
        ys.append(P[1])
        qs.append(P[2])
        plt.scatter(xs,ys, color="red",marker='o',alpha=P[3]/max_I)
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.show()
    #plt.draw()
    #plt.pause(.3)
    #plt.cla()
#ax.set_xlim3d(0, 15)
#ax.set_ylim3d(0, 15)
#ax.set_zlim3d(0, 6)
plt.show()
"""

#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in range(100):

    layer = link_layer()
    #print(j)
    #if j < 43:
    #    continue
    # 11, 20, 28, 30, 43
    #if not j == 11:
    #  continue
    if not j in [11,20,28,30,43]:
        continue

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
#ax.set_zlim3d(0, 6)
plt.show()
    #plt.title(j)
    #plt.draw()
    #plt.pause(1)
    #plt.cla()
