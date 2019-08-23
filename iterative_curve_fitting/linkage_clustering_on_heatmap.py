
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import colorsys

import numpy as np

from data_loading.data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C


"""
Load Data and Peak Data
"""
dataGrid = DataGrid_TiNiSn_500C()


data_dir = "/home/sasha/Desktop/iterative_curve_fitting_save_test/"
regex = """params_(?P<num>.*?).csv"""
peakGrid = DataGrid(data_dir,regex)


used_points = set() #dictionary of used points

total_peaks = 0
for loc in peakGrid.data.keys():
    total_peaks += len(peakGrid.data_at_loc(loc))

"""
Get the adjacent peaks that "connect" to a given peak
"""
def get_adjacent_points(x,y,Q_i,is_vert):
    def dir(dx,dy):
        if not peakGrid.in_grid(x+dx,y+dy):
            return None
        Q = peakGrid.data_at(x,y)[Q_i,1]
        FWHM = peakGrid.data_at(x,y)[Q_i,5]

        #all peaks except peak Q_i
        peakQ_in_pattern = np.append(peakGrid.data_at(x,y)[:Q_i-1,1],peakGrid.data_at(x,y)[Q_i:,1])
        seps = np.abs(peakQ_in_pattern -Q)
        min_sep = np.min(seps[np.nonzero(seps)])
        R = min(min_sep/2,FWHM/2) #range for linked peak
        #print(R)
        def dist(i):
            return np.abs(peakGrid.data_at(x+dx,y+dy)[i,1] - Q)
        neighbor_peaks = [[dist(i),i] for i in range(len(peakGrid.data_at(x+dx,y+dy))) if dist(i) <= R]
        if len(neighbor_peaks) == 0:
            return None
        neighbor_peaks = sorted(neighbor_peaks,key=lambda x:x[0])
        for peak in neighbor_peaks:
            i = peak[1]
            if (x+dx,y+dy,i) in used_points:
                return None
            return [x+dx,y+dy,i]
        return None

    points = []
    if is_vert:
        for direction in [[0,-1],[0,1]]:
            p = dir(*direction)
            if not p == None:
                points.append(p)
    else:
        for direction in [[-1,0],[1,0]]:
            p = dir(*direction)
            if not p == None:
                points.append(p)
    return points

"""
Find the largest intensity peak that does not yet belong to a layer
"""
def find_max_peak(locs):
    max_P = []
    max_I = 0
    for loc in locs:
        x,y = peakGrid.coord(loc)
        for i,P in enumerate(peakGrid.data_at_loc(loc)):
            if (x,y,i) in used_points:
                continue
            if P[0] < max_I:
                continue
            max_I = P[0]
            max_P = [x,y,i]
    return max_P

"""
Link a full layer and return associated peaks
"""
def link_layer(locs,is_vert):
    P0 = find_max_peak(locs)
    if P0 == []:
        return None
    Border = [P0]
    Layer = [P0]
    used_grid_locs = set()
    used_grid_locs.add((Border[0][0],Border[0][1]))

    while len(Border) > 0:
        new_Border = []
        for B in Border:
            points = get_adjacent_points(*B,is_vert)
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




V_list = [3,10,20,32,45,59,74,89,104,119,133,146,158,168,175]
H_list = [82,83,84,85,86,87,88,89,90,91,92,93,94,95,96]
locations = V_list

layer_labels = {}
layer_num = 0
while len(layer_labels.keys()) < total_peaks:
    layer = link_layer(locations,True)
    if layer == None:
        break
    if len(layer) == 1:
        continue
    for P in layer:
        layer_labels[(P[0],P[1],P[2])] = layer_num
    layer_num += 1

#generate colors
N = layer_num
HSV_tuples = [(x*1.0/N, 1, 1) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
np.random.shuffle(RGB_tuples)

#generate heat plot for background
lst = []
for L in locations:
        lst.append(dataGrid.data[L][:,1])
im = np.array(lst)
im_log = np.log(im + 1)
im_sqrt = np.sqrt(im)

SCALE = 10
plt.imshow(np.repeat(im_log,SCALE,axis=0))
for i,L in enumerate(locations):
    X = dataGrid.data_at_loc(L)[:,0]
    for j,peak in enumerate(peakGrid.data_at_loc(L)):
        x,y = peakGrid.coord(L)
        try:
            color = layer_labels[(x,y,j)]
            p =np.argmin(np.abs(X - peak[1]))
            plt.scatter(p,i*SCALE + SCALE//2,marker='o',color=RGB_tuples[color],s=30)
        except:
            p =np.argmin(np.abs(X - peak[1]))
            plt.scatter(p,i*SCALE + SCALE//2,marker='x',color="red",s=30)

plt.gca().invert_yaxis()
#plt.xlim((0,400))
plt.show()
