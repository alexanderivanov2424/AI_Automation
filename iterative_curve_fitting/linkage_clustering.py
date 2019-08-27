"""
Link peaks together based on Q, FWHM, and grid location
to identify phase transitions.

Produces "layers" where a given peak propagates through a portion of the wafer.

3D and 2D visuals are produced

"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib
import colorsys

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import numpy as np

from data_loading.data_grid_TiNiSn import DataGrid


"""
Load Data and Peak Data
"""


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
def get_adjacent_points(x,y,Q_i):
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
        for i,P in enumerate(peakGrid.data_at_loc(loc)):
            if (x,y,i) in used_points:
                continue
            if P[0] < max_I:
                continue
            max_I = P[0]
            max_P = [x,y,i]
    return max_P,max_I

"""
Link a full layer and return associated peaks
"""
def link_layer():
    P0,max_I = find_max_peak()
    if P0 == []:
        return None,0
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
    return Layer,max_I



layer_list = []

while True:
    layer,max_I = link_layer()
    if layer == None:
        break
    if len(layer) == 1:
        continue
    layer_list.append((layer,max_I))
#layer_list = sorted(layer_list,key=lambda x:200 - len(x))
layer_list = [L for L in layer_list if len(L[0]) > 5]


peak_reduced = np.zeros((peakGrid.size,len(layer_list)))

for i,L in enumerate(layer_list):
    for P in L[0]:
        peak_reduced[peakGrid.grid_num(P[0],P[1])-1,i] = peakGrid.data_at(P[0],P[1])[P[2],0]

pca = PCA(n_components = 'mle',svd_solver='full').fit_transform(peak_reduced)
print(len(pca[0]))

def cluster_from_peak_reduced(i):
    agg = AgglomerativeClustering(n_clusters=i).fit(pca)

    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

    cluster_grid = np.zeros(shape = (15,15,3))
    for val in range(1,178):
        x,y = peakGrid.coord(val)
        cluster = agg.labels_[val-1]
        if cluster == -1:
            continue
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])
    return cluster_grid

for i in range(2,20):
    cg = cluster_from_peak_reduced(i)
    plt.imshow(cg)
    for j in range(peakGrid.size):
        x,y = peakGrid.coord(j+1)
        plt.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.title(i)
    plt.show()

"""
"""
# 2D
"""


#generate colors
N = len(layer_list)
HSV_tuples = [(x*1.0/N, 1, 1) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
np.random.shuffle(RGB_tuples)

#print(np.median([len(L) for L in layer_list]))
for i,layer_tuple in enumerate(layer_list):
    layer = layer_tuple[0]
    if i >= 36:
        continue
    plt.subplot(6,6, i+1)
    max_I = np.max([peakGrid.data_at(P[0],P[1])[P[2],0] for P in layer])
    xs = []
    ys = []
    qs = []
    for P in layer:
        xs.append(P[0])
        ys.append(P[1])
        qs.append(peakGrid.data_at(P[0],P[1])[P[2],1])
        I = peakGrid.data_at(P[0],P[1])[P[2],0]
        alpha = np.power(I/max_I,2)
        plt.scatter(xs,ys, color=RGB_tuples[i],marker='o',alpha=alpha)
    #print(max_I)
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    #plt.show()
    #plt.draw()
    #plt.pause(.01)
    #plt.cla()
plt.show()



#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for L in layer_list:
    layer = L[0]
    xs = []
    ys = []
    qs = []
    for P in layer:
        xs.append(P[0])
        ys.append(P[1])
        qs.append(peakGrid.data_at(P[0],P[1])[P[2],1])
    ax.scatter(xs,ys,qs, marker='o',alpha=1)
ax.set_xlim3d(0, 15)
ax.set_ylim3d(0, 15)
#ax.set_zlim3d(0, 6)
plt.show()
    #plt.title(j)
    #plt.draw()
    #plt.pause(1)
    #plt.cla()
"""
