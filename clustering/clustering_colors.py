
from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

import warnings
warnings.simplefilter("ignore")

# LOAD DATA
dataGrid = DataGrid_TiNiSn_500C()

"""cosine similarity of two vectors"""
def similarity_vector(A,B):
    cosine =  np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    #return math.sqrt(np.sum(np.square((A-B))))/np.linalg.norm(A)/np.linalg.norm(B)
    #return np.sum(np.abs(A-B))
    return cosine

"""cosine similarity function using two grid positions"""
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    #return np.mean(np.abs(a-b))
    #return math.sqrt(np.sum(np.square(a-b)))
    return similarity_vector(a,b)

points = [[6,1]]
for val in range(2,178):
    x,y = dataGrid.coord(val)
    points = np.append(points,[[x,y]],axis=0)

size = len(points)
D = np.ones(shape=(size,size))
for x in range(size):
    for y in range(size):
        D[x,y] = 1-similarity(x+1,y+1)


def get_averages(agg,clusters):
    """
    Get the everage values for each cluster
    """
    grouped_data = [[] for x in range(clusters)]
    for loc,val in enumerate(agg.labels_):
        grouped_data[val].append(dataGrid.data_at_loc(loc+1)[:,1])

    averages = [np.nanmean(x,axis=0) for x in grouped_data]
    return averages

def get_avg_loc(agg,clusters,averages):
    """
    Get the locations of the average points in the clustering
    """
    points_x = [-1 for x in range(clusters)]
    points_y = [-1 for x in range(clusters)]
    points_loc = [-1 for x in range(clusters)]

    for loc in range(1,178):
        cluster = agg.labels_[loc-1]
        cur_x = points_x[cluster]
        cur_y = points_y[cluster]
        if cur_x == -1:
            x,y = dataGrid.coord(loc)
            points_x[cluster] = x
            points_y[cluster] = y
            points_loc[cluster] = loc
            continue
        sim_cur = similarity_vector(dataGrid.data_at(cur_x,cur_y)[:,1],averages[cluster])
        sim_new = similarity_vector(dataGrid.data_at_loc(loc)[:,1],averages[cluster])
        if sim_cur < sim_new:
            x,y = dataGrid.coord(loc)
            points_x[cluster] = x
            points_y[cluster] = y
            points_loc[cluster] = loc
    return points_x, points_y, points_loc

def update_lists(hue,points,labels,labels_new,k1,k2,parent):
    """
    Figure out what order k1, k2 need to replace the parent.
    Based on this order compute the new hues and update "hue"
    Based on this order compute the new labels and update "labels"
    """
    i = points.index(parent)
    prev = points[(i-1) % len(points)]
    next = points[(i+1) % len(points)]
    hue_prev = hue[(i-1) % len(points)]
    hue_next = hue[(i+1) % len(points)]
    hue_delta = min(abs(hue_next-hue_prev),abs(hue_next-hue_prev+1.0))

    points.remove(parent)
    hue.remove(hue[i])

    if similarity(prev,k1) > similarity(prev,k2):
        a = similarity(prev,k1)
        b = similarity(k1,k2)
        c = similarity(k2,next)
        hue_k1 = (hue_prev + a * hue_delta/(a+b+c)) % 1.
        hue_k2 = (hue_prev + (a+b) * hue_delta/(a+b+c)) % 1.
        points.insert(i,k2)
        hue.insert(i,hue_k2)

        points.insert(i,k1)
        hue.insert(i,hue_k1)

        for val in range(1,178):
            if labels[val-1] == i:
                if labels_new[val-1] == labels_new[k1-1]:
                    labels[val-1] = i
                else:
                    labels[val-1] = (i+1)% len(points)
            else:
                if labels[val-1] > i:
                    labels[val-1] += 1
                labels[val-1] = (labels[val-1])% len(points)

    else:
        a = similarity(prev,k2)
        b = similarity(k2,k1)
        c = similarity(k1,next)
        hue_k2 = (hue_prev + a * hue_delta / (a+b+c)) % 1.
        hue_k1 = (hue_prev + (a+b) * hue_delta / (a+b+c)) % 1.
        points.insert(i,k1)
        hue.insert(i,hue_k1)

        points.insert(i,k2)
        hue.insert(i,hue_k2)

        for val in range(1,178):
            if labels[val-1] == i:
                if labels_new[val-1] == labels_new[k2-1]:
                    labels[val-1] = i
                else:
                    labels[val-1] = (i+1)% len(points)
            else:
                if labels[val-1] > i:
                    labels[val-1] += 1
                labels[val-1] = (labels[val-1])% len(points)

#the base number of clusters where the colors are initialized with equal spread
#does not work with 2 or 1 because of update_lists()
base_clusters = 3
def get_cluster_grids(i):
    """
    generate all clustering up to desired amount
    returns visuals in list of arrays were each index is for a cluster
    """

    #save intermitent data
    list_cluster_grid = []
    list_px = []
    list_py = []
    list_pl = []


    hues = [float(float(x)/float(base_clusters)) for x in range(1,base_clusters+1)]

    pl_prev = None
    labels_prev = None
    for clusters in range(base_clusters,i+1):
        agg = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed',linkage='complete')
        agg.fit(D)
        avg = get_averages(agg,clusters)
        px,py,pl = get_avg_loc(agg,clusters,avg)
        if clusters == base_clusters:
            pl_prev = pl
            dict = {}
            for l,p in enumerate(pl):
                dict[agg.labels_[p-1]] = l
            labels_prev = [dict[agg.labels_[val]] for val in range(0,177)]
        else:
            new_centers = list(set(pl).difference(set(pl_prev)))
            k1 = new_centers[0]
            if len(new_centers) == 2:
                k2 = new_centers[1]
            else:
                k2 = pl_prev[labels_prev[k1-1]]
            parent = pl_prev[labels_prev[k1-1]]

            update_lists(hues,pl_prev,labels_prev,agg.labels_,k1,k2,parent)

        cluster_grid = np.zeros(shape = (15,15,3))
        for val in range(1,178):
            x,y = dataGrid.coord(val)
            cluster = labels_prev[val-1]
            cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

        list_cluster_grid.append(cluster_grid)
        list_px.append(px)
        list_py.append(py)
        list_pl.append(pl)
    return list_cluster_grid, list_px, list_py, list_pl


start = 177
end = 177
cluster_range = range(start,end+1)
if start == end:
    cluster_range = [start,start]

fig = plt.figure()
ax = fig.subplots(nrows=2, ncols=len(cluster_range))

list_cg, list_px, list_py, list_pl = get_cluster_grids(end)

for n,i in enumerate(cluster_range):
    px = [15-x for x in list_px[i-3]]
    py = [y-1 for y in list_py[i-3]]
    ax[0,n].imshow(list_cg[i-3])
    ax[0,n].invert_yaxis()
    ax[0,n].axis("off")
    ax[0,n].title.set_text(i)

fig.tight_layout()
k=.01
plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
#plt.savefig("/home/sasha/Desktop/cluster_images/clustering-" + str(min) + "-" + str(max) + ".png")
plt.show()
