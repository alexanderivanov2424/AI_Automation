
from data_loading.data_grid_TiNiSn import DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

import warnings
warnings.simplefilter("ignore")



#folder with data files

dataGrid = DataGrid_TiNiSn_500C()


def similarity_vector(A,B):
    pA, _ = find_peaks(A)
    pB, _ = find_peaks(B)
    p = np.append(pA,pB,axis=0)
    cosine =  np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    #peaks = np.dot(A[p],B[p])/np.linalg.norm(A[p])/np.linalg.norm(B[p])
    return cosine

#cosine similarity function using two grid positions
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    return similarity_vector(a,b)

points = [[6,1]]
for val in range(2,178):
    x,y = dataGrid.coord(val)
    points = np.append(points,[[x,y]],axis=0)

size = len(points)
D = np.ones(shape=(size,size))
for x in range(size):
    for y in range(size):
        D[x,y] = 1 - similarity(x+1,y+1)


def get_averages(agg):
    grouped_data = [[] for x in range(i)]
    for loc,val in enumerate(agg.labels_):
        grouped_data[val].append(dataGrid.data_at_loc(loc+1)[:,1])

    averages = [np.nanmean(x,axis=0) for x in grouped_data]
    return averages

def get_avg_loc(agg,clusters,averages):
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


#calculate i clusters and create grid visuals and center points
base_clusters = 3
def get_cluster_grids(i):
    #generate all clustering up to desired amount

    hues = [float(float(x)/float(base_clusters)) for x in range(1,base_clusters+1)]

    pl_prev = None
    labels_prev = None
    for clusters in range(base_clusters,i+1):
        agg = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed',linkage='complete')
        agg.fit(D)
        avg = get_averages(agg)
        px,py,pl = get_avg_loc(agg,clusters,avg)

        if clusters == base_clusters:
            pl_prev = pl
            dict = {}
            for l,p in enumerate(pl):
                dict[agg.labels_[p-1]] = l
            labels_prev = [dict[agg.labels_[val]] for val in range(0,177)]
            continue
        new_centers = list(set(pl).difference(set(pl_prev)))
        k1 = new_centers[0]
        if len(new_centers) == 2:
            k2 = new_centers[1]
        else:
            k2 = pl_prev[labels_prev[k1-1]]
        parent = pl_prev[labels_prev[k1-1]]

        update_lists(hues,pl_prev,labels_prev,agg.labels_,k1,k2,parent)


    averages = get_averages(agg)
    points_x, points_y, points_loc = get_avg_loc(agg,i,averages)

    cluster_grid = np.zeros(shape = (15,15,3))
    cluster_grid_scale = np.zeros(shape = (15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        #cluster = agg.labels_[val-1]
        cluster = labels_prev[val-1]
        similarity = similarity_vector(dataGrid.data_at_loc(val)[:,1],averages[cluster])

        #similarity = math.pow(similarity,10)
        cluster_grid_scale[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,similarity])
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

    return cluster_grid, cluster_grid_scale, points_x, points_y, points_loc


start = 177
end = 177
cluster_range = range(start,end+1)

fig = plt.figure(figsize=(5,10))
ax = fig.subplots(nrows=3, ncols=1)

for i in cluster_range:
    cg, cgs, px, py, pl = get_cluster_grids(i)
    px = [15-x for x in px]
    py = [y-1 for y in py]
    ax[0].imshow(cg)
    ax[0].invert_yaxis()
    ax[0].axis("off")
    ax[0].title.set_text(i)

    ax[1].imshow(cgs)
    ax[1].scatter(px,py,s=3,c='black')
    ax[1].invert_yaxis()
    ax[1].axis("off")

    ax[2].imshow(cgs)
    ax[2].scatter(px,py,s=3,c='black')
    ax[2].invert_yaxis()
    ax[2].axis("off")
    for i,txt in enumerate(pl):
        ax[2].annotate(txt,(px[i],py[i]))
    fig.tight_layout()
    plt.draw()
    plt.pause(.001)

fig.tight_layout()
k=.01
plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
#plt.savefig("/home/sasha/Desktop/cluster_images/clustering-" + str(min) + "-" + str(max) + ".png")
plt.show()
