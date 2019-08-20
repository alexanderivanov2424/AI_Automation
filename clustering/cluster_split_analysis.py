'''
Script for analyzing the differences between clusters and
for determining relative similarity between cluster centers

The produced visual is saved as a PNG file.

Cluster labels and similarity to cluster center is computed
for all grid locations and saved to an excel file.
'''


## TODO:  ADD AVERAGE TO PLOT


from data_loading.data_grid import DataGrid


from sklearn.cluster import AgglomerativeClustering
from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math


# PLOT X clusters going to (X+1) clusters
clusters = 10

#Adjust the similarity gradient
power = 10

#Adjust the size of the figure
figsize = (14,8)

#if the colors should match up between plots
match_colors = True #False



regex_500 = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
regex_600 = """TiNiSn_600C_Y20190219_14x14_t65_(?P<num>.*?)_bkgdSub_1D.csv"""

data_path ="/path/to/data/here/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
data_path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
#data_path = "/home/sasha/Desktop/TiNiSn_600C-20190607T173525Z-001/TiNiSn_600C/"

save_path = "C:/path/to/save/images/to"

excel_save_path = "C:/path/to/save/exel/to/" + "clustering-" + str(clusters) + ".xlsx"
excel_save_path = "/home/sasha/Desktop/" + "clustering-" + str(clusters) + ".xlsx"

#CHANGE IF FIGURE IS SAVED
save_figure = False #True
save_excel = True #True

dataGrid = DataGrid(data_path,regex_500)


# __________________________________________
# CODE
# __________________________________________

def similarity_vector(A,B):
    return np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)


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




def get_cluster_grids(i,prev_agg=None,prev_points=None,prev_hue=None):
    #cluster the data based on similarity metric
    agg = AgglomerativeClustering(n_clusters=i, compute_full_tree = True, affinity='precomputed',linkage='complete')
    agg.fit(D)

    #cluster colors
    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

    #make the clusters colors match up
    if not prev_agg==None and not prev_points==None and not prev_hue==None:
        dict = {}
        for x,l in enumerate(agg.labels_):
            dict[l] = prev_agg.labels_[x]
        cluster_split = [0,0]
        for index1,v1 in dict.items():
            for index2,v2 in dict.items():
                if not index1 == index2 and v1 ==v2:
                    cluster_split = [index1,index2]
        cluster_split.sort()

        hues_new = np.zeros(i)
        for x in range(i):
            if x in cluster_split:
                delta = float(cluster_split.index(x)*2-1)/float(i)
                hues_new[x] = prev_hue[dict[x]] + delta/2
            else:
                hues_new[x] = prev_hue[dict[x]]
        hues = hues_new

    #data grouped by cluster
    grouped_data = [[] for x in range(i)]
    for loc,val in enumerate(agg.labels_):
        grouped_data[val].append(dataGrid.data_at_loc(loc+1)[:,1])
    #average spectra for each cluster
    averages = [np.mean(x,axis=0) for x in grouped_data]


    #cluster grid to show where clusters fall
    cluster_grid = np.ones(shape = (15,15,3))
    #cluster grid with similarity to cluster average as brightness
    cluster_grid_scale = np.ones(shape = (15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        similarity = similarity_vector(dataGrid.data_at_loc(val)[:,1],averages[cluster])
        #adjusting the similarity gradient
        similarity = math.pow(similarity,power)
        cluster_grid_scale[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,similarity])
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

    # x, y locations of spectra closest to average
    points_x = [-1 for x in range(i)]
    points_y = [-1 for x in range(i)]
    # grid location of averages
    points_loc = [-1 for x in range(i)]

    #find grid locations closest to average
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


    return agg, cluster_grid, cluster_grid_scale, points_x, points_y, points_loc, agg.labels_, hues


fig = plt.figure(figsize=figsize,num=str(clusters))

agg, cg, cgs, px, py, pl, labels, hue  = get_cluster_grids(clusters)
if match_colors:
    _, cg_next, cgs_next, px_next, py_next, pl_next, _, _ = get_cluster_grids(clusters+1,agg,pl,hue)
else:
    _, cg_next, cgs_next, px_next, py_next, pl_next, _, _ = get_cluster_grids(clusters+1)

#flip point cooridinates to match grid image
px = [15-x for x in px]
py = [y-1 for y in py]
px_next = [15-x for x in px_next]
py_next = [y-1 for y in py_next]

# SHOW CLUSTER GRADIENT FOR X CLUSTERS
ax_similarity = fig.add_subplot(2,3,1)
ax_similarity.imshow(cgs)
#ax_similarity.scatter(px,py,s=3,c='black')
ax_similarity.invert_yaxis()
ax_similarity.axis("off")
ax_similarity.title.set_text(clusters)
for i,txt in enumerate(pl):
    ax_similarity.annotate(txt,(px[i]-.4,py[i]-.4),size=8)

# SHOW CLUSTER GRADIENT FOR X+1 CLUSTERS
ax_similarity_next = fig.add_subplot(2,3,2)
ax_similarity_next.imshow(cgs_next)
#ax_similarity_next.scatter(px_next,py_next,s=3,c='black')
ax_similarity_next.invert_yaxis()
ax_similarity_next.axis("off")
ax_similarity_next.title.set_text(clusters+1)
for i,txt in enumerate(pl_next):
    ax_similarity_next.annotate(txt,(px_next[i]-.4,py_next[i]-.4),size=8)

# SHOW CLUSTER NAMES FOR X CLUSTERS
ax_clusters = fig.add_subplot(2,3,3)
ax_clusters.imshow(cg)
ax_clusters.invert_yaxis()
ax_clusters.axis("off")
ax_clusters.title.set_text(clusters)
for i,loc in enumerate(pl):
    ax_clusters.annotate(chr(65 + labels[loc-1]),(px[i]-.4,py[i]-.4),size=8)

# SHOW GRID LABELS FOR X CLUSTERS
ax_locs = fig.add_subplot(2,3,4)
ax_locs.imshow(cg)
ax_locs.invert_yaxis()
ax_locs.axis("off")
for i in range(dataGrid.size):
    x,y = dataGrid.coord(i+1)
    ax_locs.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)

# SHOW GRID LABELS FOR X+1 CLUSTERS
ax_locs_next = fig.add_subplot(2,3,5)
ax_locs_next.imshow(cg_next)
ax_locs_next.invert_yaxis()
ax_locs_next.axis("off")
for i in range(dataGrid.size):
    x,y = dataGrid.coord(i+1)
    ax_locs_next.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)

# CALCULATE NEW CENTERS AFTER SPLIT
new_centers = list(set(pl_next).difference(set(pl)))
k1 = new_centers[0]
if len(new_centers) == 2:
    k2 = new_centers[1]
else:
    k2 = pl[labels[k1-1]]

# PLOT BOTH NEW CENTERS
ax_plot = fig.add_subplot(2,3,6)
x = dataGrid.data_at_loc(k1)[:,1]
y = dataGrid.data_at_loc(k1)[:,0]
ax_plot.plot(y,x,label=str(k1))
x = dataGrid.data_at_loc(k2)[:,1]
y = dataGrid.data_at_loc(k2)[:,0]
ax_plot.plot(y,x,label=str(k2))
ax_plot.legend(title=str(similarity(k1,k2)))

# PRINT CENTERS AND SIMILARITY TO CONSOLE
print(str(clusters) + " to " + str(clusters+1) + ": " + str(similarity(k1,k2)))
print("centers: " + str(k1) + " " + str(k2))
print()

fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=.1)
# SAVE FIGURE
if save_figure:
    plt.savefig(save_path + "clustering-" + str(clusters) + ".png")
    print()
    print("saved image to " + save_path)
    print()

# SAVE EXCEL
if save_excel:
    locations = list(range(1,dataGrid.size+1))
    clusters = [chr(65 + labels[loc-1]) for loc in locations]
    avg_sim = np.zeros(dataGrid.size)
    for loc in range(1,dataGrid.size+1):
        avg_sim[loc-1] = similarity(loc,pl[agg.labels_[loc-1]])
    df = DataFrame({'Grid Location': locations, 'Cluster': clusters, 'Similarity':avg_sim})
    df.to_excel(excel_save_path, sheet_name='sheet1', index=False)
    print()
    print("saved exel file to " + excel_save_path)
    print()

# SHOW FIGURE
plt.show()
