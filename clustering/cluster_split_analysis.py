
from data_loading.data_grid import DataGrid


from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math



regex_500 = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
regex_600 = """TiNiSn_600C_Y20190219_14x14_t65_(?P<num>.*?)_bkgdSub_1D.csv"""

#CHANGE THIS
data_path ="/path/to/data/here/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
data_path = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"
#data_path = "/home/sasha/Desktop/TiNiSn_600C-20190607T173525Z-001/TiNiSn_600C/"

save_path = "C:/path/to/save/images/to"


dataGrid = DataGrid(data_path,regex_500)


# PLOT X clusters going to (X+1) clusters
clusters = 8

#Adjust the similarity gradient
power = 10

#Adjust the size of the figure
figsize = (14,8)

'''
for k in dataGrid.data.keys():
    x = dataGrid.data[k][:,1]
    plt.plot(x)
    peaks,_ = find_peaks(x)
    plt.plot(peaks,x[peaks], "x")
    plt.show()


x = dataGrid.data[32][:,1]
y = dataGrid.data[32][:,0]
plt.plot(y,x)
#peaks,_ = find_peaks(x)
#plt.plot(peaks,x[peaks], "x")
x = dataGrid.data[34][:,1]
y = dataGrid.data[34][:,0]
plt.plot(y,x)
#peaks,_ = find_peaks(x)
#plt.plot(peaks,x[peaks], "x")
plt.show()

'''

def similarity_vector(A,B):
    pA, _ = find_peaks(A)
    pB, _ = find_peaks(B)
    p = np.append(pA,pB,axis=0)
    cosine =  np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    peaks = np.dot(A[p],B[p])/np.linalg.norm(A[p])/np.linalg.norm(B[p])
    return peaks

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



def get_cluster_grids(i):
    #cluster the data based on similarity metric
    agg = AgglomerativeClustering(n_clusters=i, compute_full_tree = True, affinity='precomputed',linkage='complete')
    agg.fit(D)

    #cluster colors
    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

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


    return cluster_grid, cluster_grid_scale, points_x, points_y, points_loc, agg.labels_


fig = plt.figure(figsize=figsize,num=str(clusters))

cg, cgs, px, py, pl, labels = get_cluster_grids(clusters)
cg_next, cgs_next, px_next, py_next, pl_next, _ = get_cluster_grids(clusters+1)

#flip point cooridinates to match grid image
px = [15-x for x in px]
py = [y-1 for y in py]
px_next = [15-x for x in px_next]
py_next = [y-1 for y in py_next]

ax_similarity = fig.add_subplot(3,3,1)
ax_similarity.imshow(cgs)
#ax_similarity.scatter(px,py,s=3,c='black')
ax_similarity.invert_yaxis()
ax_similarity.axis("off")
for i,txt in enumerate(pl):
    ax_similarity.annotate(txt,(px[i]-.4,py[i]-.4),size=8)

ax_similarity_next = fig.add_subplot(3,3,2)
ax_similarity_next.imshow(cgs_next)
#ax_similarity_next.scatter(px_next,py_next,s=3,c='black')
ax_similarity_next.invert_yaxis()
ax_similarity_next.axis("off")
for i,txt in enumerate(pl_next):
    ax_similarity_next.annotate(txt,(px_next[i]-.4,py_next[i]-.4),size=8)

ax_clusters = fig.add_subplot(3,3,3)
ax_clusters.imshow(cg)
ax_clusters.invert_yaxis()
ax_clusters.axis("off")
ax_clusters.title.set_text(clusters)
for i,loc in enumerate(pl):
    ax_clusters.annotate(chr(65 + labels[loc-1]),(px[i]-.4,py[i]-.4),size=8)

ax_locs = fig.add_subplot(3,3,4)
ax_locs.imshow(cg)
ax_locs.invert_yaxis()
ax_locs.axis("off")
for i in range(dataGrid.size):
    x,y = dataGrid.coord(i+1)
    ax_locs.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)

ax_locs_next = fig.add_subplot(3,3,5)
ax_locs_next.imshow(cg_next)
ax_locs_next.invert_yaxis()
ax_locs_next.axis("off")
for i in range(dataGrid.size):
    x,y = dataGrid.coord(i+1)
    ax_locs_next.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)


new_centers = list(set(pl_next).difference(set(pl)))
k1 = new_centers[0]
if len(new_centers) == 2:
    k2 = new_centers[1]
else:
    k2 = pl[labels[k1-1]]

ax_plot = fig.add_subplot(3,1,3)
x = dataGrid.data_at_loc(k1)[:,1]
y = dataGrid.data_at_loc(k1)[:,0]
ax_plot.plot(y,x,label=str(k1))
x = dataGrid.data_at_loc(k2)[:,1]
y = dataGrid.data_at_loc(k2)[:,0]
ax_plot.plot(y,x,label=str(k2))
ax_plot.legend(title=str(similarity(k1,k2)))
'''
x = dataGrid.data_at_loc(k1)[:,1]
ax[0,1].plot(x,label=str(k1))
peaks,_ = find_peaks(x)
#ax[0,1].plot(peaks,x[peaks], "x")

x = dataGrid.data_at_loc(k2)[:,1]
ax[0,1].plot(x,label=str(k2))
peaks,_ = find_peaks(x)
#ax[0,1].plot(peaks,x[peaks], "x")
ax[0,1].legend(title=str(similarity(k1,k2)))
'''
print(str(clusters) + " to " + str(clusters+1) + ": " + str(similarity(k1,k2)))
print("centers: " + str(k1) + " " + str(k2))
print()

fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=.1)
#k = .01
#plt.subplots_adjust(left=.1, right=1-k,bottom=k, top=1-k, wspace=k, hspace=k)
#plt.savefig(save_path + "clustering-" + str(clusters) + ".png")
plt.show()
