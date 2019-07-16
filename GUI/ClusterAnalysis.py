from data_grid import DataGrid

from sklearn.cluster import AgglomerativeClustering

from appJar import gui
import numpy as np
import matplotlib
import math

#dataDir = ""
#peakDir = ""
#curveDir = ""

dataGrid = None
peakGrid = None

power = 10

#TODO save cluster data
#cosineClusteringData = []


def cosineClustering(num_clusters):
    #cosine similarity function using two grid positions
    def similarity_vector(A,B):
        return np.dot(A,B)/np.linalg.norm(A)/np.linalg.norm(B)
    def similarity(d1,d2):
        a = dataGrid.data[d1][:,1]
        b = dataGrid.data[d2][:,1]
        return similarity_vector(a,b)

    D = np.ones(shape=(dataGrid.size,dataGrid.size))
    for x in range(dataGrid.size):
        for y in range(dataGrid.size):
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

    return (get_cluster_grids(num_clusters))





#fileMenus = ["Open", "Save", "Save as...", "-", "Close"]
#app.addMenuList("File", fileMenus, menuPress)
#app.createMenu("File")

#app.addStatusbar(fields=1)
#app.setStatusbar("Data Directory: " + str(dataDir), 0)


###############################################################################
###############################################################################
# START OF GUI CODE
###############################################################################
###############################################################################


#################################################
# GUI FUNCTION CALLS
#################################################

def loadData(button):
    global dataGrid,peakGrid
    if button == "Load Data":
        dataDir = app.getEntry("data")
        dataRegex = app.getEntry("Data Regex")
        peakDir = app.getEntry("peaks")
        peakRegex = app.getEntry("Peak Data Regex")

        dataGrid = DataGrid(dataDir,dataRegex)
        peakGrid = DataGrid(peakDir,peakRegex)



def runClustering(button):
    global dataGrid,peakGrid
    mode = app.getRadioButton("mode")
    grid = np.zeros(shape=(15,15))
    if mode == "Cosine Clustering":
        grid = cosineClustering(10)[1]
    else:
        print()
    ax_cluster.imshow(grid)
    ax_cluster.invert_yaxis()
    ax_cluster.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        ax_cluster.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)
    app.refreshPlot("cluster")



#################################################
# GUI LAYOUT
#################################################

# create a GUI variable called app
app = gui("Clustering Analysis","800x400")
app.addLabel("title", "Welcome to appJar")
app.setLabelBg("title", "red")


app.startPanedFrame("options", row=0, column=0)
app.setSticky("news")
app.setStretch("column")
app.addLabel("opt_title", "Data Options")
app.setLabelBg("opt_title", "grey")

app.addLabel("Background Subtracted 1D Data")
app.addDirectoryEntry("data")

app.addLabel("Peak Parameters")
app.addDirectoryEntry("peaks")

app.addLabel("regex_label","Custom Regex for Files")
app.setLabelBg("regex_label","grey")
app.addLabelEntry("Data Regex")
app.setEntry("Data Regex", ".*_(?P<num>.*?)_bkgdSub_1D.csv")
app.addLabelEntry("Peak Data Regex")
app.setEntry("Peak Data Regex", ".*_(?P<num>.*?)_bkg_curveParams.csv")

app.addButtons(["Load Data"],loadData)
app.stopPanedFrame()

#app.directoryBox(title=None, dirName=None, parent=None)

app.startFrame("all plots",row=0,column=1)

app.startFrame("plots",row=0,column=0)

app.startFrame("clustering_options",row=0,column=0)
app.setSticky("news")
app.setStretch("column")
app.addLabel("clustering_opt", "Clustering Options")
app.setLabelBg("clustering_opt", "grey")
app.addRadioButton("mode", "Cosine Clustering")
app.addRadioButton("mode", "Peak Reduction Clustering")
app.addButton("Cluster",runClustering)
app.setButtonBg("Cluster","red")
app.stopFrame()

app.startFrame("clustering_plot",row=0,column=1)
fig = app.addPlotFig("cluster")
ax_cluster = fig.add_subplot(1,1,1)
ax_cluster.imshow(np.zeros(shape=(15,15)))
app.refreshPlot("cluster")
app.stopFrame()

app.stopFrame()

app.startFrame("diffraction",row=1,column=0)
fig = app.addPlotFig("diff",showNav=True)
ax_plot = fig.add_subplot(1,1,1)
ax_plot.plot([i for i in range(10)],[np.random.random() for i in range(10)])
app.refreshPlot("diff")
app.stopFrame()

app.stopFrame()
app.go()
