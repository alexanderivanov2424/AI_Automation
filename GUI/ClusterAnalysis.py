from data_grid import DataGrid


from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from appJar import gui
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

#Data objects to store x-ray diffraction data
#Note: Upon loading new data old plots may be lost
dataGrid = None
peakGrid = None
curveGrid = None


#parameter used in cosing clustering for cosine clustering similarity plot
#used to scale similarity (easier to see gradient)
power = 10

#list of locations on the wafer selected to plot
#first location is for main window and the rest are the pop-ups
grid_location_list = [[]]
cluster_grid = None
num_clusters = 1



#used to increment pop-up windows
#each window needs unique "names" so this is used to keep track of them
win_number = 0

#list of axis objects for all the pop-up windows
#there is no way to access these from the window name (i think)
win_cluster_plot_list = []
win_cluster_fig_list = []
win_diff_plot_list = []

#Stores all clustering data for faster lookup
#keys are of the form (num_clusters,mode)
#values are tuple outputs of the respective mode methods
cluster_dict = {}


def cosineClustering(num_clusters):
    global cluster_dict
    if (num_clusters,"c") in cluster_dict.keys():
        return cluster_dict[(num_clusters,"c")]
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

    cluster_dict[(num_clusters,"c")] = (get_cluster_grids(num_clusters))
    return cluster_dict[(num_clusters,"c")]


def peakReductionClustering(num_clusters):
    global cluster_dict
    if (num_clusters,"p") in cluster_dict.keys():
        return cluster_dict[(num_clusters,"p")]
    #Create Point Cloud
    SCALE = 100
    def to_point(x,y,p):
        return [(x-1)/15.,(y-1)/15.,SCALE*float(p)/5]
    peaks = []
    for k in curveGrid.data.keys():
        x,y = curveGrid.coord(k)
        [peaks.append(to_point(x,y,p)) for p in curveGrid.data_at_loc(k)[:,2]]

    #Cluster Point Cloud
    X = np.array(peaks)
    clustering = DBSCAN(eps=0.25, min_samples=5).fit(X)
    C = len(set(clustering.labels_).difference(set([-1])))

    #reduce dimentions
    M = np.zeros(shape=(curveGrid.size,C))
    for k in curveGrid.data.keys():
        x,y = curveGrid.coord(k)
        V = np.zeros(shape=C)
        for i,p in enumerate(curveGrid.data_at_loc(k)[:,2]):
            loc = clustering.labels_[peaks.index(to_point(x,y,p))]
            if loc == -1:
                continue
            M[k-1,loc] = 1#peakGrid.data_at_loc(k)[i,3]

    pca = PCA(n_components = 'mle',svd_solver='full').fit_transform(M)
    #pca = PCA(n_components = 20,svd_solver='full').fit_transform(M)

    def get_cluster_grids(i):
        agg = AgglomerativeClustering(n_clusters=i).fit(pca)

        hues = [float(float(x)/float(i)) for x in range(1,i+1)]

        cluster_centers = np.zeros(i)
        #sum vectors in each cluster
        cluster_sums = np.zeros(shape=(i,len(pca[0])))
        for val in range(1,178):
            cluster = agg.labels_[val-1]
            cluster_sums[cluster] = cluster_sums[cluster] + pca[0]

        #divide by cluster size to get average point
        for cluster in range(0,i):
            count = np.count_nonzero(agg.labels_==cluster)
            cluster_sums[cluster] = cluster_sums[cluster] / count

        for cluster,center_v in enumerate(cluster_sums):
            cluster_v = pca[np.where(agg.labels_==cluster)[0]]
            index = np.argmin(np.sum(np.square(cluster_v - center_v),axis=1))
            cluster_centers[cluster] = np.where(agg.labels_==cluster)[0][index] + 1

        cluster_grid = np.zeros(shape = (15,15,3))
        for val in range(1,178):
            x,y = dataGrid.coord(val)
            cluster = agg.labels_[val-1]
            cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])


        peak_max_counts = np.zeros(i)
        for val in range(1,178):
            cluster = agg.labels_[val-1]
            peak_max_counts[cluster] = max(peak_max_counts[cluster],len(curveGrid.data_at_loc(val)[:,1]))

        peak_grid = np.zeros(shape =(15,15,3))
        for val in range(1,178):
            x,y = dataGrid.coord(val)
            cluster = agg.labels_[val-1]
            k = len(curveGrid.data_at_loc(val)[:,1])/peak_max_counts[cluster]
            peak_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([1,1,k])

        width_max = 0
        for val in range(1,178):
            width_max = max(width_max,np.nanmax(curveGrid.data_at_loc(val)[:,2].astype(np.float)))

        width_grid = np.zeros(shape =(15,15))
        width_grid.fill(np.nan)
        for val in range(1,178):
            x,y = dataGrid.coord(val)
            cluster = agg.labels_[val-1]
            k = np.nanmax(curveGrid.data_at_loc(val)[:,2].astype(np.float))
            width_grid[y-1][15-x] = k

        return cluster_grid, peak_grid, width_grid, cluster_centers,agg.labels_

    cluster_dict[(num_clusters,"p")] = (get_cluster_grids(num_clusters))
    return cluster_dict[(num_clusters,"p")]




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
    global dataGrid,peakGrid, curveGrid
    if button == "Load Data":
        dataDir = app.getEntry("data")
        dataRegex = app.getEntry("Data Regex")
        peakDir = app.getEntry("peaks")
        peakRegex = app.getEntry("Peak Data Regex")
        curveRegex = app.getEntry("Curve Data Regex")

        dataDir = "/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C"
        peakDir = "/home/sasha/Desktop/peakTest"

        try:
            dataGrid = DataGrid(dataDir,dataRegex)
        except RuntimeError:
            print("## WARNING ##")
            print(RuntimeError("Missing Data Files"))
            print()
        try:
            peakGrid = DataGrid(peakDir,peakRegex)
        except RuntimeError:
            print("## WARNING ##")
            print(RuntimeError("Missing Peak Param Files"))
            print()
        try:
            curveGrid = DataGrid(peakDir,curveRegex)
        except RuntimeError:
            print("## WARNING ##")
            print(RuntimeError("Missing Curve Param Files"))
            print()

def sendToWindow(button):
    global cluster_grid,num_clusters, win_number
    name = "Window Number " + str(win_number)
    win = str(win_number)
    grid_location_list.append([])
    app.startSubWindow(win + "clustering window",title=name,modal=False,transient=False, blocking=False)
    app.setSize(800, 600)
    app.setSticky("news")
    app.startFrame(win+"all plots",row=0,column=0)
    app.startFrame(win+"clustering_options",row=0,column=0)
    app.setSticky("news")
    app.setStretch("column")
    app.addLabel(win+"clustering_opt", "Clustering Options")
    app.setLabelBg(win+"clustering_opt", "grey")
    app.addRadioButton(win+"mode", "Cosine Clustering")
    app.addRadioButton(win+"mode", "Peak Reduction Clustering")
    app.setRadioButtonChangeFunction(win+"mode",lambda x : switchModeOption(x,win))

    #SLIDER
    app.addScale(win+"num_clusters")
    app.setScaleRange(win+"num_clusters", 1, 30, curr=1)
    app.setScaleIncrement(win+"num_clusters", 1)
    app.showScaleIntervals(win+"num_clusters", 5)
    app.showScaleValue(win+"num_clusters",show=True)
    app.setScaleChangeFunction(win+"num_clusters",lambda x : runClustering(x,win))

    app.addNamedButton("Cluster",win+"Cluster",lambda x : runClustering(x,win))
    app.setButtonBg(win+"Cluster","red")

    app.startFrame(win+"cosine options",row=6,column=0)
    app.addLabel(win+"cosine_options","Additional Options")
    app.setLabelBg(win+"cosine_options","grey")
    app.addRadioButton(win+"cos_mode", "Cosine Clustering")
    app.addRadioButton(win+"cos_mode", "Cluster Similarity")
    app.addRadioButton(win+"cos_mode", "Cluster Centers")
    app.stopFrame()


    app.startFrame(win+"peak options",row=6,column=0)
    app.addLabel(win+"peak_options","Additional Options")
    app.setLabelBg(win+"peak_options","grey")
    app.addRadioButton(win+"peak_mode", "Peak Reduction Clustering")
    app.addRadioButton(win+"peak_mode", "Peak Counts")
    app.addRadioButton(win+"peak_mode", "Peak FWHM")
    app.stopFrame()
    app.hideFrame(win+"peak options")

    app.stopFrame()

    app.startFrame(win+"clustering_plot",row=0,column=1)
    win_fig_cluster = app.addPlotFig(win+"cluster",showNav=True)
    location_select = win_fig_cluster.canvas.mpl_connect('button_press_event', lambda x : selectClusterLocation(x,win))
    win_ax_cluster = win_fig_cluster.add_subplot(1,1,1)
    win_ax_cluster.imshow(np.zeros(shape=(15,15)))
    win_cluster_plot_list.append(win_ax_cluster)
    win_cluster_fig_list.append(win_fig_cluster)
    app.refreshPlot(win+"cluster")
    app.stopFrame()

    app.stopFrame()

    app.startFrame(win+"diffraction",row=1,column=0)
    win_fig_plot = app.addPlotFig(win+"diff",showNav=True)
    win_ax_plot = win_fig_plot.add_subplot(1,1,1)
    win_diff_plot_list.append(win_ax_plot)
    app.refreshPlot(win+"diff")
    app.stopFrame()

    app.stopSubWindow()
    app.showSubWindow(win+"clustering window",hide = False)
    win_number+=1

def runClustering(button,win):
    global cluster_grid,num_clusters

    num_clusters = app.getScale(win+"num_clusters")
    mode = app.getRadioButton(win+"mode")
    #sub_mode = app.getRadioButton(win+)
    if mode == "Cosine Clustering":
        cluster_grid = cosineClustering(num_clusters)[1]
    else:
        cluster_grid = peakReductionClustering(num_clusters)[0]
    plotClustering(win)

def plotClustering(win):
    global cluster_grid, cluster_dict
    if win == "":
        ax = ax_cluster#main window
        fig = fig_cluster
        locs = grid_location_list[0]
    else:
        ax = win_cluster_plot_list[int(win)]
        fig = win_cluster_fig_list[int(win)]
        locs = grid_location_list[int(win)+1]

    #Adjust for options:
    justCenters = False
    centers = []
    num_clusters = app.getScale(win+"num_clusters")

    #detect the clustering mode and read additional options accordingly
    mode = app.getRadioButton(win+"mode")
    if mode == "Cosine Clustering":
        button = app.getRadioButton(win+"cos_mode")
    else:
        button = app.getRadioButton(win+"peak_mode")

    if button == "Cosine Clustering":
        cluster_grid = cluster_dict[(num_clusters,"c")][1]
        plotCosineClustering(win,ax,locs)
    elif button == "Cluster Similarity":
        cluster_grid = cluster_dict[(num_clusters,"c")][2]
        plotCosineClusterSimilarity(win,ax,locs)
    elif button == "Cluster Centers":
        cluster_grid = cluster_dict[(num_clusters,"c")][1]
        centers =  cluster_dict[(num_clusters,"c")][5]
        plotCosineClusterCenters(win,ax,locs,centers)
    elif button == "Peak Reduction Clustering":
        cluster_grid = cluster_dict[(num_clusters,"p")][0]
        plotPeakReductionClustering(win,ax,locs)
    elif button == "Peak Counts":
        cluster_grid = cluster_dict[(num_clusters,"p")][1]
        peakCountFun = lambda loc : len(curveGrid.data_at_loc(loc)[:,0])
        plotPeakReductionCounts(win,ax,locs,peakCountFun)
    elif button == "Peak FWHM":
        cluster_grid = cluster_dict[(num_clusters,"p")][2]
        plotPeakReductionFWHM(win,fig,ax,locs)
    else:
        pass

    #Remove color bar
    if not button == "Peak FWHM" and len(fig.axes) > 1:
        fig.axes[-1].remove()

"""
Plotting Cosine Clustering
"""
def plotCosineClustering(win,ax,locs):
    global cluster_grid
    ax.cla()
    ax.imshow(cluster_grid)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        if (i+1) in locs:
            ax.scatter(15-x-.1,y-1-.1,marker='o',s=70,color="white")
            ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=8)
        else:
            ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)
    app.refreshPlot(win+"cluster")

def plotCosineClusterSimilarity(win,ax,locs):
    global cluster_grid
    ax.cla()
    ax.imshow(cluster_grid)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        if (i+1) in locs:
            ax.scatter(15-x-.1,y-1-.1,marker='o',s=70,color="white")
            ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=8)
        else:
            ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)
    app.refreshPlot(win+"cluster")

def plotCosineClusterCenters(win,ax,locs,centers):
    global cluster_grid

    ax.cla()
    ax.imshow(cluster_grid)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        if (i+1) in locs:
            ax.scatter(15-x-.1,y-1-.1,marker='o',s=70,color="white")
            if i+1 in centers:
                ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=8)
        else:
            if i+1 in centers:
                ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)
    app.refreshPlot(win+"cluster")

def plotPeakReductionClustering(win,ax,locs):
    global cluster_grid
    ax.cla()
    ax.imshow(cluster_grid)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        if (i+1) in locs:
            ax.scatter(15-x-.1,y-1-.1,marker='o',s=70,color="white")
            ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=8)
        else:
            ax.annotate(str(i+1),(15-x-.4,y-1-.4),size=6)
    app.refreshPlot(win+"cluster")

def plotPeakReductionCounts(win,ax,locs,peakCountFun):
    global cluster_grid
    ax.cla()
    ax.imshow(cluster_grid)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        if (i+1) in locs:
            ax.scatter(15-x-.1,y-1-.1,marker='o',s=70,color="white")
            ax.annotate(str(peakCountFun(i+1)),(15-x-.4,y-1-.4),size=8)
        else:
            ax.annotate(str(peakCountFun(i+1)),(15-x-.4,y-1-.4),size=8)
    app.refreshPlot(win+"cluster")

def plotPeakReductionFWHM(win,fig,ax,locs):
    global cluster_grid
    ax.cla()
    #fig.clf()
    ax.invert_yaxis()
    heatmap = ax.pcolor(np.flip(cluster_grid,axis=0),cmap="viridis_r")
    if len(fig.axes) > 1:
        pts = fig.axes[-1].get_position().get_points()
        label = fig.axes[-1].get_ylabel()
        fig.axes[-1].remove()
        cax= fig.add_axes([pts[0][0],pts[0][1],pts[1][0]-pts[0][0],pts[1][1]-pts[0][1]  ])
        cbar = plt.colorbar(heatmap, cax=cax)
        cbar.ax.set_ylabel(label)
    else:
        fig.colorbar(heatmap, ax=ax)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(dataGrid.size):
        x,y = dataGrid.coord(i+1)
        if (i+1) in locs:
            ax.scatter(15-x-.1,y-1-.1,marker='o',s=70,color="white")
            ax.annotate(str(i+1),(15-x+.2,y-1+.2),size=8)
        else:
            ax.annotate(str(i+1),(15-x+.2,y-1+.2),size=8)
    app.refreshPlot(win+"cluster")

def selectClusterLocation(event,win):
    global grid_location_list

    if win == "":
        ax = ax_plot#main window
        locs = grid_location_list[0]
    else:
        ax = win_diff_plot_list[int(win)]
        locs = grid_location_list[int(win)+1]

    print(event.xdata,event.ydata)
    loc = dataGrid.grid_num(int(15-round(event.xdata)),int(round(event.ydata)+1))
    if loc in locs:
        locs.remove(loc)
    else:
        locs.append(loc)

    #Update cluster plot
    plotClustering(win)

    ax.cla()
    for loc in locs:
        X = dataGrid.data_at_loc(loc)[:,0]
        Y = dataGrid.data_at_loc(loc)[:,1]
        ax.plot(X,Y,label=str(loc))
    ax.legend()
    app.refreshPlot(win+"diff")

def switchModeOption(mode,win):
    print(win)
    mode = app.getRadioButton(win+"mode")
    if mode == "Cosine Clustering":
        app.showFrame(win+"cosine options")
        app.hideFrame(win+"peak options")
    else:
        app.hideFrame(win+"cosine options")
        app.showFrame(win+"peak options")


#################################################
# GUI LAYOUT
#################################################

# create a GUI variable called app
app = gui("Clustering Analysis","1200x600")
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
app.setEntry("Peak Data Regex", ".*_(?P<num>.*?)_bkgdSu_peakParams.csv")
app.addLabelEntry("Curve Data Regex")
app.setEntry("Curve Data Regex", ".*_(?P<num>.*?)_bkg_curveParams.csv")

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
app.setRadioButtonChangeFunction("mode",lambda x : switchModeOption(x,""))
#SLIDER
app.addScale("num_clusters")
app.setScaleRange("num_clusters", 1, 30, curr=1)
app.setScaleIncrement("num_clusters", 1)
app.showScaleIntervals("num_clusters", 5)
app.showScaleValue("num_clusters",show=True)
app.setScaleChangeFunction("num_clusters",lambda x : runClustering(x,""))

app.addButton("Cluster",lambda x : runClustering(x,""))
app.setButtonBg("Cluster","red")

app.addButton("Send To Window",sendToWindow)

app.startFrame("cosine options",row=6,column=0)
app.addLabel("cosine_options","Additional Options")
app.setLabelBg("cosine_options","grey")
app.addRadioButton("cos_mode", "Cosine Clustering")
app.addRadioButton("cos_mode", "Cluster Similarity")
app.addRadioButton("cos_mode", "Cluster Centers")
app.stopFrame()


app.startFrame("peak options",row=6,column=0)
app.addLabel("peak_options","Additional Options")
app.setLabelBg("peak_options","grey")
app.addRadioButton("peak_mode", "Peak Reduction Clustering")
app.addRadioButton("peak_mode", "Peak Counts")
app.addRadioButton("peak_mode", "Peak FWHM")
app.stopFrame()
app.hideFrame("peak options")

app.stopFrame()

app.startFrame("clustering_plot",row=0,column=1)
fig_cluster = app.addPlotFig("cluster",showNav=True)
location_select = fig_cluster.canvas.mpl_connect('button_press_event', lambda x : selectClusterLocation(x,""))
ax_cluster = fig_cluster.add_subplot(1,1,1)
ax_cluster.imshow(np.zeros(shape=(15,15)))
ax_cluster.axis("off")
app.refreshPlot("cluster")
app.stopFrame()

app.stopFrame()

app.startFrame("diffraction",row=1,column=0)
fig_plot = app.addPlotFig("diff",showNav=True)
ax_plot = fig_plot.add_subplot(1,1,1)
app.refreshPlot("diff")
app.stopFrame()

app.stopFrame()
app.go()
